"""Stage II continual adaptation with explicit trainable module scopes."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from dma_kws.pathing import resolve_dict_path
from dma_kws.phonemes import normalize_english_text
from dma_kws.stage2.adapt_config import (
    adapt_method_label,
    is_full_adapt_method,
    resolve_adapt_method,
)
from dma_kws.stage2.adapt_dataset import (
    KeywordAdaptationDataset,
    MixedAdaptationDataset,
    TargetKeywordValDataset,
)
from dma_kws.stage2.adapt_paths import adapt_data_root, adapt_exp_root, phase_manifest, slugify
from dma_kws.stage2.encoder_schedule import EncoderFinetuneSchedule
from dma_kws.stage2.module import Stage2LightningModule
from dma_kws.stage2.objective import (
    assert_sequence_objective_matches,
    resolve_sequence_objective,
)
from dma_kws.stage2.train import _build_val_dataloader, _resolve_path
from dma_kws.training.adapt_params import (
    load_adapt_params_file,
    merge_adapt_params,
    resolve_adapt_lr,
    resolve_encoder_adapt_lr,
)
from dma_kws.training.checkpoint_io import (
    STAGE2_BASE_FINGERPRINT_KEY,
    assert_qbyt_readout_version,
    assert_stream_policy_matches,
    extract_state_dict,
    fingerprint_stage2_base,
    restore_best_checkpoint_weights,
    stamp_qbyt_readout_version,
)
from dma_kws.training.distributed_metrics import sum_across_processes
from dma_kws.training.ddp import process_rank
from dma_kws.training.lora import (
    count_lora_params,
    inject_qbyt_lora,
    load_lora_state_dict,
    lora_state_dict,
    lora_targets_for_qbyt_family,
    merge_lora,
    normalize_lora_targets,
)
from dma_kws.training.score_diagnostics import BinaryScoreDiagnostics
from dma_kws.training.random_state import (
    capture_rank_random_states,
    restore_rank_random_states,
    validate_rank_random_states,
)


@dataclass
class Stage2AdaptArgs:
    """Runtime options for Stage II adaptation."""

    init_checkpoint: str = ""
    resume_checkpoint: str = ""
    resume_from: str = ""
    device: str = "cuda"
    devices: int = 1
    limit_steps: int = 0
    params_file: str = ""


def _adapt_section(config: dict[str, Any]) -> dict[str, Any]:
    adapt = config.get("adapt")
    if not isinstance(adapt, dict):
        raise ValueError("Config section 'adapt' must be a mapping")
    return adapt


def _resolve_adapt_paths(config: dict[str, Any]) -> dict[str, Path]:
    adapt = _adapt_section(config)
    keyword = str(adapt.get("keyword", ""))
    if not keyword:
        raise ValueError("adapt.keyword is required")

    slug = str(adapt.get("slug", "")) or slugify(keyword)
    data_root = Path(adapt.get("data_root", "")) if adapt.get("data_root") else adapt_data_root(
        config, keyword
    )
    exp_root = adapt_exp_root(config, keyword)
    phase = str(adapt.get("phase", "tts"))

    return {
        "keyword": Path(keyword),  # type: ignore[dict-item]
        "slug": Path(slug),  # type: ignore[dict-item]
        "keyword_str": keyword,
        "slug_str": slug,
        "data_root": data_root,
        "fbank_root": data_root / "fbank",
        "exp_root": exp_root,
        "phase": Path(phase),  # type: ignore[dict-item]
        "phase_str": phase,
        "phase_dir": exp_root / phase,
        "merged_path": exp_root / "stage2_adapted.pt",
        "train_manifest": phase_manifest(data_root, phase, split="train"),
        "eval_manifest": phase_manifest(data_root, phase, split="eval"),
    }


def _atomic_torch_save(payload: Any, destination: Path) -> Path:
    """Publish a torch payload without exposing a partially written alias."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def _apply_adapt_overrides(config: dict[str, Any], args: Stage2AdaptArgs) -> dict[str, Any]:
    """Merge ``adapt.params_file`` overrides (e.g. sweep best params) into the config."""
    if not args.params_file:
        return {}
    return merge_adapt_params(
        _adapt_section(config),
        load_adapt_params_file(args.params_file),
    )


def _validate_lora_runtime_config(stage2: dict[str, Any], *, method: str = "lora") -> None:
    label = "LoRA" if method == "lora" else method
    noise_augmentation = stage2.get("noise_augmentation", {}) or {}
    if bool(noise_augmentation.get("enabled", False)):
        raise ValueError(
            f"stage2.noise_augmentation.enabled is not supported for {label} "
            "adaptation: adaptation manifests contain precomputed features, not "
            "the source waveforms required for online mixing. Disable it for "
            "adaptation or run base Stage II QbyT training."
        )

    ema = stage2.get("ema", {}) or {}
    if bool(ema.get("enabled", False)):
        raise ValueError(
            f"stage2.ema.enabled is not supported for {label} adaptation: Lightning "
            "weight averaging is not part of adaptation checkpoint selection and "
            "can update frozen modules. Keep EMA disabled."
        )

    precision = str(stage2.get("precision", "") or "").lower()
    if precision.endswith("-true") and precision != "32-true":
        raise ValueError(
            f"stage2.precision={precision!r} changes frozen base parameter dtypes "
            "after adapter identity is checked. Use 'bf16-mixed', '16-mixed', or "
            f"'32-true' for {label} adaptation."
        )


def _metadata_candidates(
    checkpoint: dict[str, Any],
    key: str,
) -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = []
    if checkpoint.get(key) is not None:
        candidates.append((f"checkpoint.{key}", checkpoint[key]))
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        return candidates
    adapt = config.get("adapt")
    if isinstance(adapt, dict) and adapt.get(key) is not None:
        value = adapt[key]
        # The resolved default uses slug="" to mean "derive it from keyword".
        if key != "slug" or str(value).strip():
            candidates.append((f"config.adapt.{key}", value))
    return candidates


def _normalize_lora_metadata(key: str, value: Any) -> Any:
    if key == "keyword":
        return normalize_english_text(str(value))
    if key == "slug":
        return slugify(str(value))
    if key in {"checkpoint_kind", "phase", STAGE2_BASE_FINGERPRINT_KEY}:
        return str(value)
    if key == "rank":
        if isinstance(value, bool):
            raise ValueError("boolean rank")
        parsed = int(value)
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("fractional rank")
        if isinstance(value, str) and str(parsed) != value.strip().lstrip("+"):
            raise ValueError("non-integer rank")
        if parsed <= 0:
            raise ValueError("non-positive rank")
        return parsed
    if key == "alpha":
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError("non-positive or non-finite alpha")
        return parsed
    if key == "lora_targets":
        values = [value] if isinstance(value, str) else value
        try:
            return frozenset(
                normalize_lora_targets(values, family="keyword_filler")
            )
        except ValueError:
            return frozenset(normalize_lora_targets(values, family="pooling"))
    return value


def _validate_lora_metadata(
    checkpoint: dict[str, Any],
    *,
    source: str | Path,
    expected: dict[str, Any],
    required: tuple[str, ...],
    label: str = "LoRA",
) -> None:
    source_label = str(source)
    missing = [key for key in required if not _metadata_candidates(checkpoint, key)]
    if missing:
        raise SystemExit(
            f"Cannot resume {label} checkpoint {source_label}: required metadata is "
            f"missing: {', '.join(missing)}."
        )

    for key, expected_value in expected.items():
        candidates = _metadata_candidates(checkpoint, key)
        if not candidates:
            continue
        try:
            normalized_expected = _normalize_lora_metadata(key, expected_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid expected {label} {key}: {expected_value!r}") from exc
        for location, saved_value in candidates:
            try:
                normalized_saved = _normalize_lora_metadata(key, saved_value)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Cannot resume {label} checkpoint {source_label}: invalid "
                    f"{key}={saved_value!r} in {location}."
                ) from exc
            matches = normalized_saved == normalized_expected
            if key == "alpha":
                matches = math.isclose(
                    normalized_saved,
                    normalized_expected,
                    rel_tol=1e-12,
                    abs_tol=0.0,
                )
            if not matches:
                raise SystemExit(
                    f"Cannot resume {label} checkpoint {source_label}: "
                    f"{key}={saved_value!r} in {location}, expected "
                    f"{expected_value!r}."
                )


def _validate_base_fingerprint(
    checkpoint: dict[str, Any],
    *,
    source: str | Path,
    expected: str,
) -> None:
    candidates = _metadata_candidates(checkpoint, STAGE2_BASE_FINGERPRINT_KEY)
    if not candidates:
        message = (
            f"LoRA checkpoint {source} has no {STAGE2_BASE_FINGERPRINT_KEY}; "
            "its frozen Stage II base identity cannot be verified."
        )
        raise SystemExit(message)
    _validate_lora_metadata(
        checkpoint,
        source=source,
        expected={STAGE2_BASE_FINGERPRINT_KEY: expected},
        required=(STAGE2_BASE_FINGERPRINT_KEY,),
    )


def _validate_adapter_checkpoint(
    checkpoint: dict[str, Any],
    *,
    source: str | Path,
    keyword: str,
    rank: int,
    alpha: float,
    targets: tuple[str, ...],
    base_model_sha256: str | None = None,
) -> dict[str, torch.Tensor]:
    """Reject adapters trained for a different keyword or LoRA parametrization."""
    source_label = str(source)
    saved_kind = checkpoint.get("checkpoint_kind")
    if saved_kind is not None and str(saved_kind) != "stage2_lora_adapter":
        raise SystemExit(
            f"Cannot resume LoRA adapter {source_label}: "
            f"checkpoint_kind={saved_kind!r}, expected 'stage2_lora_adapter'."
        )
    _validate_lora_metadata(
        checkpoint,
        source=source,
        expected={
            "keyword": keyword,
            "rank": rank,
            "alpha": alpha,
            "lora_targets": targets,
        },
        required=("keyword", "rank", "alpha", "lora_targets"),
    )
    if base_model_sha256 is not None:
        _validate_base_fingerprint(
            checkpoint,
            source=source,
            expected=base_model_sha256,
        )

    adapter_state = checkpoint.get("lora_state_dict")
    if not isinstance(adapter_state, dict) or not adapter_state:
        raise SystemExit(
            f"Cannot resume LoRA adapter {source_label}: lora_state_dict is missing or empty."
        )
    return adapter_state


def grouped_utt_bce_totals(
    logits: torch.Tensor,
    labels: torch.Tensor,
    source: torch.Tensor,
    utt_sample_mask: torch.Tensor,
) -> torch.Tensor:
    """Local (keyword_loss_sum, keyword_count, lph_loss_sum, lph_count) as float64.

    Callers reduce this tensor across ranks before dividing. Empty groups have a
    zero count so the logger can emit NaN rather than a mean of no samples.
    """
    valid_mask = utt_sample_mask.bool()
    keyword_mask = source.bool() & valid_mask
    lph_mask = ~source.bool() & valid_mask
    per_sample = F.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none"
    )
    return torch.stack(
        (
            per_sample.detach()[keyword_mask].sum(),
            keyword_mask.sum().to(per_sample),
            per_sample.detach()[lph_mask].sum(),
            lph_mask.sum().to(per_sample),
        )
    ).to(dtype=torch.float64)


def grouped_domain_bce_totals(logits, labels, domains, valid_mask):
    """Per domain: BCE sum, valid count, all draws, positive draws.

    Keep the legacy keyword/replay flag separate: a MUSAN sample paired with
    the target keyword is still a MUSAN audio source.
    """
    per_sample = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    rows = []
    for domain in range(4):
        selected = domains == domain
        valid = selected & valid_mask.bool()
        rows.append(torch.stack((
            per_sample.detach()[valid].sum(), valid.sum().to(per_sample),
            selected.sum().to(per_sample), (selected & labels.bool()).sum().to(per_sample),
        )))
    return torch.stack(rows).to(dtype=torch.float64)


def _joint_data_signature(
    config: dict, manifests: dict[str, Path], *, parquet_file: Path, wav_dir: Path, dict_path: Path,
) -> str:
    """Bind full-state resumes to the same data and sampling policy."""
    adapt, stage2 = config["adapt"], config["stage2"]
    bg = stage2.get("background_negative", {}) or {}
    files = {**manifests, "libri_parquet": Path(parquet_file), "tokenizer": Path(dict_path)}
    for key, value in (
        ("background_train", bg.get("audio_list_path")),
        ("background_cache", bg.get("cache_manifest")),
        ("background_eval", (adapt.get("joint") or {}).get("background_eval_list")),
    ):
        if value:
            files[key] = Path(value)
    digests = {}
    for key, path in files.items():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digests[key] = {"path": str(path.resolve()), "sha256": digest.hexdigest()}
    payload = {
        "files": digests,
        "joint": adapt.get("joint", {}),
        "mix_ratio": adapt.get("mix_ratio", 0.5),
        "seed": config.get("training", {}).get("seed", 2025),
        "sample_lens": adapt.get("sample_lens"),
        "background": bg,
        "fbank": config.get("fbank"),
        "wav_dir": str(Path(wav_dir).resolve()),
        "keyword": adapt.get("keyword"),
        "replay": {key: stage2.get(key) for key in
                   ("parquet_file", "wav_dir", "negative_ratio", "hard_negative_ratio")},
        "accumulate_grad_batches": stage2.get("accumulate_grad_batches", 1),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _validate_adapt_checkpoint_method(checkpoint: dict[str, Any], *, expected: str) -> None:
    """A historical LoRA checkpoint has no method field; its kind is sufficient."""
    kind = str(checkpoint.get("checkpoint_kind", ""))
    inferred = {
        "stage2_lora": "lora",
        "stage2_qbyt_full": "qbyt_full",
        "stage2_encoder_qbyt_full": "encoder_qbyt_full",
    }.get(kind)
    candidates = _metadata_candidates(checkpoint, "method")
    if inferred is not None:
        candidates.append(("checkpoint.checkpoint_kind", inferred))
    for location, saved in candidates:
        if saved != expected:
            raise SystemExit(
                f"Cannot resume adapt.method={expected!r} from {saved!r} in {location}. "
                "To change method, start a new run from complete model weights "
                "with run.init_checkpoint; merge LoRA first."
            )


def _validate_qbyt_full_initialization(
    checkpoint: dict[str, Any], *, source: str | Path, method: str = "qbyt_full",
) -> None:
    if not isinstance(checkpoint, dict):
        raise SystemExit(
            f"Cannot initialize {method} from {source}: expected a complete Stage II checkpoint."
        )
    state = extract_state_dict(checkpoint)
    unmerged = (
        checkpoint.get("checkpoint_kind") in {"stage2_lora", "stage2_lora_adapter"}
        or "lora_state_dict" in checkpoint
        or isinstance(state, dict) and any(
            isinstance(key, str) and (
                ".parametrizations." in key or key.endswith((".lora_A", ".lora_B"))
            )
            for key in state
        )
    )
    if unmerged:
        raise SystemExit(
            f"Cannot initialize {method} from unmerged LoRA checkpoint {source}. "
            "Merge LoRA first and use the complete stage2_adapted.pt model, or convert "
            "the full LoRA .ckpt with scripts/convert_stage2_checkpoints.py --lora-output merged."
        )


def fingerprint_frozen_modules(
    state: dict[str, torch.Tensor], module_names: tuple[str, ...],
) -> str:
    """Hash a declared frozen subset; no frozen modules has a valid empty hash."""
    prefixes = tuple(name + "." for name in module_names)
    trunk = {key: value for key, value in state.items() if key.startswith(prefixes)}
    for name in module_names:
        if not any(key.startswith(name + ".") for key in trunk):
            raise ValueError(f"Cannot identify frozen adaptation module without {name}.* weights")
    digest = hashlib.sha256()
    for key in sorted(trunk):
        tensor = trunk[key].detach().cpu().contiguous()
        metadata = f"{key}\0{tensor.dtype}\0{tuple(tensor.shape)}\0{tensor.layout}\0".encode()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def fingerprint_adaptation_trunk(state: dict[str, torch.Tensor]) -> str:
    """Backward-compatible identity of a frozen encoder and optional adapter."""
    names = ("encoder", "adapter") if any(key.startswith("adapter.") for key in state) else ("encoder",)
    return fingerprint_frozen_modules(state, names)


class Stage2AdaptationModule(Stage2LightningModule):
    """Shared adaptation objective, diagnostics and explicit training policy."""

    adaptation_method: str
    train_encoder = False

    def __init__(
        self,
        config: dict[str, Any],
        *,
        vocab_size: int,
        init_checkpoint: str | Path | None = None,
        restoring_full_checkpoint: bool = False,
    ) -> None:
        requested = _adapt_section(config)
        if (
            requested.get("method") is not None
            and resolve_adapt_method(requested) != self.adaptation_method
        ):
            raise ValueError(
                f"{type(self).__name__} requires adapt.method={self.adaptation_method!r}"
            )
        module_config = copy.deepcopy(config)
        module_config["adapt"]["method"] = self.adaptation_method
        module_config["stage2"]["freeze_encoder"] = not self.train_encoder
        adapter_cfg = module_config.get("stage2", {}).get("phoneme_adapter")
        if isinstance(adapter_cfg, dict):
            adapter_cfg["freeze"] = True
            adapter_cfg["ctc_weight"] = 0.0
            if restoring_full_checkpoint or (
                self.adaptation_method != "lora" and init_checkpoint
            ):
                # Full tuning takes the frozen trunk solely from the selected
                # complete model. An older Step-A file must not overwrite it.
                adapter_cfg["init_checkpoint"] = ""
        super().__init__(
            module_config,
            vocab_size=vocab_size,
            freeze_encoder=not self.train_encoder,
            init_checkpoint=init_checkpoint,
            require_full_qbyt_init=bool(init_checkpoint),
        )
        for param in self.parameters():
            param.requires_grad = False
        self.freeze_adapter = self.adapter is not None
        self.ctc_weight = 0.0
        self.encoder.requires_grad_(self.train_encoder)
        self._set_frozen_submodules_to_eval()
        self._adapt_cfg = _adapt_section(module_config)
        self.target_score_diagnostics = BinaryScoreDiagnostics(
            deployment_threshold=self.deployment_threshold,
            ece_num_bins=self.ece_num_bins,
            sync_on_compute=True,
        )
        self.joint_score_diagnostics = torch.nn.ModuleDict()
        if str(self._adapt_cfg.get("phase")) == "joint":
            for domain in ("tts", "musan"):
                self.joint_score_diagnostics[domain] = BinaryScoreDiagnostics(
                    deployment_threshold=self.deployment_threshold,
                    ece_num_bins=self.ece_num_bins,
                    sync_on_compute=True,
                )

    @property
    def adaptation_param_counts(self) -> dict[str, int]:
        return {
            "total": sum(param.numel() for param in self.parameters()),
            "trainable": sum(param.numel() for param in self.parameters() if param.requires_grad),
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        _validate_adapt_checkpoint_method(checkpoint, expected=self.adaptation_method)
        super().on_load_checkpoint(checkpoint)

    def _optimizer_config(self) -> dict[str, Any]:
        adapt = self._adapt_cfg
        lr, _ = resolve_adapt_lr(adapt)
        return {
            "optimizer": str(adapt.get("optimizer", "adam")).lower(),
            "lr": lr,
            "weight_decay": float(adapt.get("weight_decay", 0.0)),
            "warmup_steps": int(adapt.get("warmup_steps", 100)),
            "total_steps": int(adapt.get("max_steps", 3000)),
        }

    def train(self, mode: bool = True):
        """Keep frozen feature extractors deterministic during adaptation."""
        super().train(mode)
        if mode:
            self._set_frozen_submodules_to_eval()
        return self

    def _optimizer_parameters(self) -> list:
        trainable = [param for param in self.parameters() if param.requires_grad]
        if not trainable:
            raise RuntimeError(f"No trainable {self.adaptation_method} parameters found")
        return trainable

    def configure_optimizers(self) -> dict:
        trainable = self._optimizer_parameters()
        optim_cfg = self._optimizer_config()

        import torch as torch_mod
        from transformers import get_cosine_schedule_with_warmup

        optimizer_name = optim_cfg["optimizer"]
        if optimizer_name == "adamw":
            optimizer = torch_mod.optim.AdamW(
                trainable,
                lr=optim_cfg["lr"],
                weight_decay=optim_cfg["weight_decay"],
            )
        elif optimizer_name == "adam":
            optimizer = torch_mod.optim.Adam(
                trainable,
                lr=optim_cfg["lr"],
                weight_decay=optim_cfg["weight_decay"],
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name!r}")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=optim_cfg["warmup_steps"],
            num_training_steps=optim_cfg["total_steps"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        total_loss, losses, logits = self._forward_train_losses(batch)
        self._log_train_losses(total_loss, losses)
        source = batch.get("source")
        if source is not None:
            self._log_source_losses(
                logits, batch["label"], source, losses["utt_sample_mask"]
            )
        if "domain_source" in batch:
            stats = sum_across_processes(grouped_domain_bce_totals(
                logits, batch["label"], batch["domain_source"], losses["utt_sample_mask"]
            ))
            total = stats[:, 2].sum()
            for name, (loss_sum, valid_count, count, positives) in zip(
                ("lph", "real", "tts", "musan"), stats
            ):
                nan = total.new_tensor(float("nan"))
                for key, value in (
                    (f"source_{name}_fraction", count / total if total > 0 else nan),
                    (f"loss_{name}_utt_raw", loss_sum / valid_count if valid_count > 0 else nan),
                    (f"positive_{name}_fraction", positives / count if count > 0 else nan),
                ):
                    self.log(f"train/microbatch/{key}", value, on_step=True, sync_dist=True)
            loader = getattr(getattr(self, "_trainer", None), "train_dataloader", None)
            if hasattr(loader, "mark_consumed"):
                loader.mark_consumed()
        return total_loss

    def _log_source_losses(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        source: torch.Tensor,
        utt_sample_mask: torch.Tensor,
    ) -> None:
        global_stats = sum_across_processes(
            grouped_utt_bce_totals(logits, labels, source, utt_sample_mask)
        )
        keyword_loss_sum, keyword_count, lph_loss_sum, lph_count = global_stats.unbind()
        total_count = keyword_count + lph_count
        no_value = global_stats.new_tensor(float("nan"))
        self.log(
            "train/microbatch/source_keyword_fraction",
            keyword_count / total_count if total_count.item() > 0 else no_value,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "train/microbatch/loss_keyword_utt_raw",
            keyword_loss_sum / keyword_count if keyword_count.item() > 0 else no_value,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "train/microbatch/loss_lph_utt_raw",
            lph_loss_sum / lph_count if lph_count.item() > 0 else no_value,
            on_step=True,
            sync_dist=True,
        )

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logits, _ = self(
            batch["feat"], batch["feat_lengths"], batch["anchor"]
        )
        labels = batch["label"].int()

        if dataloader_idx == 0:
            self._update_score_diagnostics(
                self.target_score_diagnostics,
                logits=logits,
                labels=labels,
                sample_ids=batch.get("sample_id"),
            )
        elif dataloader_idx == 1:
            self._update_score_diagnostics(
                self.score_diagnostics,
                logits=logits,
                labels=labels,
                sample_ids=batch.get("sample_id"),
            )
        else:
            name = {2: "tts", 3: "musan"}[dataloader_idx]
            self._update_score_diagnostics(
                self.joint_score_diagnostics[name], logits=logits, labels=labels,
                sample_ids=batch.get("sample_id"),
            )

    def on_validation_epoch_end(self) -> None:
        target_metrics = self._log_score_diagnostics(
            self.target_score_diagnostics,
            namespace="val/target_",
            progress_bar=True,
        )
        self.log(
            "val/target_utt_loss",
            target_metrics["log_loss"],
            prog_bar=True,
            sync_dist=True,
        )
        lph_metrics = self._log_score_diagnostics(
            self.score_diagnostics,
            namespace="val/lph_",
            progress_bar=True,
        )
        self.log(
            "val/lph_utt_loss",
            lph_metrics["log_loss"],
            prog_bar=True,
            sync_dist=True,
        )
        # Filename/monitor aliases are kept out of CSV/TensorBoard because a
        # slash cannot be used safely in a checkpoint format field.
        self.log(
            "val_target_auc", target_metrics["auc"], sync_dist=True, logger=False
        )
        self.log("val_lph_auc", lph_metrics["auc"], sync_dist=True, logger=False)
        if self._adapt_cfg.get("phase") == "joint":
            # In joint mode the primary target metric always means real speech.
            for key, value in target_metrics.items():
                self.log(f"val/real_{key}", value.float(), sync_dist=True)
            for name, metric in self.joint_score_diagnostics.items():
                if name == "musan" and not (self._adapt_cfg.get("joint") or {}).get("background_eval_list"):
                    continue
                metrics = self._log_score_diagnostics(metric, namespace=f"val/{name}_")
                self.log(f"val/{name}_utt_loss", metrics["log_loss"], sync_dist=True)
                metric.reset()
        self._log_train_window_metrics()

        self.target_score_diagnostics.reset()
        self.score_diagnostics.reset()


class Stage2LoraAdaptationModule(Stage2AdaptationModule):
    """Frozen Stage II base with trainable LoRA adapters."""

    adaptation_method = "lora"

    def __init__(
        self,
        config: dict[str, Any],
        *,
        vocab_size: int,
        init_checkpoint: str | Path | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_targets: tuple[str, ...] | None = None,
        adapter_checkpoint: str | Path | None = None,
        restoring_full_checkpoint: bool = False,
    ) -> None:
        super().__init__(
            config, vocab_size=vocab_size, init_checkpoint=init_checkpoint,
            restoring_full_checkpoint=restoring_full_checkpoint,
        )
        self._base_model_sha256 = fingerprint_stage2_base(self.state_dict())
        normalized_targets = lora_targets_for_qbyt_family(
            lora_targets, family=self.qbyt_score.family
        )
        self.lora_injected = inject_qbyt_lora(
            self.qbyt,
            rank=lora_rank,
            alpha=lora_alpha,
            targets=normalized_targets,
        )
        self._lora_rank = int(lora_rank)
        self._lora_alpha = float(lora_alpha)
        self._lora_targets = normalized_targets
        checkpoint_adapt = _adapt_section(self._checkpoint_config)
        checkpoint_adapt["rank"] = self._lora_rank
        checkpoint_adapt["alpha"] = self._lora_alpha
        checkpoint_adapt["lora_targets"] = list(self._lora_targets)
        if adapter_checkpoint:
            state = torch.load(adapter_checkpoint, map_location="cpu")
            # LoRA weights are tuned against a specific encoder operating point.
            assert_stream_policy_matches(state, self.stream_policy, source=adapter_checkpoint)
            # LoRA is meaningful only on the exact same bounded path topology.
            assert_qbyt_readout_version(
                state,
                source=adapter_checkpoint,
                expected_alignment=self.qbyt_score,
            )
            adapter_state = _validate_adapter_checkpoint(
                state,
                source=adapter_checkpoint,
                keyword=str(checkpoint_adapt.get("keyword", "")),
                rank=self._lora_rank,
                alpha=self._lora_alpha,
                targets=self._lora_targets,
                base_model_sha256=self._base_model_sha256,
            )
            load_lora_state_dict(self.qbyt, adapter_state, strict=True)

        self.lora_param_counts = count_lora_params(self)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Make LoRA Lightning checkpoints self-describing for later export."""
        super().on_save_checkpoint(checkpoint)
        adapt = _adapt_section(self._checkpoint_config)
        keyword = str(adapt.get("keyword", ""))
        checkpoint.update(
            {
                "checkpoint_kind": "stage2_lora",
                "method": self.adaptation_method,
                "keyword": keyword,
                "slug": str(adapt.get("slug", "")) or slugify(keyword),
                "phase": str(adapt.get("phase", "tts")),
                "rank": self._lora_rank,
                "alpha": self._lora_alpha,
                "lora_targets": list(self._lora_targets),
                STAGE2_BASE_FINGERPRINT_KEY: self._base_model_sha256,
            }
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject a resume whose Python-only LoRA scaling metadata changed."""
        assert_qbyt_readout_version(
            checkpoint,
            source="the LoRA checkpoint being restored",
            expected_alignment=self.qbyt_score,
        )
        super().on_load_checkpoint(checkpoint)
        adapt = _adapt_section(self._checkpoint_config)
        keyword = str(adapt.get("keyword", ""))
        _validate_lora_metadata(
            checkpoint,
            source="the LoRA checkpoint being restored",
            expected={
                "checkpoint_kind": "stage2_lora",
                "method": self.adaptation_method,
                "keyword": keyword,
                "slug": str(adapt.get("slug", "")) or slugify(keyword),
                "phase": str(adapt.get("phase", "tts")),
                "rank": self._lora_rank,
                "alpha": self._lora_alpha,
                "lora_targets": self._lora_targets,
            },
            required=(
                "checkpoint_kind",
                "keyword",
                "phase",
                "rank",
                "alpha",
                "lora_targets",
            ),
        )
        state = checkpoint.get("state_dict")
        if not isinstance(state, dict):
            raise SystemExit(
                "Cannot resume LoRA checkpoint: state_dict is missing or invalid."
            )
        actual_base_sha256 = fingerprint_stage2_base(state)
        _validate_base_fingerprint(
            checkpoint,
            source="the LoRA checkpoint being restored",
            expected=actual_base_sha256,
        )
        self._base_model_sha256 = actual_base_sha256


class Stage2FullAdaptationModule(Stage2AdaptationModule):
    """Common initialization and checkpoint contract for full model adaptation."""

    checkpoint_kind: str

    def __init__(
        self,
        config: dict[str, Any],
        *,
        vocab_size: int,
        init_checkpoint: str | Path | None = None,
        restoring_full_checkpoint: bool = False,
    ) -> None:
        super().__init__(
            config, vocab_size=vocab_size, init_checkpoint=init_checkpoint,
            restoring_full_checkpoint=restoring_full_checkpoint,
        )
        self.qbyt.requires_grad_(True)
        self._frozen_trunk_sha256 = fingerprint_frozen_modules(
            self.state_dict(), self.frozen_module_names,
        )
        self._initial_model_sha256 = (
            None if restoring_full_checkpoint else fingerprint_stage2_base(self.state_dict())
        )

    @property
    def frozen_module_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, frozen, module in (
                ("encoder", self.freeze_encoder, self.encoder),
                ("adapter", self.freeze_adapter, self.adapter),
            )
            if frozen and module is not None
        )

    def assert_frozen_modules_unchanged(self) -> None:
        current = fingerprint_frozen_modules(self.state_dict(), self.frozen_module_names)
        if current != self._frozen_trunk_sha256:
            raise RuntimeError(
                f"Frozen modules {self.frozen_module_names} changed during {self.adaptation_method}"
            )

    def full_adaptation_metadata(self) -> dict[str, Any]:
        """Pure export metadata, excluding training progress and stochastic state."""
        return {
            "checkpoint_kind": self.checkpoint_kind,
            "method": self.adaptation_method,
            "frozen_modules": list(self.frozen_module_names),
            "frozen_trunk_sha256": self._frozen_trunk_sha256,
            "initial_model_sha256": self._initial_model_sha256,
        }

    def _load_init_checkpoint(
        self, checkpoint_path: Path, *, require_full_qbyt: bool = False,
    ) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        _validate_qbyt_full_initialization(
            checkpoint, source=checkpoint_path, method=self.adaptation_method,
        )
        state = extract_state_dict(checkpoint)
        if self.adapter is not None and not any(
            isinstance(key, str) and key.startswith("adapter.") for key in state
        ):
            raise SystemExit(
                f"Cannot initialize {self.adaptation_method} from {checkpoint_path}: "
                "stage2.phoneme_adapter is enabled but the complete model has no "
                "adapter.* weights. Supply a model trained with the adapter or "
                "disable the adapter; external Step-A initialization cannot replace "
                "the selected model's frozen trunk."
            )
        super()._load_init_checkpoint(checkpoint_path, require_full_qbyt=require_full_qbyt)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        adapt = _adapt_section(self._checkpoint_config)
        keyword = str(adapt.get("keyword", ""))
        checkpoint.update({
            **self.full_adaptation_metadata(),
            "keyword": keyword,
            "slug": str(adapt.get("slug", "")) or slugify(keyword),
            "phase": str(adapt.get("phase", "tts")),
            "adapt_optimizer": self._optimizer_config(),
        })

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        source = f"the {self.adaptation_method} checkpoint being restored"
        adapt = _adapt_section(self._checkpoint_config)
        keyword = str(adapt.get("keyword", ""))
        _validate_lora_metadata(
            checkpoint, source=source,
            expected={
                "checkpoint_kind": self.checkpoint_kind,
                "method": self.adaptation_method,
                "keyword": keyword,
                "slug": str(adapt.get("slug", "")) or slugify(keyword),
                "phase": str(adapt.get("phase", "tts")),
            },
            required=("checkpoint_kind", "method", "keyword", "phase", "frozen_trunk_sha256"),
            label=self.adaptation_method,
        )
        assert_sequence_objective_matches(checkpoint, self._stage2_cfg, source=source)
        state = checkpoint.get("state_dict")
        if not isinstance(state, dict):
            raise SystemExit(f"Cannot resume {self.adaptation_method} checkpoint: state_dict is missing or invalid.")
        _validate_qbyt_full_initialization(checkpoint, source=source, method=self.adaptation_method)
        saved_modules = checkpoint.get("frozen_modules")
        if saved_modules is not None and saved_modules != list(self.frozen_module_names):
            raise SystemExit(f"Cannot resume {self.adaptation_method}: frozen module selection changed")
        actual_trunk = fingerprint_frozen_modules(state, self.frozen_module_names)
        if checkpoint.get("frozen_trunk_sha256") != actual_trunk:
            raise SystemExit(
                f"Cannot resume {self.adaptation_method} checkpoint: frozen_trunk_sha256 "
                "does not match its frozen module weights."
            )
        if checkpoint.get("adapt_optimizer") != self._optimizer_config():
            raise SystemExit(
                f"Cannot resume {self.adaptation_method} checkpoint with changed optimizer, learning rate, "
                "weight decay, warmup or schedule horizon. Use run.init_checkpoint for a new weights-only run."
            )
        self._frozen_trunk_sha256 = actual_trunk
        self._initial_model_sha256 = checkpoint.get("initial_model_sha256")


class Stage2QbytFullAdaptationModule(Stage2FullAdaptationModule):
    """Train every QbyT parameter while keeping encoder and adapter fixed."""

    adaptation_method = "qbyt_full"
    checkpoint_kind = "stage2_qbyt_full"


class Stage2EncoderQbytFullAdaptationModule(Stage2FullAdaptationModule):
    """Fine-tune encoder and QbyT, with an optional frozen differentiable adapter."""

    adaptation_method = "encoder_qbyt_full"
    checkpoint_kind = "stage2_encoder_qbyt_full"
    train_encoder = True

    def __init__(
        self, config: dict[str, Any], *, vocab_size: int,
        init_checkpoint: str | Path | None = None,
        restoring_full_checkpoint: bool = False,
    ) -> None:
        super().__init__(
            config, vocab_size=vocab_size, init_checkpoint=init_checkpoint,
            restoring_full_checkpoint=restoring_full_checkpoint,
        )
        self._encoder_schedule = (
            EncoderFinetuneSchedule.from_config(self._checkpoint_config)
            if self.stream_policy.backend == "icefall_zipformer" else None
        )
        self._pending_microbatches = 0
        self._pending_random_state = None
        self._resume_training_device = None
        if self._encoder_schedule is not None:
            if not callable(getattr(self.encoder, "set_batch_count", None)):
                raise ValueError("Icefall encoder fine-tuning requires an encoder.set_batch_count hook")
            self.encoder.set_batch_count(self._encoder_schedule.batch_count)

    def _optimizer_config(self) -> dict[str, Any]:
        config = super()._optimizer_config()
        config["param_groups"] = [
            {"name": "qbyt", "lr": config["lr"]},
            {"name": "encoder", "lr": resolve_encoder_adapt_lr(self._adapt_cfg)},
        ]
        config["accumulate_grad_batches"] = int(self._stage2_cfg.get("accumulate_grad_batches", 1))
        stage1 = self._checkpoint_config["stage1"]
        scheduled_dropout = bool(stage1.get("use_icefall_dropout_schedule", True))
        config["encoder_training"] = {
            "stream_policy": asdict(self.stream_policy),
            "use_icefall_dropout_schedule": scheduled_dropout,
            "dropout_rate": (
                None if scheduled_dropout and self.stream_policy.backend == "icefall_zipformer"
                else float(stage1.get("dropout_rate", 0.1))
            ),
            "warmup_batches": float(stage1.get("warmup_batches", 4000.0)),
            "positional_dropout_rate": float(stage1.get("positional_dropout_rate", 0.1)),
            "attention_dropout_rate": float(stage1.get("attention_dropout_rate", 0.0)),
            "precision": str(self._stage2_cfg.get("precision", "32-true")),
            "gradient_clip_val": float(self._stage2_cfg.get("gradient_clip_val", 1.0)),
        }
        return config

    def _optimizer_parameters(self) -> list[dict[str, Any]]:
        groups = [
            {**spec, "params": list(getattr(self, spec["name"]).parameters())}
            for spec in self._optimizer_config()["param_groups"]
        ]
        parameters = [parameter for group in groups for parameter in group["params"]]
        expected = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        actual = [id(parameter) for parameter in parameters]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise RuntimeError("Encoder/QbyT optimizer groups must partition all trainable parameters exactly once")
        return groups

    def on_train_start(self) -> None:
        super().on_train_start()
        if self._pending_random_state is not None:
            if self.device.type != self._resume_training_device:
                raise SystemExit(
                    f"Cannot resume encoder adaptation on {self.device.type} from "
                    f"{self._resume_training_device} training: stochastic state is accelerator-specific. "
                    "Use run.init_checkpoint for a new weights-only run."
                )
            restore_rank_random_states(self._pending_random_state)
            self._pending_random_state = None

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self._encoder_schedule is not None:
            self.encoder.set_batch_count(self._encoder_schedule.batch_count)
            self.log("train/microbatch/encoder_batch_count", self._encoder_schedule.batch_count, on_step=True)
        loss = super().training_step(batch, batch_idx)
        if self._encoder_schedule is not None:
            frames = sum_across_processes(batch["feat_lengths"].to(dtype=torch.int64).sum())
            self._encoder_schedule.advance(int(frames.item()))
        self._pending_microbatches += 1
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        super().on_before_optimizer_step(optimizer)
        from pytorch_lightning.utilities import grad_norm

        for group in optimizer.param_groups:
            name = group["name"]
            self.log(f"train/optimizer/{name}_lr", group["lr"], on_step=True)
            if self._log_grad_norm:
                total = grad_norm(getattr(self, name), norm_type=2).get("grad_2.0_norm_total")
                if total is not None:
                    self.log(f"train/optimizer/{name}_grad_norm_pre_clip", total, on_step=True)
        # Lightning checkpoints do not preserve partially accumulated gradients.
        # This hook also handles an epoch's final, shorter accumulation window.
        self._pending_microbatches = 0

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self._pending_microbatches:
            raise RuntimeError("Cannot checkpoint encoder adaptation during gradient accumulation; save after an optimizer step")
        super().on_save_checkpoint(checkpoint)
        checkpoint["encoder_schedule"] = (
            self._encoder_schedule.state_dict() if self._encoder_schedule is not None else None
        )
        checkpoint["adapt_training_device"] = self.device.type
        # Trainer.save_checkpoint invokes this hook on every rank before the
        # strategy persists the rank-zero payload, so collecting RNG here is safe.
        checkpoint["adapt_random_state"] = capture_rank_random_states()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        if checkpoint.get("frozen_modules") != list(self.frozen_module_names):
            raise SystemExit("Cannot resume encoder adaptation: frozen module selection is missing or changed")
        try:
            if self._encoder_schedule is not None:
                self._encoder_schedule.load_state_dict(checkpoint.get("encoder_schedule"))
                self.encoder.set_batch_count(self._encoder_schedule.batch_count)
            elif checkpoint.get("encoder_schedule") is not None:
                raise ValueError("Cannot restore an Icefall schedule into a different encoder backend")
            validate_rank_random_states(checkpoint.get("adapt_random_state"))
        except ValueError as exc:
            raise SystemExit(f"Cannot resume encoder adaptation: {exc}") from exc
        device = checkpoint.get("adapt_training_device")
        if not isinstance(device, str) or not device:
            raise SystemExit("Cannot resume encoder adaptation: training device type is missing")
        self._resume_training_device = device
        self._pending_random_state = checkpoint["adapt_random_state"]


def _resolve_init_checkpoint(config: dict[str, Any], adapt_paths: dict[str, Any], args: Stage2AdaptArgs) -> str:
    adapt = _adapt_section(config)
    prep = config.get("prep", {}) or {}

    init_checkpoint = (
        args.init_checkpoint
        or prep.get("stage2_ckpt", "")
        or adapt.get("init_checkpoint", "")
        or config.get("stage2", {}).get("init_checkpoint", "")
    )
    if not init_checkpoint:
        raise ValueError(
            "init_checkpoint is required for adaptation (set prep.stage2_ckpt or adapt.init_checkpoint)"
        )
    return str(init_checkpoint)


def _resolve_adapter_resume(
    adapt_paths: dict[str, Any],
    phase: str,
    explicit_checkpoint: str = "",
) -> str | None:
    if explicit_checkpoint:
        path = Path(explicit_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"LoRA adapter checkpoint not found: {path}")
        return str(path)
    if phase == "real":
        tts_adapter = adapt_paths["phase_dir"].parent / "tts" / f"adapter_{adapt_paths['slug_str']}.pt"
        if tts_adapter.exists():
            return str(tts_adapter)
    return None


def _resolve_full_init_checkpoint(
    config: dict[str, Any], adapt_paths: dict[str, Any], args: Stage2AdaptArgs,
) -> str:
    """Continue sequential full tuning from complete prior-phase weights."""
    if args.resume_checkpoint:
        path = Path(args.resume_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Full-model weights checkpoint not found: {path}")
        return str(path)
    if not args.init_checkpoint and adapt_paths["phase_str"] == "real":
        previous = adapt_paths["phase_dir"].parent / "tts" / "stage2_adapted.pt"
        if previous.is_file():
            return str(previous)
    return _resolve_init_checkpoint(config, adapt_paths, args)


def run_stage2_adaptation(config: dict[str, Any], args: Stage2AdaptArgs) -> dict[str, Path]:
    """Run Stage II adaptation for the configured keyword phase and method."""
    try:
        import pytorch_lightning as pl_mod
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/pytorch-lightning. Install CUDA PyTorch on the training machine first."
        ) from exc

    from dma_kws.config import (
        fbank_kwargs,
        get_eval_fbank_config,
        get_fbank_config,
        get_tokenizer_config,
        require_sections,
    )
    from dma_kws.runlog import build_loggers, logger_backend_names
    from dma_kws.stage2 import adapt_console
    from dma_kws.stage2.collate import test_collate_fn, train_collate_fn
    from dma_kws.stage2.dataset import LibriPhraseTrainDataset, stage2_worker_init_fn
    from dma_kws.tokenizer import load_char_tokenizer
    from dma_kws.training.callbacks import (
        build_stage2_callbacks,
        print_run_summary,
        print_training_result_summary,
    )
    from dma_kws.training.metrics_history import (
        append_wide_row,
        build_metrics_history_callback,
        build_run_record,
        collect_hparams,
        numeric_callback_metrics,
    )
    from dma_kws.training.run_context import build_run_context, stamp_run_context
    from dma_kws.training.run_context_callback import RunContextCheckpointCallback
    from dma_kws.training.resume import resolve_versioned_resume_path
    from dma_kws.training.ddp import apply_step_based_validation, build_trainer_kwargs
    from dma_kws.training.device import resolve_accelerator_and_devices

    _apply_adapt_overrides(config, args)
    require_sections(config, ["paths", "stage1", "stage2", "tokenizer", "training", "adapt"])

    adapt = _adapt_section(config)
    method = resolve_adapt_method(adapt)
    adapt["method"] = method
    adapt_paths = _resolve_adapt_paths(config)
    paths = config["paths"]
    stage1 = config["stage1"]
    stage2 = config["stage2"]
    _validate_lora_runtime_config(stage2, method=method)
    training = config["training"]
    tokenizer_cfg = get_tokenizer_config(config)
    reporter = adapt_console.adapt_reporter(config)
    is_primary_process = process_rank() == 0

    phase = adapt_paths["phase_str"]
    joint = phase == "joint"
    joint_cfg = adapt.get("joint", {}) or {}
    joint_manifests = {}
    if joint:
        from dma_kws.stage2.joint_manifest import (
            validate_joint_manifests,
            validate_background_eval_split,
        )
        joint_manifests = validate_joint_manifests(adapt_paths["data_root"])
        validate_background_eval_split(
            stage2.get("background_negative", {}) or {},
            joint_cfg.get("background_eval_list", ""),
        )
    train_manifest = joint_manifests.get("real_train", adapt_paths["train_manifest"])
    eval_manifest = joint_manifests.get("real_eval", adapt_paths["eval_manifest"])
    if joint:
        adapt_paths["train_manifest"] = train_manifest
        adapt_paths["eval_manifest"] = eval_manifest
    if not train_manifest.exists():
        raise SystemExit(f"Adaptation train manifest not found: {train_manifest}")
    if not eval_manifest.exists():
        raise SystemExit(f"Adaptation eval manifest not found: {eval_manifest}")

    if args.limit_steps:
        # Keep the cosine schedule horizon in sync with the truncated run, otherwise
        # training stops while the LR is still on its way down.
        adapt["max_steps"] = int(args.limit_steps)
    accelerator, devices = resolve_accelerator_and_devices(args.device, args.devices)

    adapt_paths["phase_dir"].mkdir(parents=True, exist_ok=True)
    checkpoint_root = adapt_paths["phase_dir"] / "checkpoints"
    log_dir = adapt_paths["phase_dir"] / "logs"
    run_name = f"adapt_{adapt_paths['slug_str']}_{phase}"
    if is_full_adapt_method(method):
        # Run IDs also appear in shared CSV/remote logs, outside the method's directory.
        run_name += f"_{method}"
    resume_path = resolve_versioned_resume_path(
        args.resume_from,
        checkpoint_root,
        run_name,
    )
    if resume_path is not None and args.resume_checkpoint:
        raise ValueError(
            "Choose either run.resume_from (full Lightning state) or "
            "run.resume_checkpoint (weights only), not both."
        )
    if resume_path is not None:
        # A full Lightning checkpoint already contains encoder, QbyT, LoRA and
        # optimizer state. Loading the original base first makes resume depend on
        # a file that may have moved and can silently introduce a different base.
        init_checkpoint = ""
        adapter_resume = None
    else:
        if is_full_adapt_method(method):
            init_checkpoint = _resolve_full_init_checkpoint(config, adapt_paths, args)
            adapter_resume = None
        else:
            init_checkpoint = _resolve_init_checkpoint(config, adapt_paths, args)
            adapter_resume = _resolve_adapter_resume(
                adapt_paths,
                phase,
                args.resume_checkpoint,
            )

    resume_payload = None
    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location="cpu")
        _validate_adapt_checkpoint_method(resume_payload, expected=method)
        assert_sequence_objective_matches(
            resume_payload,
            stage2,
            source=resume_path,
        )

    run_context = build_run_context(
        config,
        section="adapt",
        log_dir=log_dir,
        run_name=run_name,
        limit_steps=args.limit_steps or None,
        resume_from=resume_path,
        resume_checkpoint=resume_payload,
    )
    checkpoint_dir = checkpoint_root / run_context.run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dict_path = resolve_dict_path(config)
    tokenizer = load_char_tokenizer(dict_path, split_with_space=tokenizer_cfg.get("split_with_space", " "))
    vocab_size = len(tokenizer._symbol_table)

    seed = int(training.get("seed", 2025))
    pl_mod.seed_everything(seed, workers=True)
    seq_label_mode = resolve_sequence_objective(stage2).target_mode
    noise_augmentation = stage2.get("noise_augmentation", {}) or {}
    if not isinstance(noise_augmentation, dict):
        raise ValueError("stage2.noise_augmentation must be a mapping")
    background_negative = stage2.get("background_negative", {}) or {}
    if not isinstance(background_negative, dict):
        raise ValueError("stage2.background_negative must be a mapping")
    # MixedAdaptationDataset ignores the sampler's index and samples from its own
    # RNG. Give num_workers=0 runs distinct streams too; worker processes apply
    # the same rank offset again from their DataLoader-provided base seed.
    sampling_seed = seed + 1_000_003 * process_rank()

    keyword_dataset = KeywordAdaptationDataset(
        manifest_path=train_manifest,
        keyword=adapt_paths["keyword_str"],
        fbank_root=adapt_paths["fbank_root"],
        tokenizer=tokenizer,
        manifest_root=adapt_paths["data_root"],
        seq_label_mode=seq_label_mode,
    )

    processed_root = Path(paths["processed_root"])
    feature_root = Path(paths.get("feature_root", processed_root))
    parquet_file = _resolve_path(
        stage2,
        "parquet_file",
        processed_root / "stage2_qbyt" / "aggregated_segments_with_g2p_distance.parquet",
    )
    wav_dir = _resolve_path(stage2, "wav_dir", feature_root / "fbank")

    libri_dataset = LibriPhraseTrainDataset(
        parquet_file=parquet_file,
        wav_dir=wav_dir,
        tokenizer=tokenizer,
        negative_ratio=int(stage2.get("negative_ratio", 1)),
        hard_negative_ratio=int(stage2.get("hard_negative_ratio", 1)),
        sample_lens=int(adapt.get("sample_lens", stage2.get("sample_lens", 5000))),
        seed=sampling_seed,
        noise_augmentation=noise_augmentation,
        background_negative=background_negative,
        fbank_kwargs=fbank_kwargs(get_fbank_config(config)),
        seq_label_mode=seq_label_mode,
        metadata_cache=stage2.get("metadata_cache", {}) or {},
    )

    if joint:
        from dma_kws.stage2.joint_dataset import JointAdaptationDataset, JointBatchSampler
        from dma_kws.stage2.joint_loader import JointDataLoader, JointEvalSampler

        tts_dataset = KeywordAdaptationDataset(
            manifest_path=joint_manifests["tts_train"],
            keyword=adapt_paths["keyword_str"], fbank_root=adapt_paths["fbank_root"],
            tokenizer=tokenizer, manifest_root=adapt_paths["data_root"],
            seq_label_mode=seq_label_mode,
        )
        train_dataset = JointAdaptationDataset(
            real_dataset=keyword_dataset, tts_dataset=tts_dataset, libri_dataset=libri_dataset,
            mix_ratio=float(adapt.get("mix_ratio", 0.5)),
            real_fraction=float(joint_cfg.get("real_fraction", 0.6)),
            background_keyword_fraction=float(joint_cfg.get("background_keyword_fraction", 0.5)),
            sample_lens=int(adapt.get("sample_lens", stage2.get("sample_lens", 5000))),
            seed=seed,
        )
    else:
        train_dataset = MixedAdaptationDataset(
            keyword_dataset=keyword_dataset,
            libri_dataset=libri_dataset,
            mix_ratio=float(adapt.get("mix_ratio", 0.5)),
            sample_lens=int(adapt.get("sample_lens", stage2.get("sample_lens", 5000))),
            seed=sampling_seed,
        )

    batch_size = int(adapt.get("batch_size_per_gpu", stage2.get("batch_size_per_gpu", 64)))
    num_workers = int(adapt.get("num_workers", stage2.get("num_workers", 2)))
    # The target-keyword val set is a few thousand clips against a 64k-sample virtual
    # train epoch; giving it the full train worker count only multiplies worker
    # processes and open file descriptors for no throughput gain.
    val_workers_cfg = adapt.get("val_num_workers")
    val_num_workers = min(num_workers, 4) if val_workers_cfg is None else int(val_workers_cfg)
    from dma_kws.training.loaders import build_loader_kwargs

    dataloader_cfg = stage2.get("dataloader", {}) or {}
    loader_kwargs = build_loader_kwargs(num_workers, dataloader_cfg)
    val_loader_kwargs = build_loader_kwargs(val_num_workers, dataloader_cfg)
    if joint:
        train_dataloader = JointDataLoader(
            train_dataset, data_signature=_joint_data_signature(
                config, joint_manifests, parquet_file=parquet_file, wav_dir=wav_dir, dict_path=dict_path,
            ),
            batch_sampler=JointBatchSampler(train_dataset, batch_size=batch_size, seed=seed),
            accumulation_steps=int(stage2.get("accumulate_grad_batches", 1)),
            num_workers=num_workers, collate_fn=train_collate_fn,
            worker_init_fn=stage2_worker_init_fn, **loader_kwargs,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn,
            drop_last=True,
            worker_init_fn=stage2_worker_init_fn,
            **loader_kwargs,
        )

    target_val_dataset = TargetKeywordValDataset(
        manifest_path=eval_manifest,
        keyword=adapt_paths["keyword_str"],
        fbank_root=adapt_paths["fbank_root"],
        tokenizer=tokenizer,
        manifest_root=adapt_paths["data_root"],
    )
    target_val_loader = DataLoader(
        target_val_dataset,
        batch_size=int(adapt.get("val_batch_size", batch_size)),
        shuffle=False,
        num_workers=val_num_workers,
        collate_fn=test_collate_fn,
        drop_last=False,
        sampler=JointEvalSampler(target_val_dataset) if joint else None,
        **val_loader_kwargs,
    )
    lph_val_loader = _build_val_dataloader(config, tokenizer)
    val_loaders = [target_val_loader, lph_val_loader]
    if joint:
        lph_val_loader = DataLoader(
            lph_val_loader.dataset, batch_size=lph_val_loader.batch_size,
            sampler=JointEvalSampler(lph_val_loader.dataset),
            num_workers=val_num_workers, collate_fn=test_collate_fn, **val_loader_kwargs,
        )
        tts_val_dataset = TargetKeywordValDataset(
            manifest_path=joint_manifests["tts_eval"], keyword=adapt_paths["keyword_str"],
            fbank_root=adapt_paths["fbank_root"], tokenizer=tokenizer,
            manifest_root=adapt_paths["data_root"],
        )
        # Each manifest may carry an explicit pronunciation. Comparing resolved
        # IDs also catches an override present in only one source or split.
        for dataset in (tts_dataset, tts_val_dataset, target_val_dataset):
            if list(dataset._anchor_seq) != list(keyword_dataset._anchor_seq):
                raise ValueError("Joint real/TTS train/eval keyword pronunciations must match")
        tts_val_loader = DataLoader(
            tts_val_dataset, batch_size=int(adapt.get("val_batch_size", batch_size)),
            sampler=JointEvalSampler(tts_val_dataset), num_workers=val_num_workers,
            collate_fn=test_collate_fn, **val_loader_kwargs,
        )
        val_loaders = [target_val_loader, lph_val_loader, tts_val_loader]
        if joint_cfg.get("background_eval_list"):
            from dma_kws.stage2.joint_validation import BackgroundValidationDataset
            bg_val_dataset = BackgroundValidationDataset(
                audio_list_path=joint_cfg["background_eval_list"],
                anchor_seq=keyword_dataset._anchor_seq,
                num_samples=joint_cfg.get("background_val_samples", 512),
                seed=joint_cfg.get("background_eval_seed", 2026),
                duration_seconds_min=background_negative.get("duration_seconds_min", 1.0),
                duration_seconds_max=background_negative.get("duration_seconds_max", 3.0),
                fbank_kwargs=fbank_kwargs(get_eval_fbank_config(config)),
            )
            val_loaders.append(DataLoader(
                bg_val_dataset, batch_size=int(adapt.get("val_batch_size", batch_size)),
                sampler=JointEvalSampler(bg_val_dataset), num_workers=val_num_workers,
                collate_fn=test_collate_fn, **val_loader_kwargs,
            ))
        elif is_primary_process:
            reporter.warn("Joint MUSAN validation is disabled: set adapt.joint.background_eval_list to a held-out list")

    mix_ratio = float(adapt.get("mix_ratio", 0.5))
    if is_primary_process:
        reporter.print_table(
            *adapt_console.dataset_table(
                [
                    (
                        "real keyword train" if joint else "keyword train",
                        len(keyword_dataset),
                        adapt_console.label_breakdown(keyword_dataset),
                    ),
                    *([("tts keyword train", len(tts_dataset), adapt_console.label_breakdown(tts_dataset))]
                      if joint else []),
                    (
                        "libriphrase train pool",
                        len(libri_dataset),
                        f"negative_ratio={stage2.get('negative_ratio', 1)}",
                    ),
                    (
                        "mixed virtual epoch",
                        len(train_dataset),
                        f"mix_ratio={mix_ratio} (keyword:libriphrase)",
                    ),
                    (
                        "target keyword val",
                        len(target_val_dataset),
                        adapt_console.label_breakdown(target_val_dataset),
                    ),
                    (
                        "libriphrase val",
                        len(lph_val_loader.dataset),
                        f"split={(stage2.get('eval', {}) or {}).get('split', 'hard')}",
                    ),
                ]
            ),
            title="Datasets",
        )
        if joint:
            reporter.print_table(
                ["joint draw", "fraction"],
                [[name, f"{weight:.2%}"] for name, weight in train_dataset.weights.items()],
                title="Joint batch quotas",
            )

    lora_rank = lora_alpha = None
    lora_targets: tuple[str, ...] = ()
    if is_full_adapt_method(method):
        full_module_class = {
            "qbyt_full": Stage2QbytFullAdaptationModule,
            "encoder_qbyt_full": Stage2EncoderQbytFullAdaptationModule,
        }[method]
        model = full_module_class(
            config,
            vocab_size=vocab_size,
            init_checkpoint=init_checkpoint,
            restoring_full_checkpoint=resume_path is not None,
        )
    else:
        lora_rank = int(adapt.get("rank", 16))
        lora_alpha = float(adapt.get("alpha", 32))
        from dma_kws.stage2.readout import resolve_qbyt_score_spec

        score = resolve_qbyt_score_spec(config.get("stage2", {}))
        lora_targets = lora_targets_for_qbyt_family(
            adapt.get("lora_targets"), family=score.family,
        )
        adapt["rank"] = lora_rank
        adapt["alpha"] = lora_alpha
        adapt["lora_targets"] = list(lora_targets)
        model = Stage2LoraAdaptationModule(
            config,
            vocab_size=vocab_size,
            init_checkpoint=init_checkpoint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_targets=lora_targets,
            adapter_checkpoint=adapter_resume,
            restoring_full_checkpoint=resume_path is not None,
        )

    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")

    loggers = build_loggers(
        log_dir,
        run_name,
        config=config,
        section="adapt",
        version=run_context.version,
    )

    hparams = collect_hparams(
        config,
        section="adapt",
        effective_max_steps=run_context.effective_max_steps,
        extra={
            "slug": adapt_paths["slug_str"],
            **run_context.identity(),
            "checkpoint_dir": str(checkpoint_dir),
            **({"joint_sampling_weights": json.dumps(train_dataset.weights, sort_keys=True),
                "joint_data_signature": train_dataloader.data_signature} if joint else {}),
        },
    )
    if is_primary_process:
        for train_logger in loggers:
            train_logger.log_hyperparams(hparams)

    limit_steps = args.limit_steps or int(adapt.get("max_steps", 3000)) or None
    trainer_kwargs = build_trainer_kwargs(
        config,
        devices,
        limit_steps=limit_steps,
        accelerator=accelerator,
    )
    trainer_kwargs["max_steps"] = limit_steps
    if joint:
        # The joint batch sampler owns rank sharding and source quotas. All
        # validation loaders have explicit dynamic rank samplers as well.
        trainer_kwargs["use_distributed_sampler"] = False

    # Adaptation runs on short virtual epochs, so it needs its own validation
    # cadence instead of inheriting the Stage II pretraining schedule.
    adapt_validation = adapt.get("validation", {}) or {}
    if adapt_validation.get("val_check_interval") is not None:
        trainer_kwargs["val_check_interval"] = int(adapt_validation["val_check_interval"])
    if adapt_validation.get("limit_val_batches") is not None:
        trainer_kwargs["limit_val_batches"] = adapt_validation["limit_val_batches"]
    if joint and int(trainer_kwargs["val_check_interval"]) % train_dataloader.accumulation_steps:
        raise ValueError(
            "Joint adapt.validation.val_check_interval must be divisible by "
            "stage2.accumulate_grad_batches so validation checkpoints retain complete optimizer updates"
        )
    # Keep Lightning's train-batch interval on one counter across virtual
    # epochs. The dataloader is sharded only later during DDP setup, so
    # comparing against its pre-DDP length is also incorrect.
    apply_step_based_validation(
        trainer_kwargs,
        len(train_dataloader),
        force=True,
    )

    recipe = str(training.get("recipe", "adapt"))
    callbacks = build_stage2_callbacks(
        config,
        recipe,
        checkpoint_dir=checkpoint_dir,
        val_check_interval=int(trainer_kwargs["val_check_interval"]),
        monitor_override=str(adapt.get("checkpoint_monitor", "val_target_auc")),
        filename_override=str(
            adapt.get(
                "checkpoint_filename",
                "step_{step:06d}_target_auc_{val_target_auc:.6f}",
            )
        ),
        section="adapt",
    )
    checkpoint_callback = callbacks[0]
    callbacks.append(RunContextCheckpointCallback(run_context))
    history_callback = build_metrics_history_callback(
        run_name=run_name,
        run_id=run_context.run_id,
        default_dir=run_context.run_dir,
    )
    callbacks.insert(0, history_callback)

    if is_primary_process:
        print_run_summary(
            config=config,
            devices=devices,
            accelerator=accelerator,
            section="adapt",
            train_samples=len(train_dataset),
            val_samples=sum(len(loader.dataset) for loader in val_loaders),
            param_counts=(model.lora_param_counts if method == "lora" else model.adaptation_param_counts),
            extra_rows=[
                ("keyword", adapt_paths["keyword_str"]),
                ("phase", phase),
                ("batches_per_epoch", str(len(train_dataloader))),
                ("mix_ratio", str(mix_ratio)),
                *([
                    ("lora_rank", str(lora_rank)),
                    ("lora_alpha", str(lora_alpha)),
                    ("lora_targets", ", ".join(lora_targets)),
                    ("lora_injected_modules", str(len(model.lora_injected))),
                ] if method == "lora" else [
                    ("trainable_module", (
                        "encoder and qbyt (all parameters)" if method == "encoder_qbyt_full"
                        else "qbyt (all parameters)"
                    )),
                    ("frozen_modules", ", ".join(
                        (["encoder"] if method == "qbyt_full" else [])
                        + (["phoneme adapter"] if model.adapter is not None else [])
                    ) or "(none)"),
                ]),
                ("params_file", str(args.params_file or "(none)")),
            ],
            paths={
                "run_id": run_context.run_id,
                "train_manifest": train_manifest,
                "eval_manifest": eval_manifest,
                "checkpoint_dir": checkpoint_dir,
                "log_dir": log_dir,
                "run_dir": run_context.run_dir,
                **(
                    {"parent_run_id": run_context.parent_run_id}
                    if run_context.parent_run_id
                    else {}
                ),
                **(
                    {"init_checkpoint": Path(init_checkpoint)}
                    if init_checkpoint
                    else {"resume_checkpoint": resume_path}
                ),
            },
            effective_max_steps=run_context.effective_max_steps,
            effective_logging_backends=logger_backend_names(loggers),
        )

    trainer = pl_mod.Trainer(
        accelerator=accelerator,
        callbacks=callbacks,
        logger=loggers,
        **trainer_kwargs,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_loaders,
        ckpt_path=resume_path,
    )

    global_step = int(trainer.global_step)
    merged_out = checkpoint_dir / "stage2_adapted.pt"
    artifacts = {"merged": merged_out}
    published_outputs = {"merged": adapt_paths["merged_path"]}
    if method == "lora":
        adapter_out = checkpoint_dir / f"adapter_{adapt_paths['slug_str']}.pt"
        artifacts["adapter"] = adapter_out
        published_outputs["phase_adapter"] = (
            adapt_paths["phase_dir"] / f"adapter_{adapt_paths['slug_str']}.pt"
        )
    else:
        published_outputs["phase_model"] = adapt_paths["phase_dir"] / "stage2_adapted.pt"

    if trainer.is_global_zero:
        artifact_step, artifact_source = restore_best_checkpoint_weights(
            model,
            checkpoint_callback,
            final_step=global_step,
        )
        if method == "lora":
            base_model_sha256 = fingerprint_stage2_base(model.state_dict())
            if base_model_sha256 != model._base_model_sha256:
                raise RuntimeError(
                    "The frozen Stage II base changed during LoRA training. Refusing "
                    "to save an adapter whose recorded base identity would be false."
                )
            adapter_payload = stamp_run_context(
                stamp_qbyt_readout_version(
                    {
                        "checkpoint_kind": "stage2_lora_adapter",
                        "method": method,
                        "lora_state_dict": lora_state_dict(model.qbyt),
                        "config": model._checkpoint_config,
                        "step": artifact_step,
                        "keyword": adapt_paths["keyword_str"],
                        "slug": adapt_paths["slug_str"],
                        "phase": phase,
                        "rank": lora_rank,
                        "alpha": lora_alpha,
                        "lora_targets": list(lora_targets),
                        STAGE2_BASE_FINGERPRINT_KEY: base_model_sha256,
                    },
                    alignment=model.qbyt_score,
                ),
                run_context,
            )
            _atomic_torch_save(adapter_payload, adapter_out)
            _atomic_torch_save(adapter_payload, published_outputs["phase_adapter"])
            merge_lora(model.qbyt)
        else:
            model.assert_frozen_modules_unchanged()
        merged_payload = stamp_run_context(
            stamp_qbyt_readout_version(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model._checkpoint_config,
                    "step": artifact_step,
                    "keyword": adapt_paths["keyword_str"],
                    "slug": adapt_paths["slug_str"],
                    "phase": phase,
                    "method": method,
                    "tokenizer_dict_path": str(dict_path),
                    "vocab_size": vocab_size,
                    **(model.full_adaptation_metadata() if is_full_adapt_method(method) else {}),
                },
                alignment=model.qbyt_score,
            ),
            run_context,
        )
        _atomic_torch_save(merged_payload, merged_out)
        # Stable full-model outputs support sequential phases and deployment.
        _atomic_torch_save(merged_payload, published_outputs["merged"])
        if is_full_adapt_method(method):
            _atomic_torch_save(merged_payload, published_outputs["phase_model"])

        final_metrics = numeric_callback_metrics(dict(trainer.callback_metrics))
        runs_csv = Path(paths["exp_root"]) / "stage2_adapt" / "runs.csv"
        append_wide_row(
            runs_csv,
            build_run_record(
                run_name=run_name,
                hparams=hparams,
                final_metrics=final_metrics,
                best_metrics=history_callback.best,
                global_step=global_step,
                duration_seconds=history_callback.duration_seconds,
                metric_step=history_callback.last_validation_step,
                best_steps=history_callback.best_steps,
                identity=run_context.identity(),
                provenance={
                    "metrics_source": "last_trainer_state",
                    "primary_artifact_source": artifact_source,
                    "primary_artifact_step": artifact_step,
                    "primary_artifact_path": str(merged_out),
                    "published_output": json.dumps(
                        {
                            name: str(path)
                            for name, path in published_outputs.items()
                        },
                        sort_keys=True,
                    ),
                },
            ),
        )

        log_files = {"runs_csv": runs_csv}
        if history_callback.csv_path is not None:
            log_files["eval_history"] = history_callback.csv_path
        metrics = {
            key: round(value, 6)
            for key, value in final_metrics.items()
            if key.startswith("val/")
        }
        print_training_result_summary(
            run_context=run_context,
            global_step=global_step,
            last_validation_step=history_callback.last_validation_step,
            best_checkpoint_monitor=getattr(checkpoint_callback, "monitor", None),
            best_checkpoint_path=getattr(
                checkpoint_callback, "best_model_path", None
            ),
            best_checkpoint_score=getattr(
                checkpoint_callback, "best_model_score", None
            ),
            final_metrics=final_metrics,
            artifact_paths={
                **artifacts,
                **log_files,
                **{
                    f"published_output/{name}": path
                    for name, path in published_outputs.items()
                },
            },
            artifact_sources={
                **{name: artifact_source for name in artifacts},
                **{
                    f"published_output/{name}": "stable output (atomic last-writer pointer)"
                    for name in published_outputs
                },
            },
            title=f"Stage II {adapt_method_label(method)} Adaptation Result · {phase}",
            rich=bool((adapt.get("console", {}) or {}).get("rich", True)),
        )
        print(
            json.dumps(
                {
                    "keyword": adapt_paths["keyword_str"],
                    "slug": adapt_paths["slug_str"],
                    "phase": phase,
                    "method": method,
                    "run_id": run_context.run_id,
                    "step": global_step,
                    "metrics": metrics,
                    "artifacts": {name: str(path) for name, path in artifacts.items()},
                    "published_outputs": {
                        name: str(path)
                        for name, path in published_outputs.items()
                    },
                    "logs": {name: str(path) for name, path in log_files.items()},
                }
            )
        )

    # With externally launched DDP all ranks execute this function. Keep non-zero
    # ranks alive until rank 0 has saved the phase handoff artifact, otherwise
    # the next phase can race ahead and silently start from the original base.
    trainer.strategy.barrier("stage2_adaptation_artifacts_saved")

    return artifacts
