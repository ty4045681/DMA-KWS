"""Stage II LoRA continual adaptation training."""

from __future__ import annotations

import copy
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from dma_kws.pathing import resolve_dict_path
from dma_kws.phonemes import normalize_english_text
from dma_kws.stage2.adapt_dataset import (
    KeywordAdaptationDataset,
    MixedAdaptationDataset,
    TargetKeywordValDataset,
)
from dma_kws.stage2.adapt_paths import adapt_data_root, phase_manifest, slugify
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
)
from dma_kws.training.checkpoint_io import (
    STAGE2_BASE_FINGERPRINT_KEY,
    assert_qbyt_readout_version,
    assert_stream_policy_matches,
    fingerprint_stage2_base,
    stamp_qbyt_readout_version,
)
from dma_kws.training.ddp import process_rank
from dma_kws.training.lora import (
    count_lora_params,
    inject_qbyt_lora,
    load_lora_state_dict,
    lora_state_dict,
    merge_lora,
    normalize_lora_targets,
)


@dataclass
class Stage2AdaptArgs:
    """Runtime options for Stage II LoRA adaptation."""

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
    paths = config["paths"]
    keyword = str(adapt.get("keyword", ""))
    if not keyword:
        raise ValueError("adapt.keyword is required")

    slug = str(adapt.get("slug", "")) or slugify(keyword)
    data_root = Path(adapt.get("data_root", "")) if adapt.get("data_root") else adapt_data_root(
        config, keyword
    )
    if adapt.get("exp_root"):
        exp_root = Path(str(adapt["exp_root"]))
    else:
        exp_root = Path(paths["exp_root"]) / "stage2_adapt" / slug
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
        "adapter_path": exp_root / f"adapter_{slug}.pt",
        "merged_path": exp_root / "stage2_adapted.pt",
        "train_manifest": phase_manifest(data_root, phase, split="train"),
        "eval_manifest": phase_manifest(data_root, phase, split="eval"),
    }


def _apply_adapt_overrides(config: dict[str, Any], args: Stage2AdaptArgs) -> dict[str, Any]:
    """Merge ``adapt.params_file`` overrides (e.g. sweep best params) into the config."""
    if not args.params_file:
        return {}
    return merge_adapt_params(
        _adapt_section(config),
        load_adapt_params_file(args.params_file),
    )


def _validate_lora_runtime_config(stage2: dict[str, Any]) -> None:
    ema = stage2.get("ema", {}) or {}
    if bool(ema.get("enabled", False)):
        raise ValueError(
            "stage2.ema.enabled is not supported for LoRA adaptation: Lightning "
            "weight averaging also updates the frozen base and invalidates "
            "adapter/base identity. Keep EMA disabled."
        )

    precision = str(stage2.get("precision", "") or "").lower()
    if precision.endswith("-true") and precision != "32-true":
        raise ValueError(
            f"stage2.precision={precision!r} changes frozen base parameter dtypes "
            "after adapter identity is checked. Use 'bf16-mixed', '16-mixed', or "
            "'32-true' for LoRA adaptation."
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
        return frozenset(normalize_lora_targets(values))
    return value


def _validate_lora_metadata(
    checkpoint: dict[str, Any],
    *,
    source: str | Path,
    expected: dict[str, Any],
    required: tuple[str, ...],
) -> None:
    source_label = str(source)
    missing = [key for key in required if not _metadata_candidates(checkpoint, key)]
    if missing:
        raise SystemExit(
            f"Cannot resume LoRA checkpoint {source_label}: required metadata is "
            f"missing: {', '.join(missing)}."
        )

    for key, expected_value in expected.items():
        candidates = _metadata_candidates(checkpoint, key)
        if not candidates:
            continue
        try:
            normalized_expected = _normalize_lora_metadata(key, expected_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid expected LoRA {key}: {expected_value!r}") from exc
        for location, saved_value in candidates:
            try:
                normalized_saved = _normalize_lora_metadata(key, saved_value)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Cannot resume LoRA checkpoint {source_label}: invalid "
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
                    f"Cannot resume LoRA checkpoint {source_label}: "
                    f"{key}={saved_value!r} in {location}, expected "
                    f"{expected_value!r}."
                )


def _validate_base_fingerprint(
    checkpoint: dict[str, Any],
    *,
    source: str | Path,
    expected: str,
    allow_missing: bool = False,
    warn_if_missing: bool = False,
) -> None:
    candidates = _metadata_candidates(checkpoint, STAGE2_BASE_FINGERPRINT_KEY)
    if not candidates:
        message = (
            f"LoRA checkpoint {source} has no {STAGE2_BASE_FINGERPRINT_KEY}; "
            "its frozen Stage II base identity cannot be verified."
        )
        if not allow_missing:
            raise SystemExit(
                f"{message} Set adapt.allow_legacy_adapter=true only if the "
                "adapter's original base checkpoint is known to match."
            )
        if warn_if_missing:
            warnings.warn(
                f"{message} Proceeding because legacy adapter loading was "
                "explicitly enabled.",
                UserWarning,
                stacklevel=3,
            )
        return
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
    allow_legacy_adapter: bool = False,
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
            allow_missing=allow_legacy_adapter,
            warn_if_missing=allow_legacy_adapter,
        )

    adapter_state = checkpoint.get("lora_state_dict")
    if not isinstance(adapter_state, dict) or not adapter_state:
        raise SystemExit(
            f"Cannot resume LoRA adapter {source_label}: lora_state_dict is missing or empty."
        )
    return adapter_state


class Stage2LoraAdaptationModule(Stage2LightningModule):
    """Stage II module with frozen encoder/base QbyT and trainable LoRA adapters."""

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
        module_config = config
        if restoring_full_checkpoint:
            module_config = copy.deepcopy(config)
            adapter_cfg = module_config.get("stage2", {}).get("phoneme_adapter")
            if isinstance(adapter_cfg, dict):
                adapter_cfg["init_checkpoint"] = ""
        super().__init__(
            module_config,
            vocab_size=vocab_size,
            freeze_encoder=True,
            init_checkpoint=init_checkpoint,
            require_full_qbyt_init=bool(init_checkpoint),
        )
        for param in self.parameters():
            param.requires_grad = False
        # The phoneme adapter trunk is part of the shared Stage I/II forward
        # pass; letting LoRA move it would break the premise that both stages
        # read one encoder pass. Freeze it and turn off the auxiliary CTC loss.
        self.freeze_adapter = self.adapter is not None
        self.ctc_weight = 0.0

        self._base_model_sha256 = fingerprint_stage2_base(self.state_dict())
        normalized_targets = normalize_lora_targets(lora_targets)
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
            # ...and against a specific QbyT readout: LoRA only moves the matcher
            # attention, so a stale adapter would be re-pointed at a frame the
            # base weights never learned to read.
            assert_qbyt_readout_version(
                state,
                source=adapter_checkpoint,
                # The base QbyT is frozen; unlike a full warm start, a legacy
                # adapter cannot relearn the corrected pooled readout.
                allow_legacy=False,
            )
            adapter_state = _validate_adapter_checkpoint(
                state,
                source=adapter_checkpoint,
                keyword=str(checkpoint_adapt.get("keyword", "")),
                rank=self._lora_rank,
                alpha=self._lora_alpha,
                targets=self._lora_targets,
                base_model_sha256=self._base_model_sha256,
                allow_legacy_adapter=bool(
                    checkpoint_adapt.get("allow_legacy_adapter", False)
                ),
            )
            load_lora_state_dict(self.qbyt, adapter_state, strict=True)

        self.lora_param_counts = count_lora_params(self)
        self._adapt_cfg = _adapt_section(config)
        self.target_auc_metric = torchmetrics.AUROC(
            task="binary",
            sync_on_compute=True,
        )
        self.target_eer_metric = torchmetrics.classification.EER(
            task="binary",
            sync_on_compute=True,
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Make LoRA Lightning checkpoints self-describing for later export."""
        super().on_save_checkpoint(checkpoint)
        adapt = _adapt_section(self._checkpoint_config)
        keyword = str(adapt.get("keyword", ""))
        checkpoint.update(
            {
                "checkpoint_kind": "stage2_lora",
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
            allow_legacy=False,
        )
        super().on_load_checkpoint(checkpoint)
        adapt = _adapt_section(self._checkpoint_config)
        keyword = str(adapt.get("keyword", ""))
        _validate_lora_metadata(
            checkpoint,
            source="the LoRA checkpoint being restored",
            expected={
                "checkpoint_kind": "stage2_lora",
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
            # A full Lightning checkpoint embeds the base tensors themselves, so
            # its identity can be derived even when an older payload lacks the
            # redundant fingerprint field.
            allow_missing=True,
        )
        self._base_model_sha256 = actual_base_sha256

    def train(self, mode: bool = True):
        """Keep frozen feature extractors deterministic while LoRA is training."""
        super().train(mode)
        if mode:
            self.encoder.eval()
            if self.adapter is not None:
                self.adapter.eval()
        return self

    def configure_optimizers(self) -> dict:
        trainable = [param for param in self.parameters() if param.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable LoRA parameters found")
        adapt = self._adapt_cfg
        lr, _ = resolve_adapt_lr(adapt)
        optim_cfg = {
            "optimizer": str(adapt.get("optimizer", "adam")).lower(),
            "lr": lr,
            "weight_decay": float(adapt.get("weight_decay", 0.0)),
            "warmup_steps": int(adapt.get("warmup_steps", 100)),
            "total_steps": int(adapt.get("max_steps", 3000)),
        }

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
            keyword_mask = source.bool()
            per_sample = F.binary_cross_entropy_with_logits(
                logits, batch["label"].float(), reduction="none"
            )
            self.log("train/keyword_frac", keyword_mask.float().mean(), on_step=True)
            if keyword_mask.any():
                self.log("train/keyword_utt_loss", per_sample[keyword_mask].mean(), on_step=True)
            if (~keyword_mask).any():
                self.log("train/libri_utt_loss", per_sample[~keyword_mask].mean(), on_step=True)
        return total_loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logits, _ = self(batch["feat"], batch["feat_lengths"], batch["anchor"])
        preds = torch.sigmoid(logits)
        labels = batch["label"].int()

        if dataloader_idx == 0:
            utt_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            self.log(
                "val/target_utt_loss",
                utt_loss,
                prog_bar=True,
                on_epoch=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )
            self.target_auc_metric.update(preds, labels)
            self.target_eer_metric.update(preds, labels)
        else:
            utt_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            self.log(
                "val/utt_loss",
                utt_loss,
                prog_bar=True,
                on_epoch=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )
            self.auc_metric.update(preds, labels)
            self.eer_metric.update(preds, labels)

    def on_validation_epoch_end(self) -> None:
        target_auc = self.target_auc_metric.compute()
        target_eer = self.target_eer_metric.compute()
        self.log("val/target_auc", target_auc, prog_bar=True, sync_dist=True)
        self.log("val/target_eer", target_eer, prog_bar=True, sync_dist=True)

        lph_auc = self.auc_metric.compute()
        lph_eer = self.eer_metric.compute()
        self.log("val/auc", lph_auc, prog_bar=True, sync_dist=True)
        self.log("val/eer", lph_eer, prog_bar=True, sync_dist=True)
        # Checkpoint-filename alias of val/auc; kept out of CSV/TensorBoard.
        self.log("val_auc", lph_auc, sync_dist=True, logger=False)

        self.target_auc_metric.reset()
        self.target_eer_metric.reset()
        self.auc_metric.reset()
        self.eer_metric.reset()


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


def run_stage2_adaptation(config: dict[str, Any], args: Stage2AdaptArgs) -> dict[str, Path]:
    """Run Stage II LoRA adaptation for the configured keyword phase."""
    try:
        import pytorch_lightning as pl_mod
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/pytorch-lightning. Install CUDA PyTorch on the training machine first."
        ) from exc

    from dma_kws.config import get_tokenizer_config, require_sections
    from dma_kws.runlog import build_loggers
    from dma_kws.stage2 import adapt_console
    from dma_kws.stage2.collate import test_collate_fn, train_collate_fn
    from dma_kws.stage2.dataset import LibriPhraseTrainDataset, stage2_worker_init_fn
    from dma_kws.tokenizer import load_char_tokenizer
    from dma_kws.training import resolve_resume_path
    from dma_kws.training.callbacks import build_stage2_callbacks, print_run_summary
    from dma_kws.training.metrics_history import (
        append_wide_row,
        build_metrics_history_callback,
        build_run_record,
        collect_hparams,
        numeric_callback_metrics,
    )
    from dma_kws.training.ddp import apply_step_based_validation, build_trainer_kwargs
    from dma_kws.training.device import resolve_accelerator_and_devices

    _apply_adapt_overrides(config, args)
    require_sections(config, ["paths", "stage1", "stage2", "tokenizer", "training", "adapt"])

    adapt = _adapt_section(config)
    adapt_paths = _resolve_adapt_paths(config)
    paths = config["paths"]
    stage1 = config["stage1"]
    stage2 = config["stage2"]
    _validate_lora_runtime_config(stage2)
    training = config["training"]
    tokenizer_cfg = get_tokenizer_config(config)
    reporter = adapt_console.adapt_reporter(config)
    is_primary_process = process_rank() == 0

    train_manifest = adapt_paths["train_manifest"]
    eval_manifest = adapt_paths["eval_manifest"]
    if not train_manifest.exists():
        raise SystemExit(f"Adaptation train manifest not found: {train_manifest}")
    if not eval_manifest.exists():
        raise SystemExit(f"Adaptation eval manifest not found: {eval_manifest}")

    phase = adapt_paths["phase_str"]
    if args.limit_steps:
        # Keep the cosine schedule horizon in sync with the truncated run, otherwise
        # training stops while the LR is still on its way down.
        adapt["max_steps"] = int(args.limit_steps)
    accelerator, devices = resolve_accelerator_and_devices(args.device, args.devices)

    adapt_paths["phase_dir"].mkdir(parents=True, exist_ok=True)
    checkpoint_dir = adapt_paths["phase_dir"] / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = resolve_resume_path(args.resume_from, checkpoint_dir)
    if resume_path is not None and args.resume_checkpoint:
        raise ValueError(
            "Choose either run.resume_from (full Lightning state) or "
            "run.resume_checkpoint (adapter weights), not both."
        )
    if resume_path is not None:
        # A full Lightning checkpoint already contains encoder, QbyT, LoRA and
        # optimizer state. Loading the original base first makes resume depend on
        # a file that may have moved and can silently introduce a different base.
        init_checkpoint = ""
        adapter_resume = None
    else:
        init_checkpoint = _resolve_init_checkpoint(config, adapt_paths, args)
        adapter_resume = _resolve_adapter_resume(
            adapt_paths,
            phase,
            args.resume_checkpoint,
        )

    if resume_path is not None:
        assert_sequence_objective_matches(
            torch.load(resume_path, map_location="cpu"),
            stage2,
            source=resume_path,
        )

    if is_primary_process:
        reporter.section(f"LoRA adaptation · {adapt_paths['keyword_str']} · phase={phase}")
        reporter.print_plan(
            adapt_console.adapt_plan_rows(
                adapt_paths=adapt_paths,
                accelerator=accelerator,
                devices=devices,
                init_checkpoint=init_checkpoint,
                adapter_resume=adapter_resume,
                resume_path=resume_path,
                params_file=args.params_file,
            ),
            title="Adaptation Plan",
        )

    dict_path = resolve_dict_path(config)
    tokenizer = load_char_tokenizer(dict_path, split_with_space=tokenizer_cfg.get("split_with_space", " "))
    vocab_size = len(tokenizer._symbol_table)

    seed = int(training.get("seed", 2025))
    pl_mod.seed_everything(seed, workers=True)
    seq_label_mode = resolve_sequence_objective(stage2).target_mode
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
        seq_label_mode=seq_label_mode,
    )

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
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        drop_last=True,
        # Reseeds both the mixed wrapper and the LibriPhrase dataset it holds;
        # otherwise every worker draws the same mix decisions and negatives.
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
        **val_loader_kwargs,
    )
    lph_val_loader = _build_val_dataloader(config, tokenizer)

    mix_ratio = float(adapt.get("mix_ratio", 0.5))
    if is_primary_process:
        reporter.print_table(
            *adapt_console.dataset_table(
                [
                    (
                        "keyword train",
                        len(keyword_dataset),
                        adapt_console.label_breakdown(keyword_dataset),
                    ),
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

    lora_rank = int(adapt.get("rank", 16))
    lora_alpha = float(adapt.get("alpha", 32))
    lora_targets = normalize_lora_targets(
        adapt.get("lora_targets", ("in_proj_weight", "out_proj.weight"))
    )
    adapt["rank"] = lora_rank
    adapt["alpha"] = lora_alpha
    adapt["lora_targets"] = list(lora_targets)

    if resume_path is not None:
        model = Stage2LoraAdaptationModule(
            config,
            vocab_size=vocab_size,
            init_checkpoint=init_checkpoint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_targets=lora_targets,
            restoring_full_checkpoint=True,
        )
    else:
        model = Stage2LoraAdaptationModule(
            config,
            vocab_size=vocab_size,
            init_checkpoint=init_checkpoint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_targets=lora_targets,
            adapter_checkpoint=adapter_resume,
        )

    if is_primary_process:
        reporter.print_plan(
            adapt_console.lora_rows(
                rank=lora_rank,
                alpha=lora_alpha,
                targets=lora_targets,
                injected=getattr(model, "lora_injected", None),
                param_counts=model.lora_param_counts,
            ),
            title="LoRA Adapters",
        )

    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")

    log_dir = adapt_paths["phase_dir"] / "logs"
    run_name = f"adapt_{adapt_paths['slug_str']}_{phase}"
    loggers = build_loggers(log_dir, run_name, config=config)

    hparams = collect_hparams(config, section="adapt", extra={"slug": adapt_paths["slug_str"]})
    if is_primary_process:
        for train_logger in loggers:
            train_logger.log_hyperparams(hparams)

    recipe = str(training.get("recipe", "adapt"))
    callbacks = build_stage2_callbacks(config, recipe, checkpoint_dir=checkpoint_dir)
    history_callback = build_metrics_history_callback(
        run_name=run_name,
        default_dir=log_dir / run_name,
    )
    callbacks.append(history_callback)

    limit_steps = args.limit_steps or int(adapt.get("max_steps", 3000)) or None
    trainer_kwargs = build_trainer_kwargs(
        config,
        devices,
        limit_steps=limit_steps,
        accelerator=accelerator,
    )
    trainer_kwargs["max_steps"] = limit_steps

    # Adaptation runs on short virtual epochs, so it needs its own validation
    # cadence instead of inheriting the Stage II pretraining schedule.
    adapt_validation = adapt.get("validation", {}) or {}
    if adapt_validation.get("val_check_interval") is not None:
        trainer_kwargs["val_check_interval"] = int(adapt_validation["val_check_interval"])
    if adapt_validation.get("limit_val_batches") is not None:
        trainer_kwargs["limit_val_batches"] = adapt_validation["limit_val_batches"]
    # Adaptation validation is configured in global training steps. Always use
    # Lightning's step-based mode: the dataloader is sharded only later during
    # DDP setup, so comparing against its pre-DDP length is incorrect.
    apply_step_based_validation(
        trainer_kwargs,
        len(train_dataloader),
        force=True,
    )

    if is_primary_process:
        print_run_summary(
            config=config,
            devices=devices,
            accelerator=accelerator,
            section="adapt",
            train_samples=len(train_dataset),
            val_samples=len(target_val_dataset) + len(lph_val_loader.dataset),
            param_counts=model.lora_param_counts,
            extra_rows=[
                ("phase", phase),
                ("batches_per_epoch", str(len(train_dataloader))),
                (
                    "val_check_interval",
                    f"{trainer_kwargs['val_check_interval']} "
                    f"({'steps' if trainer_kwargs.get('check_val_every_n_epoch', 1) is None else 'batches/epoch'})",
                ),
                ("mix_ratio", str(mix_ratio)),
                ("lora_rank", str(lora_rank)),
                ("lora_alpha", str(lora_alpha)),
            ],
            paths={
                "train_manifest": train_manifest,
                "eval_manifest": eval_manifest,
                "checkpoint_dir": checkpoint_dir,
                "log_dir": log_dir,
                **(
                    {"init_checkpoint": Path(init_checkpoint)}
                    if init_checkpoint
                    else {"resume_checkpoint": resume_path}
                ),
            },
        )

    if is_primary_process:
        reporter.section(f"Training · {run_name}")
    trainer = pl_mod.Trainer(
        accelerator=accelerator,
        callbacks=callbacks,
        logger=loggers,
        **trainer_kwargs,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=[target_val_loader, lph_val_loader],
        ckpt_path=resume_path,
    )

    global_step = int(trainer.global_step)
    adapter_out = adapt_paths["phase_dir"] / f"adapter_{adapt_paths['slug_str']}.pt"
    merged_out = adapt_paths["merged_path"]
    final_adapter = adapt_paths["adapter_path"]
    artifacts = {
        "adapter": adapter_out,
        "merged": merged_out,
        "final_adapter": final_adapter,
    }

    if trainer.is_global_zero:
        base_model_sha256 = fingerprint_stage2_base(model.state_dict())
        if base_model_sha256 != model._base_model_sha256:
            raise RuntimeError(
                "The frozen Stage II base changed during LoRA training. Refusing "
                "to save an adapter whose recorded base identity would be false."
            )
        torch.save(
            stamp_qbyt_readout_version(
                {
                    "checkpoint_kind": "stage2_lora_adapter",
                    "lora_state_dict": lora_state_dict(model.qbyt),
                    "config": model._checkpoint_config,
                    "step": global_step,
                    "keyword": adapt_paths["keyword_str"],
                    "slug": adapt_paths["slug_str"],
                    "phase": phase,
                    "rank": lora_rank,
                    "alpha": lora_alpha,
                    "lora_targets": list(lora_targets),
                    STAGE2_BASE_FINGERPRINT_KEY: base_model_sha256,
                }
            ),
            adapter_out,
        )

        merge_lora(model.qbyt)
        torch.save(
            stamp_qbyt_readout_version(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model._checkpoint_config,
                    "step": global_step,
                    "keyword": adapt_paths["keyword_str"],
                    "slug": adapt_paths["slug_str"],
                    "phase": phase,
                    "tokenizer_dict_path": str(dict_path),
                    "vocab_size": vocab_size,
                }
            ),
            merged_out,
        )

        if adapter_out.resolve() != final_adapter.resolve():
            torch.save(torch.load(adapter_out, map_location="cpu"), final_adapter)

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
            ),
        )

        reporter.section("Artifacts")
        reporter.print_table(*adapt_console.artifact_rows(artifacts), title="Saved Checkpoints")
        log_files = {"runs_csv": runs_csv}
        if history_callback.csv_path is not None:
            log_files["eval_history"] = history_callback.csv_path
        reporter.print_plan(
            [(name, str(path)) for name, path in sorted(log_files.items())],
            title="Metrics CSVs",
        )
        metrics = {
            key: round(value, 6)
            for key, value in final_metrics.items()
            if key.startswith("val/")
        }
        if metrics:
            reporter.print_plan(
                [(key, f"{value:.4f}") for key, value in sorted(metrics.items())],
                title="Final Validation Metrics",
            )
        reporter.done(f"LoRA adaptation complete for phase {phase!r} at step {global_step}.")
        print(
            json.dumps(
                {
                    "keyword": adapt_paths["keyword_str"],
                    "slug": adapt_paths["slug_str"],
                    "phase": phase,
                    "step": global_step,
                    "metrics": metrics,
                    "artifacts": {name: str(path) for name, path in artifacts.items()},
                    "logs": {name: str(path) for name, path in log_files.items()},
                }
            )
        )

    # With externally launched DDP all ranks execute this function. Keep non-zero
    # ranks alive until rank 0 has saved the TTS adapter, otherwise the real phase
    # can race ahead and silently start without it.
    trainer.strategy.barrier("stage2_lora_artifacts_saved")

    return artifacts
