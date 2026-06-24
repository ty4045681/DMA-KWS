"""Stage I Wenet-aligned Conformer+CTC Lightning module."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from dma_kws.audio import extract_fbank, load_audio
from dma_kws.stage1.prepare_fbank import resolve_record_fbank_path
from dma_kws.config import get_tokenizer_config, require_sections
from dma_kws.jsonl import read_jsonl
from dma_kws.metrics import collapse_ctc, edit_distance
from dma_kws.nn import build_encoder
from dma_kws.runlog import build_loggers
from dma_kws.tokenizer import load_char_tokenizer, tokenize_phoneme_string
from dma_kws.training.checkpoint_avg import average_lightning_checkpoints
from dma_kws.training.scheduler import build_cosine_warmup_optimizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLANK_ID = 0


def _ensure_qbyt_on_path() -> None:
    qbyt_root = PROJECT_ROOT / "qbyt"
    qbyt_path = str(qbyt_root)
    if qbyt_path not in sys.path:
        sys.path.insert(0, qbyt_path)


def _load_ctc():
    _ensure_qbyt_on_path()
    from models.ctc import CTC

    return CTC


def phonemes_to_g2p_string(phonemes: list[str] | str) -> str:
    """Convert a phoneme list or space-separated string to a G2P target string."""
    if isinstance(phonemes, str):
        return phonemes.strip()
    return " ".join(phonemes)


def encode_manifest_target(record: dict[str, Any], tokenizer) -> list[int]:
    """Encode a manifest record using Wenet ``CharTokenizer``."""
    if "phonemes_g2p" in record:
        return tokenize_phoneme_string(tokenizer, str(record["phonemes_g2p"]))
    if "phonemes" in record:
        return tokenize_phoneme_string(tokenizer, phonemes_to_g2p_string(record["phonemes"]))
    raise KeyError("Manifest record must contain 'phonemes_g2p' or 'phonemes'")


class Stage1Dataset(Dataset):
    """JSONL manifest dataset for Stage I CTC training."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        tokenizer,
        sample_rate: int,
        num_mel_bins: int,
        fbank_root: Path | str | None = None,
        audio_root: Path | str | None = None,
    ) -> None:
        self.records = read_jsonl(manifest_path)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.fbank_root = Path(fbank_root) if fbank_root else None
        self.audio_root = Path(audio_root) if audio_root else None

    def __len__(self) -> int:
        return len(self.records)

    def _load_features(self, record: dict[str, Any]) -> torch.Tensor:
        fbank_path = resolve_record_fbank_path(
            record,
            fbank_root=self.fbank_root,
            audio_root=self.audio_root,
        )
        if fbank_path is not None and fbank_path.exists():
            import numpy as np

            return torch.from_numpy(np.load(fbank_path))

        waveform, sr = load_audio(record["wav_path"], sample_rate=self.sample_rate)
        return extract_fbank(
            waveform,
            num_mel_bins=self.num_mel_bins,
            sample_rate=sr,
            dither=0.1,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        feat = self._load_features(record)
        target = torch.tensor(
            encode_manifest_target(record, self.tokenizer),
            dtype=torch.long,
        )
        return {"feat": feat, "target": target}


def stage1_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    feats = [item["feat"] for item in batch]
    targets = [item["target"] for item in batch]
    return {
        "feats": pad_sequence(feats, batch_first=True, padding_value=0.0),
        "feat_lengths": torch.tensor([feat.size(0) for feat in feats], dtype=torch.long),
        "targets": pad_sequence(targets, batch_first=True, padding_value=0),
        "target_lengths": torch.tensor([target.size(0) for target in targets], dtype=torch.long),
    }


class Stage1LightningModule(pl.LightningModule):
    """Lightning wrapper for ConformerEncoder + CTC Stage I training."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        vocab_size: int,
        blank_id: int = BLANK_ID,
        num_decode_batches: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "vocab_size": vocab_size,
                "blank_id": blank_id,
                "num_decode_batches": num_decode_batches,
            }
        )

        stage1 = config["stage1"]
        encoder_dim = int(stage1.get("encoder_output_dim", 144))
        self.encoder = build_encoder(stage1, output_dim=encoder_dim)

        CTC = _load_ctc()
        self.ctc = CTC(
            odim=vocab_size,
            encoder_output_size=encoder_dim,
            dropout_rate=float(stage1.get("ctc_dropout", 0.0)),
            blank_id=blank_id,
        )

        self._stage1_cfg = stage1
        self.blank_id = blank_id
        self.num_decode_batches = num_decode_batches

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_out, encoder_mask = self.encoder(feats, feat_lengths)
        encoder_lens = encoder_mask.squeeze(1).sum(dim=1).to(dtype=torch.long)
        return encoder_out, encoder_lens

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        encoder_out, encoder_lens = self(batch["feats"], batch["feat_lengths"])
        loss, _ = self.ctc(
            encoder_out,
            encoder_lens,
            batch["targets"],
            batch["target_lengths"],
        )
        batch_size = batch["feats"].size(0)
        self.log("train/loss", loss, on_step=True, prog_bar=True, batch_size=batch_size)

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, prog_bar=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_total_dist = 0
        self._val_total_ref = 0

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        if self.num_decode_batches and batch_idx >= self.num_decode_batches:
            return

        encoder_out, encoder_lens = self(batch["feats"], batch["feat_lengths"])
        log_probs = self.ctc.log_softmax(encoder_out)
        preds = log_probs.argmax(dim=2)
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]

        for i in range(preds.size(0)):
            frames = int(encoder_lens[i].item())
            raw_ids = preds[i, :frames].tolist()
            hyp_ids = collapse_ctc(raw_ids, blank_id=self.blank_id)
            ref_len = int(target_lengths[i].item())
            ref_ids = targets[i, :ref_len].tolist()
            self._val_total_dist += edit_distance(ref_ids, hyp_ids)
            self._val_total_ref += ref_len

    def on_validation_epoch_end(self) -> None:
        if self._val_total_ref > 0:
            per = self._val_total_dist / self._val_total_ref
            self.log("val/per", per, prog_bar=True)

    def configure_optimizers(self) -> dict | torch.optim.Optimizer:
        stage1 = self._stage1_cfg
        lr = float(stage1.get("learning_rate", 1e-3))
        warmup_steps = int(stage1.get("warmup_steps", 0))
        total_steps = int(stage1.get("total_scheduler_steps", stage1.get("max_train_steps", 0)))

        if warmup_steps > 0 and total_steps > 0:
            return build_cosine_warmup_optimizer(self, lr, warmup_steps, total_steps)
        return torch.optim.Adam(self.parameters(), lr=lr)


@dataclass
class Stage1TrainArgs:
    """Runtime options for Stage I training."""

    train_manifest: str = ""
    dev_manifest: str = ""
    device: str = "cuda"
    devices: int = 1
    limit_steps: int = 0


def _resolve_dict_path(config: dict[str, Any]) -> Path:
    tokenizer_cfg = get_tokenizer_config(config)
    dict_path = Path(tokenizer_cfg["dict_path"])
    if not dict_path.is_absolute():
        dict_path = PROJECT_ROOT / dict_path
    return dict_path


def _resolve_manifest(
    override: str,
    default: Path,
    *,
    config_value: str = "",
) -> Path:
    if override:
        return Path(override)
    if config_value:
        return Path(config_value)
    return default


def export_stage1_encoder_pt(
    model: Stage1LightningModule,
    output_path: Path,
    *,
    config: dict[str, Any],
    dict_path: Path,
    vocab_size: int,
    blank_id: int,
    step: int,
) -> Path:
    """Save Stage I weights in a format loadable by Stage II ``init_checkpoint``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "step": step,
            "tokenizer_dict_path": str(dict_path),
            "vocab_size": vocab_size,
            "blank_id": blank_id,
        },
        output_path,
    )
    return output_path


def _maybe_average_checkpoints(
    checkpoint_dir: Path,
    stage1: dict[str, Any],
) -> Path | None:
    avg_cfg = stage1.get("checkpoint_avg", {}) or {}
    if not avg_cfg.get("enabled", False):
        return None

    last_k = int(avg_cfg.get("last_k", 10))
    pattern = str(avg_cfg.get("pattern", "*.ckpt"))
    output_name = str(avg_cfg.get("output_name", "avg_10.ckpt"))

    candidates = sorted(checkpoint_dir.glob(pattern))
    if not candidates:
        print(f"No checkpoints matched {pattern!r} in {checkpoint_dir}; skipping average.")
        return None

    selected = candidates[-last_k:]
    output_path = checkpoint_dir / output_name
    average_lightning_checkpoints(selected, output_path)
    print(f"Averaged {len(selected)} Stage I checkpoints -> {output_path}")
    return output_path


def run_stage1_training(config: dict[str, Any], args: Stage1TrainArgs) -> None:
    """Train Stage I CTC from a loaded config dict."""
    require_sections(config, ["paths", "stage1", "tokenizer", "training"])

    paths = config["paths"]
    stage1 = config["stage1"]
    training = config["training"]

    processed_dir = Path(paths["processed_root"]) / "stage1_phoneme_ctc"
    validation_cfg = stage1.get("validation", {}) or {}

    train_manifest = _resolve_manifest(
        args.train_manifest,
        processed_dir / "train.jsonl",
    )
    dev_manifest = _resolve_manifest(
        args.dev_manifest,
        processed_dir / "dev.jsonl",
        config_value=str(validation_cfg.get("dev_manifest", "")).strip(),
    )

    dict_path = _resolve_dict_path(config)
    split_with_space = get_tokenizer_config(config).get("split_with_space", " ")
    tokenizer = load_char_tokenizer(dict_path, split_with_space=split_with_space)
    vocab_size = len(tokenizer._symbol_table)
    blank_id = int(tokenizer.symbol_table.get("<blank>", BLANK_ID))

    sample_rate = int(stage1.get("sample_rate", 16000))
    num_mel_bins = int(stage1.get("input_dim", 80))
    num_decode_batches = int(validation_cfg.get("num_decode_batches", 0))
    num_workers = int(stage1.get("num_workers", 2))
    fbank_root = stage1.get("fbank_root", "")
    if not fbank_root:
        fbank_root = str(Path(paths.get("feature_root", "")) / "stage1_fbank")
    audio_root = stage1.get("audio_root", paths.get("librispeech_root", ""))

    pl.seed_everything(int(training.get("seed", 2025)), workers=True)

    dataset = Stage1Dataset(
        train_manifest,
        tokenizer=tokenizer,
        sample_rate=sample_rate,
        num_mel_bins=num_mel_bins,
        fbank_root=fbank_root or None,
        audio_root=audio_root or None,
    )
    if len(dataset) == 0:
        raise SystemExit(f"No records found in {train_manifest}")

    batch_size = int(stage1.get("batch_size_per_gpu", 16)) * max(1, int(args.devices))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=stage1_collate_fn,
    )

    dev_dataloader = None
    if dev_manifest.exists():
        dev_dataset = Stage1Dataset(
            dev_manifest,
            tokenizer=tokenizer,
            sample_rate=sample_rate,
            num_mel_bins=num_mel_bins,
            fbank_root=fbank_root or None,
            audio_root=audio_root or None,
        )
        if len(dev_dataset) > 0:
            dev_dataloader = DataLoader(
                dev_dataset,
                batch_size=int(validation_cfg.get("batch_size", stage1.get("batch_size_per_gpu", 16))),
                shuffle=False,
                num_workers=num_workers,
                collate_fn=stage1_collate_fn,
            )

    model = Stage1LightningModule(
        config,
        vocab_size=vocab_size,
        blank_id=blank_id,
        num_decode_batches=num_decode_batches,
    )

    max_epochs = int(stage1.get("max_epochs", 1))
    max_steps = args.limit_steps or int(stage1.get("max_train_steps", 0))
    accelerator = "gpu" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    devices = max(1, int(args.devices)) if accelerator == "gpu" else 1

    checkpoint_dir = Path(
        stage1.get("checkpoint_dir", Path(paths["exp_root"]) / "stage1_phoneme_ctc" / "checkpoints")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []
    checkpoint_callback = None
    trainer_kwargs: dict[str, Any] = {}
    if dev_dataloader is not None:
        from pytorch_lightning.callbacks import ModelCheckpoint

        avg_cfg = stage1.get("checkpoint_avg", {}) or {}
        save_all = bool(avg_cfg.get("enabled", False))
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="val/per",
            mode="min",
            save_top_k=1 if not save_all else -1,
            filename="stage1_{epoch:03d}_{val_per:.4f}",
        )
        callbacks.append(checkpoint_callback)
        trainer_kwargs["check_val_every_n_epoch"] = int(
            validation_cfg.get("check_val_every_n_epoch", 1)
        )

    log_dir = stage1.get("log_dir", Path(paths["exp_root"]) / "stage1_phoneme_ctc" / "logs")
    loggers = build_loggers(log_dir, str(stage1.get("run_name", "stage1_phoneme_ctc")))

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        max_steps=max_steps if max_steps else -1,
        gradient_clip_val=float(stage1.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(stage1.get("log_interval", 10)),
        logger=loggers,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    if dev_dataloader is not None:
        trainer.fit(model, dataloader, dev_dataloader)
    else:
        trainer.fit(model, dataloader)

    if checkpoint_callback is not None and checkpoint_callback.best_model_path:
        best_state = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
        model.load_state_dict(best_state["state_dict"])
        print(f"Loaded best Stage I weights from {checkpoint_callback.best_model_path}")

    avg_path = _maybe_average_checkpoints(checkpoint_dir, stage1)

    global_step = int(trainer.global_step)
    ckpt_path = export_stage1_encoder_pt(
        model,
        checkpoint_dir / f"stage1_step{global_step:06d}.pt",
        config=config,
        dict_path=dict_path,
        vocab_size=vocab_size,
        blank_id=blank_id,
        step=global_step,
    )
    print(f"Saved {ckpt_path}")

    if avg_path is not None:
        avg_state = torch.load(avg_path, map_location="cpu")
        model.load_state_dict(avg_state["state_dict"])
        avg_pt_path = export_stage1_encoder_pt(
            model,
            checkpoint_dir / "stage1_avg.pt",
            config=config,
            dict_path=dict_path,
            vocab_size=vocab_size,
            blank_id=blank_id,
            step=global_step,
        )
        print(f"Saved averaged Stage I encoder weights to {avg_pt_path}")
        print(f"Use {avg_path} or {avg_pt_path} as Stage II --init-checkpoint")
