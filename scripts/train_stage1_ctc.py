#!/usr/bin/env python3
"""Train a minimal Stage I phoneme CTC model from prepared manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.audio import extract_fbank, load_audio
from dma_kws.config import load_config, require_sections
from dma_kws.jsonl import read_jsonl
from dma_kws.nn import build_encoder
from dma_kws.phonemes import PhonemeVocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--train-manifest", default="", help="Override train manifest path")
    parser.add_argument("--vocab", default="", help="Override phoneme vocab path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices for the Lightning Trainer")
    parser.add_argument("--limit-steps", type=int, default=0, help="Optional training step cap for smoke runs")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    try:
        import pytorch_lightning as pl
        import torch
        from torch.nn.utils.rnn import pad_sequence
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc

    qbyt_root = PROJECT_ROOT / "qbyt"
    if str(qbyt_root) not in sys.path:
        sys.path.insert(0, str(qbyt_root))
    from models.ctc import CTC

    config = load_config(args.config)
    require_sections(config, ["paths", "stage1"])
    paths = config["paths"]
    stage1 = config["stage1"]

    processed_dir = Path(paths["processed_root"]) / "stage1_phoneme_ctc"
    train_manifest = Path(args.train_manifest) if args.train_manifest else processed_dir / "train.jsonl"
    vocab_path = Path(args.vocab) if args.vocab else processed_dir / "phoneme_vocab.txt"
    vocab = PhonemeVocabulary.read(vocab_path)

    sample_rate = int(stage1.get("sample_rate", 16000))
    num_mel_bins = int(stage1.get("input_dim", 80))

    class Stage1Dataset(Dataset):
        def __init__(self, manifest_path: Path):
            self.records = read_jsonl(manifest_path)

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict:
            record = self.records[index]
            waveform, sr = load_audio(record["wav_path"], sample_rate=sample_rate)
            feat = extract_fbank(
                waveform,
                num_mel_bins=num_mel_bins,
                sample_rate=sr,
                dither=0.1,
            )
            target = torch.tensor(vocab.encode(record["phonemes"]), dtype=torch.long)
            return {"feat": feat, "target": target}

    def collate_fn(batch: list[dict]) -> dict:
        feats = [item["feat"] for item in batch]
        targets = [item["target"] for item in batch]
        return {
            "feats": pad_sequence(feats, batch_first=True, padding_value=0.0),
            "feat_lengths": torch.tensor([feat.size(0) for feat in feats], dtype=torch.long),
            "targets": pad_sequence(targets, batch_first=True, padding_value=0),
            "target_lengths": torch.tensor([target.size(0) for target in targets], dtype=torch.long),
        }

    class Stage1Module(pl.LightningModule):
        def __init__(self):
            super().__init__()
            encoder_dim = int(stage1.get("encoder_output_dim", 144))
            self.encoder = build_encoder(stage1, output_dim=encoder_dim)
            self.ctc = CTC(
                odim=len(vocab.token_to_id),
                encoder_output_size=encoder_dim,
                dropout_rate=float(stage1.get("ctc_dropout", 0.0)),
                blank_id=0,
            )

        def training_step(self, batch, batch_idx):
            encoder_out, encoder_mask = self.encoder(batch["feats"], batch["feat_lengths"])
            encoder_lens = encoder_mask.squeeze(1).sum(1)
            loss, _ = self.ctc(encoder_out, encoder_lens, batch["targets"], batch["target_lengths"])
            self.log("train_loss", loss, prog_bar=True, batch_size=batch["feats"].size(0))
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=float(stage1.get("learning_rate", 1e-3)))

    dataset = Stage1Dataset(train_manifest)
    if len(dataset) == 0:
        raise SystemExit(f"No records found in {train_manifest}")

    dataloader = DataLoader(
        dataset,
        batch_size=int(stage1.get("batch_size_per_gpu", 16)) * max(1, int(args.devices)),
        shuffle=True,
        num_workers=int(stage1.get("num_workers", 2)),
        collate_fn=collate_fn,
    )

    model = Stage1Module()
    max_epochs = int(stage1.get("max_epochs", 1))
    max_steps = args.limit_steps or int(stage1.get("max_train_steps", 0))
    accelerator = "gpu" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=max(1, int(args.devices)),
        max_epochs=max_epochs,
        max_steps=max_steps if max_steps else -1,
        gradient_clip_val=float(stage1.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(stage1.get("log_interval", 10)),
    )
    trainer.fit(model, dataloader)

    global_step = int(trainer.global_step)
    checkpoint_dir = Path(stage1.get("checkpoint_dir", Path(paths["exp_root"]) / "stage1_phoneme_ctc" / "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"stage1_step{global_step:06d}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab.token_to_id,
            "config": config,
            "step": global_step,
        },
        ckpt_path,
    )
    print(f"Saved {ckpt_path}")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
