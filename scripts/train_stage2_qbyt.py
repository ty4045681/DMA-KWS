#!/usr/bin/env python3
"""Train a minimal Stage II QbyT verifier from prepared pair JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.audio import extract_fbank
from dma_kws.config import load_config, require_sections
from dma_kws.jsonl import read_jsonl
from dma_kws.nn import build_encoder
from dma_kws.phonemes import PhonemeVocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--pairs", default="", help="Override Stage II pair JSONL path")
    parser.add_argument("--vocab", default="", help="Override phoneme vocab path")
    parser.add_argument("--stage1-ckpt", default="", help="Optional Stage I checkpoint for encoder init")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--devices", type=int, default=1, help="Number of visible GPUs to use")
    parser.add_argument("--limit-steps", type=int, default=0, help="Optional training step cap for smoke runs")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        import torch.nn.functional as F
        import pytorch_lightning as pl
        from torch.nn.utils.rnn import pad_sequence
        from torch.utils.data import DataLoader, Dataset
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc

    qbyt_root = PROJECT_ROOT / "qbyt"
    if str(qbyt_root) not in sys.path:
        sys.path.insert(0, str(qbyt_root))
    from model import QbyT

    config = load_config(args.config)
    require_sections(config, ["paths", "stage1", "stage2"])
    paths = config["paths"]
    stage1 = config["stage1"]
    stage2 = config["stage2"]

    processed_root = Path(paths["processed_root"])
    pair_path = Path(args.pairs) if args.pairs else processed_root / "stage2_qbyt" / "train.jsonl"
    vocab_path = Path(args.vocab) if args.vocab else processed_root / "stage1_phoneme_ctc" / "phoneme_vocab.txt"
    vocab = PhonemeVocabulary.read(vocab_path)

    sample_rate = int(stage1.get("sample_rate", 16000))
    num_mel_bins = int(stage1.get("input_dim", 80))
    stage2_dir = processed_root / "stage2_qbyt"

    def resolve_wav_path(raw_path: str) -> str:
        path = Path(raw_path)
        if path.is_absolute():
            return str(path)
        return str(stage2_dir / path)

    class Stage2Dataset(Dataset):
        def __init__(self, pairs_path: Path):
            self.records = read_jsonl(pairs_path)

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict:
            record = self.records[index]
            wav_path = resolve_wav_path(record["wav_path"])
            audio = np.load(wav_path).astype("float32")
            waveform = torch.from_numpy(audio).reshape(1, -1)
            feat = extract_fbank(
                waveform,
                num_mel_bins=num_mel_bins,
                sample_rate=sample_rate,
                dither=0.1,
            )
            anchor = torch.tensor(vocab.encode(record["anchor_phonemes"]), dtype=torch.long)
            label = torch.tensor(float(record["label"]), dtype=torch.float32)
            return {"feat": feat, "anchor": anchor, "label": label}

    def collate_fn(batch: list[dict]) -> dict:
        feats = [item["feat"] for item in batch]
        anchors = [item["anchor"] for item in batch]
        return {
            "feats": pad_sequence(feats, batch_first=True, padding_value=0.0),
            "feat_lengths": torch.tensor([feat.size(0) for feat in feats], dtype=torch.long),
            "anchors": pad_sequence(anchors, batch_first=True, padding_value=0),
            "anchor_lengths": torch.tensor([anchor.size(0) for anchor in anchors], dtype=torch.long),
            "labels": torch.stack([item["label"] for item in batch]),
        }

    class Stage2Module(pl.LightningModule):
        def __init__(self):
            super().__init__()
            encoder_dim = int(stage2.get("encoder_output_dim", 144))
            self.encoder = build_encoder(stage1, output_dim=encoder_dim)
            self.qbyt = QbyT(
                encoder_output_size=encoder_dim,
                num_embeds=len(vocab.token_to_id),
                embed_dim=int(stage2.get("qbyt_embed_dim", 128)),
                post_num_layers=int(stage2.get("qbyt_layers", 2)),
            )

        def forward(self, feats, feat_lengths, anchors, anchor_lengths):
            encoder_out, encoder_mask = self.encoder(feats, feat_lengths)
            encoder_lens = encoder_mask.squeeze(1).sum(1)
            logits, _ = self.qbyt(encoder_out, anchors, encoder_lens, anchor_lengths)
            return logits

        def training_step(self, batch, batch_idx):
            logits = self(batch["feats"], batch["feat_lengths"], batch["anchors"], batch["anchor_lengths"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=float(stage2.get("learning_rate", 1e-3)))

    dataset = Stage2Dataset(pair_path)
    if len(dataset) == 0:
        raise SystemExit(f"No records found in {pair_path}")

    dataloader = DataLoader(
        dataset,
        batch_size=int(stage2.get("batch_size_per_gpu", 64)) * max(1, int(args.devices)),
        shuffle=True,
        num_workers=int(stage2.get("num_workers", stage1.get("num_workers", 2))),
        collate_fn=collate_fn,
    )

    model = Stage2Module()

    if args.stage1_ckpt:
        ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        encoder_state = {key.replace("encoder.", "", 1): value for key, value in state.items() if key.startswith("encoder.")}
        missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
        print(f"Loaded Stage I encoder weights: missing={len(missing)} unexpected={len(unexpected)}")

    max_steps = args.limit_steps or int(stage2.get("max_steps", 50000))
    checkpoint_dir = Path(stage2.get("checkpoint_dir", Path(paths["exp_root"]) / "stage2_qbyt" / "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    accelerator = "gpu" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    devices = max(1, int(args.devices)) if accelerator == "gpu" else 1
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_steps=max_steps,
        gradient_clip_val=float(stage2.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(stage2.get("log_interval", 10)),
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, dataloader)

    global_step = int(trainer.global_step)
    ckpt_path = checkpoint_dir / f"stage2_step{global_step:06d}.pt"
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
