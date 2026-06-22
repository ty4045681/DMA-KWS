#!/usr/bin/env python3
"""Train a minimal Stage II QbyT verifier from prepared pair JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.config import load_config, require_sections
from dma_kws.phonemes import PhonemeVocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--pairs", default="", help="Override Stage II pair JSONL path")
    parser.add_argument("--vocab", default="", help="Override phoneme vocab path")
    parser.add_argument("--stage1-ckpt", default="", help="Optional Stage I checkpoint for encoder init")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--devices", type=int, default=1, help="Number of visible GPUs to use via DataParallel")
    parser.add_argument("--limit-steps", type=int, default=0, help="Optional training step cap for smoke runs")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi
        import torch.nn.functional as F
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
    from models.encoder import ConformerEncoder

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
            feat = kaldi.fbank(
                waveform,
                num_mel_bins=num_mel_bins,
                frame_length=25,
                frame_shift=10,
                dither=0.1,
                sample_frequency=sample_rate,
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

    class Stage2QbyTModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            encoder_dim = int(stage2.get("encoder_output_dim", 144))
            self.encoder = ConformerEncoder(
                input_size=int(stage1.get("input_dim", 80)),
                output_size=encoder_dim,
                attention_heads=int(stage1.get("attention_heads", 4)),
                linear_units=int(stage1.get("linear_units", 576)),
                num_blocks=int(stage1.get("num_blocks", 6)),
                dropout_rate=float(stage1.get("dropout_rate", 0.1)),
                positional_dropout_rate=float(stage1.get("positional_dropout_rate", 0.1)),
                attention_dropout_rate=float(stage1.get("attention_dropout_rate", 0.0)),
                use_cnn_module=True,
                input_layer="conv2d",
                pos_enc_layer_type="rel_pos",
                selfattention_layer_type="rel_selfattn",
                cnn_module_kernel=int(stage1.get("cnn_module_kernel", 3)),
            )
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

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = Stage2QbyTModel().to(device)

    if args.stage1_ckpt:
        ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        encoder_state = {key.replace("encoder.", "", 1): value for key, value in state.items() if key.startswith("encoder.")}
        missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
        print(f"Loaded Stage I encoder weights: missing={len(missing)} unexpected={len(unexpected)}")

    if args.devices > 1 and device.type == "cuda" and torch.cuda.device_count() >= args.devices:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.devices)))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(stage2.get("learning_rate", 1e-3)))
    max_steps = args.limit_steps or int(stage2.get("max_steps", 50000))
    checkpoint_dir = Path(stage2.get("checkpoint_dir", Path(paths["exp_root"]) / "stage2_qbyt" / "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    model.train()
    while global_step < max_steps:
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch["feats"], batch["feat_lengths"], batch["anchors"], batch["anchor_lengths"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(stage2.get("gradient_clip_val", 1.0)))
            optimizer.step()
            global_step += 1
            if global_step % int(stage2.get("log_interval", 10)) == 0:
                print(f"step={global_step} loss={loss.item():.4f}")
            if global_step >= max_steps:
                break

    ckpt_model = model.module if hasattr(model, "module") else model
    ckpt_path = checkpoint_dir / f"stage2_step{global_step:06d}.pt"
    torch.save(
        {
            "model_state_dict": ckpt_model.state_dict(),
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
