#!/usr/bin/env python3
"""Train a minimal Stage I phoneme CTC model from prepared manifests."""

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
    parser.add_argument("--train-manifest", default="", help="Override train manifest path")
    parser.add_argument("--vocab", default="", help="Override phoneme vocab path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--devices", type=int, default=1, help="Number of visible GPUs to use via DataParallel")
    parser.add_argument("--limit-steps", type=int, default=0, help="Optional training step cap for smoke runs")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sum_loss_part(value):
    if hasattr(value, "sum"):
        return value.sum()
    if isinstance(value, (list, tuple)):
        return sum(value)
    return value


def reduce_stage1_loss(loss_output):
    """Reduce possibly gathered DataParallel CTC loss parts to one scalar.

    ``Stage1CTCModel.forward`` returns ``(local_loss_sum, local_batch_size)``.
    With ``nn.DataParallel`` each per-GPU scalar is gathered into a vector, so
    reducing both parts here keeps ``backward()`` scalar-valued and preserves a
    proper batch-size weighted mean for uneven last batches.
    """
    loss_sum, batch_size = loss_output
    return _sum_loss_part(loss_sum) / _sum_loss_part(batch_size)


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi
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
    from models.encoder import ConformerEncoder

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
            wav_path = record["wav_path"]
            waveform, sr = torchaudio.load(wav_path)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != sample_rate:
                waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
            feat = kaldi.fbank(
                waveform,
                num_mel_bins=num_mel_bins,
                frame_length=25,
                frame_shift=10,
                dither=0.1,
                sample_frequency=sample_rate,
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

    class Stage1CTCModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            encoder_dim = int(stage1.get("encoder_output_dim", 144))
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
            self.ctc = CTC(
                odim=len(vocab.token_to_id),
                encoder_output_size=encoder_dim,
                dropout_rate=float(stage1.get("ctc_dropout", 0.0)),
                blank_id=0,
            )

        def forward(self, feats, feat_lengths, targets, target_lengths):
            encoder_out, encoder_mask = self.encoder(feats, feat_lengths)
            encoder_lens = encoder_mask.squeeze(1).sum(1)
            loss, _ = self.ctc(encoder_out, encoder_lens, targets, target_lengths)
            batch_size = loss.new_tensor(feats.size(0))
            return loss * batch_size, batch_size

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

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = Stage1CTCModel().to(device)
    if args.devices > 1 and device.type == "cuda" and torch.cuda.device_count() >= args.devices:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.devices)))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(stage1.get("learning_rate", 1e-3)))
    max_epochs = int(stage1.get("max_epochs", 1))
    max_steps = args.limit_steps or int(stage1.get("max_train_steps", 0))
    checkpoint_dir = Path(stage1.get("checkpoint_dir", Path(paths["exp_root"]) / "stage1_phoneme_ctc" / "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    model.train()
    for epoch in range(max_epochs):
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = reduce_stage1_loss(
                model(batch["feats"], batch["feat_lengths"], batch["targets"], batch["target_lengths"])
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(stage1.get("gradient_clip_val", 1.0)))
            optimizer.step()
            global_step += 1
            if global_step % int(stage1.get("log_interval", 10)) == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item():.4f}")
            if max_steps and global_step >= max_steps:
                break
        ckpt_model = model.module if hasattr(model, "module") else model
        ckpt_path = checkpoint_dir / f"stage1_epoch{epoch:03d}_step{global_step:06d}.pt"
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
        if max_steps and global_step >= max_steps:
            break


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
