#!/usr/bin/env python3
"""Run the two-stage DMA-KWS demo with trained Stage I and Stage II checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.audio import extract_fbank, load_audio
from dma_kws.config import load_config, require_sections
from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.nn import build_encoder
from dma_kws.phonemes import PhonemeVocabulary
from dma_kws.stage1.candidates import PhonemeFrame, find_keyword_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--stage1-ckpt", required=True, help="Stage I checkpoint produced by train_stage1_ctc.py")
    parser.add_argument("--stage2-ckpt", required=True, help="Stage II checkpoint produced by train_stage2_qbyt.py")
    parser.add_argument("--audio", required=True, help="Input audio path")
    parser.add_argument("--keyword", required=True, help="Keyword text")
    parser.add_argument("--vocab", default="", help="Override phoneme vocab path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    return parser.parse_args()


def num_fbank_frames(
    num_samples: int,
    *,
    sample_rate: int,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
) -> int:
    frame_length_samples = round(sample_rate * frame_length_ms / 1000.0)
    frame_shift_samples = round(sample_rate * frame_shift_ms / 1000.0)
    if num_samples < frame_length_samples:
        return 0
    return 1 + (num_samples - frame_length_samples) // frame_shift_samples


def min_samples_for_fbank_frames(
    min_frames: int,
    *,
    sample_rate: int,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
) -> int:
    if min_frames <= 0:
        return 0
    frame_length_samples = round(sample_rate * frame_length_ms / 1000.0)
    frame_shift_samples = round(sample_rate * frame_shift_ms / 1000.0)
    return frame_length_samples + (min_frames - 1) * frame_shift_samples


def has_min_fbank_frames(
    num_samples: int,
    *,
    min_frames: int,
    sample_rate: int,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
) -> bool:
    return num_samples >= min_samples_for_fbank_frames(
        min_frames,
        sample_rate=sample_rate,
        frame_length_ms=frame_length_ms,
        frame_shift_ms=frame_shift_ms,
    )


def load_model_state(model, ckpt_path: str, load_fn):
    ckpt = load_fn(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    return model


def run(args: argparse.Namespace) -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc

    qbyt_root = PROJECT_ROOT / "qbyt"
    if str(qbyt_root) not in sys.path:
        sys.path.insert(0, str(qbyt_root))
    from model import QbyT
    from models.ctc import CTC

    config = load_config(args.config)
    require_sections(config, ["paths", "stage1", "stage2", "demo"])
    paths = config["paths"]
    stage1 = config["stage1"]
    stage2 = config["stage2"]
    demo = config["demo"]

    vocab_path = Path(args.vocab) if args.vocab else Path(paths["processed_root"]) / "stage1_phoneme_ctc" / "phoneme_vocab.txt"
    vocab = PhonemeVocabulary.read(vocab_path)
    g2p = make_g2p()
    keyword_phonemes = text_to_phonemes(g2p, args.keyword)
    keyword_ids = vocab.encode(keyword_phonemes)

    sample_rate = int(stage1.get("sample_rate", 16000))
    num_mel_bins = int(stage1.get("input_dim", 80))
    encoder_dim = int(stage1.get("encoder_output_dim", 144))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    class Stage1Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = build_encoder(stage1, output_dim=encoder_dim)
            self.ctc = CTC(len(vocab.token_to_id), encoder_dim, blank_id=0)

        def forward(self, feats, feat_lengths):
            encoder_out, encoder_mask = self.encoder(feats, feat_lengths)
            return self.ctc.log_softmax(encoder_out), encoder_mask

    class Stage2Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            stage2_encoder_dim = int(stage2.get("encoder_output_dim", 144))
            self.encoder = build_encoder(stage1, output_dim=stage2_encoder_dim)
            self.qbyt = QbyT(
                encoder_output_size=stage2_encoder_dim,
                num_embeds=len(vocab.token_to_id),
                embed_dim=int(stage2.get("qbyt_embed_dim", 128)),
                post_num_layers=int(stage2.get("qbyt_layers", 2)),
            )

        def forward(self, feats, feat_lengths, anchors, anchor_lengths):
            encoder_out, encoder_mask = self.encoder(feats, feat_lengths)
            encoder_lens = encoder_mask.squeeze(1).sum(1)
            logits, _ = self.qbyt(encoder_out, anchors, encoder_lens, anchor_lengths)
            return torch.sigmoid(logits)

    def load_state(model, ckpt_path: str):
        try:
            load_model_state(model, ckpt_path, torch.load)
        except RuntimeError as exc:
            raise SystemExit(f"Checkpoint {ckpt_path} is incompatible with the model architecture: {exc}") from exc
        return model.to(device).eval()

    waveform, sample_rate = load_audio(args.audio, sample_rate=sample_rate)
    feat = extract_fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        sample_rate=sample_rate,
        dither=0.0,
    ).unsqueeze(0)
    feat_lengths = torch.tensor([feat.size(1)], dtype=torch.long)

    stage1_model = load_state(Stage1Model(), args.stage1_ckpt)
    with torch.no_grad():
        log_probs, mask = stage1_model(feat.to(device), feat_lengths.to(device))
    token_ids = log_probs.argmax(dim=-1).squeeze(0).cpu().tolist()
    valid_len = int(mask.squeeze(1).sum(1).item())

    decoded_frames: list[PhonemeFrame] = []
    previous = 0
    output_frame_shift_sec = 0.04
    for index, token_id in enumerate(token_ids[:valid_len]):
        if token_id == 0 or token_id == previous:
            previous = token_id
            continue
        phoneme = vocab.id_to_token.get(token_id, "<unk>")
        decoded_frames.append(
            PhonemeFrame(
                phoneme=phoneme,
                start_sec=index * output_frame_shift_sec,
                end_sec=(index + 1) * output_frame_shift_sec,
                log_score=float(log_probs[0, index, token_id].cpu().item()),
            )
        )
        previous = token_id

    candidates = find_keyword_candidates(
        decoded_frames,
        keyword_phonemes,
        margin_sec=float(demo.get("stage1_candidate_margin_sec", 0.15)),
        max_insertions=int(demo.get("max_stage1_insertions", 2)),
    )

    stage2_model = load_state(Stage2Model(), args.stage2_ckpt)
    stage2_scores = []
    anchor = torch.tensor([keyword_ids], dtype=torch.long).to(device)
    anchor_lengths = torch.tensor([len(keyword_ids)], dtype=torch.long).to(device)
    min_stage2_fbank_frames = int(demo.get("min_stage2_fbank_frames", 7))
    for candidate in candidates:
        start = max(0, int(candidate.start_sec * sample_rate))
        end = min(waveform.size(1), int(candidate.end_sec * sample_rate))
        if end <= start:
            continue
        if not has_min_fbank_frames(
            end - start,
            min_frames=min_stage2_fbank_frames,
            sample_rate=sample_rate,
        ):
            continue
        candidate_wave = waveform[:, start:end]
        candidate_feat = extract_fbank(
            candidate_wave,
            num_mel_bins=num_mel_bins,
            sample_rate=sample_rate,
            dither=0.0,
        ).unsqueeze(0)
        candidate_lens = torch.tensor([candidate_feat.size(1)], dtype=torch.long)
        with torch.no_grad():
            score = float(
                stage2_model(
                    candidate_feat.to(device),
                    candidate_lens.to(device),
                    anchor,
                    anchor_lengths,
                )
                .cpu()
                .item()
            )
        stage2_scores.append(
            {
                "start_sec": candidate.start_sec,
                "end_sec": candidate.end_sec,
                "stage1_score": candidate.stage1_score,
                "qbyt_score": score,
            }
        )

    threshold = float(demo.get("qbyt_threshold", 0.5))
    result = {
        "audio": args.audio,
        "keyword": args.keyword,
        "keyword_phonemes": keyword_phonemes,
        "decoded_phonemes": [frame.phoneme for frame in decoded_frames],
        "stage1_candidates": [candidate.__dict__ for candidate in candidates],
        "stage2_scores": stage2_scores,
        "threshold": threshold,
        "detected": any(item["qbyt_score"] >= threshold for item in stage2_scores),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
