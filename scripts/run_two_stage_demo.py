#!/usr/bin/env python3
"""Run the two-stage DMA-KWS demo with trained Stage I and Stage II checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.audio import extract_fbank, load_audio
from dma_kws.config import get_tokenizer_config, require_sections
from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.nn import build_encoder
from dma_kws.pathing import ensure_qbyt_on_path, resolve_dict_path
from dma_kws.stage1.streaming_search import build_keyword_context_graph, decode_keyword_candidates
from dma_kws.tokenizer import load_char_tokenizer, tokenize_phoneme_string
from dma_kws.training.device import resolve_accelerator

NUM_EMBEDS = 73


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


def run(cfg: DictConfig) -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc

    ensure_qbyt_on_path()
    from model import QbyT
    from models.ctc import CTC
    from models.search import ctc_greedy_search

    config = resolved_config(cfg)
    require_sections(config, ["paths", "stage1", "stage2", "demo", "tokenizer"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}
    run_cfg = cfg.run

    stage1 = config["stage1"]
    stage2 = config["stage2"]
    demo = config["demo"]

    stage1_ckpt = str(prep.get("stage1_ckpt", ""))
    stage2_ckpt = str(prep.get("stage2_ckpt", ""))
    audio_path = str(prep.get("audio", ""))
    keyword = str(prep.get("keyword", ""))
    if not all([stage1_ckpt, stage2_ckpt, audio_path, keyword]):
        raise SystemExit("prep.stage1_ckpt, prep.stage2_ckpt, prep.audio, and prep.keyword are required")

    tokenizer_cfg = get_tokenizer_config(config)
    dict_path = resolve_dict_path(config)
    split_with_space = tokenizer_cfg.get("split_with_space", " ")
    tokenizer = load_char_tokenizer(dict_path, split_with_space=split_with_space)

    g2p = make_g2p()
    keyword_phonemes = text_to_phonemes(g2p, keyword)
    keyword_g2p_text = " ".join(keyword_phonemes)
    keyword_ids = tokenize_phoneme_string(tokenizer, keyword_g2p_text)

    sample_rate = int(stage1.get("sample_rate", 16000))
    num_mel_bins = int(stage1.get("input_dim", 80))
    encoder_dim = int(stage1.get("encoder_output_dim", 144))
    accelerator, _ = resolve_accelerator(str(run_cfg.device))
    device = torch.device(accelerator if accelerator == "cpu" else "cuda")

    class Stage1Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = build_encoder(stage1, output_dim=encoder_dim)
            self.ctc = CTC(NUM_EMBEDS, encoder_dim, blank_id=0)

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
                num_embeds=NUM_EMBEDS,
                embed_dim=int(stage2.get("qbyt_embed_dim", 128)),
                post_num_layers=int(stage2.get("qbyt_layers", 2)),
            )

        def forward(self, feats, feat_lengths, anchors, anchor_lengths):
            encoder_out, encoder_mask = self.encoder(feats, feat_lengths)
            encoder_lens = encoder_mask.squeeze(1).sum(1)
            logits, _ = self.qbyt(
                encoder_out,
                anchors,
                speech_lengths=encoder_lens,
                text_lengths=anchor_lengths,
            )
            return torch.sigmoid(logits)

    def load_state(model, ckpt_path: str):
        try:
            load_model_state(model, ckpt_path, torch.load)
        except RuntimeError as exc:
            raise SystemExit(f"Checkpoint {ckpt_path} is incompatible with the model architecture: {exc}") from exc
        return model.to(device).eval()

    waveform, sample_rate = load_audio(audio_path, sample_rate=sample_rate)
    feat = extract_fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        sample_rate=sample_rate,
        dither=0.0,
    ).unsqueeze(0)
    feat_lengths = torch.tensor([feat.size(1)], dtype=torch.long)

    stage1_model = load_state(Stage1Model(), stage1_ckpt)
    with torch.no_grad():
        log_probs, mask = stage1_model(feat.to(device), feat_lengths.to(device))
    encoder_lens = mask.squeeze(1).sum(1).to(torch.long)

    context_graph = build_keyword_context_graph(keyword_ids, tokenizer.symbol_table)
    output_frame_shift_sec = 0.04
    candidates = decode_keyword_candidates(
        log_probs,
        encoder_lens,
        context_graph,
        frame_shift_sec=output_frame_shift_sec,
        margin_sec=float(demo.get("stage1_candidate_margin_sec", 0.15)),
        symbol_table=tokenizer.symbol_table,
    )

    greedy_results = ctc_greedy_search(log_probs, encoder_lens, blank_id=0)
    decoded_phonemes = tokenizer.ids2tokens(greedy_results[0].tokens)

    stage2_model = load_state(Stage2Model(), stage2_ckpt)
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
        "audio": audio_path,
        "keyword": keyword,
        "keyword_phonemes": keyword_phonemes,
        "decoded_phonemes": decoded_phonemes,
        "stage1_candidates": [candidate.__dict__ for candidate in candidates],
        "stage2_scores": stage2_scores,
        "threshold": threshold,
        "detected": any(item["qbyt_score"] >= threshold for item in stage2_scores),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
