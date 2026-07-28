#!/usr/bin/env python3
"""Control experiment: validate the PER probe before trusting its verdict.

``probe_frozen_per_on_libriphrase.py`` reports ~40% PER on LibriPhrase clips
while Step A reported 5.5% on LibriSpeech sentences. Before reading that as a
property of the frozen representation, two confounds have to be ruled out,
because either one produces the same number:

  A. the probe's model/decode path is wrong
  B. precomputed eval fbank != the on-the-fly features Step A trained on
     (``dma_kws/stage1/dataset.py`` warns these are not interchangeable: the
     fallback feeds kaldi.fbank unscaled while FbankExtractor scales by 1<<15)

Three measurements, one run:

  1. LibriSpeech dev, on-the-fly features   -- Step A's own setting. Must land
                                               near 0.0553 or the probe is broken.
  2. LibriPhrase clips, precomputed .npy    -- reproduces the ~40% number.
  3. LibriPhrase clips, on-the-fly from wav -- same clips, Step A's feature path.

  1 far from 0.0553      -> confound A: the probe is wrong, ignore its verdict.
  3 much better than 2   -> confound B: the precomputed eval fbank is the problem.
  1 ok and 2 ~= 3        -> the 40% is real: the frozen stack cannot read these
                            clips, and no readout on top of it will separate
                            hard negatives.

Usage:
  python scripts/probe_per_controls.py \
    +experiment=icefall_zipformer_stage2_adapter \
    stage2.init_checkpoint=/path/to/encoder.pt \
    stage2.phoneme_adapter.init_checkpoint=/path/to/adapter.pt \
    prep.limit=600
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import (
    fbank_kwargs,
    get_eval_fbank_config,
    get_fbank_config,
    get_tokenizer_config,
    require_sections,
)
from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.metrics import collapse_ctc, edit_distance
from dma_kws.pathing import resolve_dict_path
from dma_kws.stage2.dataset import (
    _build_eval_dataframe,
    _filter_eval_split,
    _resolve_eval_fbank_path,
    resolve_stage2_eval_paths,
)
from dma_kws.tokenizer import load_char_tokenizer, tokenize_phoneme_string

STEP_A_REFERENCE_PER = 0.0553


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Missing torch on this machine.") from exc

    from dma_kws.audio import load_audio
    from dma_kws.jsonl import read_jsonl
    from dma_kws.nn import run_encoder
    from dma_kws.stage1.dataset import encode_manifest_target
    from dma_kws.stage2.fbank import FbankExtractor
    from dma_kws.stage2.module import Stage2LightningModule

    config = resolved_config(cfg)
    require_sections(config, ["paths", "stage1", "stage2", "tokenizer"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}
    limit = int(prep.get("limit", 0))
    if limit <= 0:
        limit = 600

    tokenizer_cfg = get_tokenizer_config(config)
    dict_path = resolve_dict_path(config)
    tokenizer = load_char_tokenizer(
        dict_path, split_with_space=tokenizer_cfg.get("split_with_space", " ")
    )
    vocab_size = len(tokenizer._symbol_table)

    encoder_checkpoint = str(config["stage2"].get("init_checkpoint", "")).strip()
    if not encoder_checkpoint:
        raise SystemExit(
            "stage2.init_checkpoint is empty, so the encoder would stay random and every "
            "number below would be noise. Pass stage2.init_checkpoint=/path/to/encoder.pt"
        )
    if not Path(encoder_checkpoint).is_file():
        raise SystemExit(f"stage2.init_checkpoint not found: {encoder_checkpoint}")
    adapter_checkpoint = str(
        (config["stage2"].get("phoneme_adapter", {}) or {}).get("init_checkpoint", "")
    ).strip()
    if adapter_checkpoint and not Path(adapter_checkpoint).is_file():
        raise SystemExit(
            f"stage2.phoneme_adapter.init_checkpoint not found: {adapter_checkpoint}"
        )

    model = Stage2LightningModule(
        config, vocab_size=vocab_size, init_checkpoint=encoder_checkpoint
    )
    if model.adapter is None:
        raise SystemExit("Needs the adapter: stage2.phoneme_adapter.enabled=true")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    blank_id = model.adapter.blank_id

    # Step A's feature path: the training fbank profile, computed from the waveform.
    train_extractor = FbankExtractor(**fbank_kwargs(get_fbank_config(config)))
    # What the precomputed eval .npy files were supposed to be built with.
    eval_params = fbank_kwargs(get_eval_fbank_config(config))
    print(f"train fbank params: {fbank_kwargs(get_fbank_config(config))}")
    print(f"eval  fbank params: {eval_params}\n")

    def decode(feats: torch.Tensor) -> list[int]:
        feats = feats.unsqueeze(0).to(device)
        lengths = torch.tensor([feats.size(1)], dtype=torch.long, device=device)
        with torch.no_grad():
            encoder_out, encoder_mask = run_encoder(
                model.encoder, feats, lengths, policy=model.stream_policy, mode="eval"
            )
            _hidden, log_probs = model.adapter(encoder_out, encoder_mask, with_log_probs=True)
            frames = int(encoder_mask.squeeze(1).sum().item())
            return collapse_ctc(
                log_probs[0, :frames].argmax(dim=-1).tolist(), blank_id=blank_id
            )

    results: dict[str, dict] = {}

    def score(name: str, pairs) -> None:
        """``pairs`` yields ``(features, reference_ids)``."""
        dist = ref_len = n = 0
        feat_sum = feat_sq = feat_n = 0.0
        for feats, ref in pairs:
            if feats is None or not ref:
                continue
            arr = feats.numpy() if hasattr(feats, "numpy") else feats
            feat_sum += float(arr.sum())
            feat_sq += float((arr.astype("float64") ** 2).sum())
            feat_n += arr.size
            dist += edit_distance(ref, decode(feats))
            ref_len += len(ref)
            n += 1
        mean = feat_sum / feat_n if feat_n else 0.0
        variance = max(0.0, feat_sq / feat_n - mean * mean) if feat_n else 0.0
        std = variance**0.5
        results[name] = {
            "clips": n,
            "ref_phonemes": ref_len,
            "per": round(dist / ref_len, 4) if ref_len else None,
            "feat_mean": round(mean, 3),
            "feat_std": round(std, 3),
        }
        print(f"{name}: {json.dumps(results[name])}", flush=True)

    # --- 1. LibriSpeech dev, on-the-fly: Step A's exact setting -----------------
    sample_rate = int(config["stage1"].get("sample_rate", 16000))
    dev_manifest = Path(
        str(config["phoneme_adapter"].get("dev_manifest", "")).strip()
        or Path(config["paths"]["processed_root"]) / "stage1_phoneme_ctc" / "dev.jsonl"
    )
    if not dev_manifest.is_file():
        print(f"SKIP librispeech_dev_onthefly: manifest not found at {dev_manifest}")
    else:
        records = read_jsonl(dev_manifest)[:limit]

        def librispeech_pairs():
            for record in records:
                try:
                    waveform, sr = load_audio(record["wav_path"], sample_rate=sample_rate)
                except Exception:
                    continue
                yield train_extractor.extract(waveform, sr), encode_manifest_target(
                    record, tokenizer
                )

        score("1_librispeech_dev_onthefly", librispeech_pairs())

    # --- LibriPhrase clips, shared sample ---------------------------------------
    eval_paths = resolve_stage2_eval_paths(config)
    split = str(prep.get("split") or eval_paths["split"])
    frame = _filter_eval_split(
        _build_eval_dataframe(
            eval_paths["test_dir"], eval_paths["csv_files"], eval_paths["aggregate_csv"]
        ),
        split,
    )
    clips = (
        frame[["comparison", "comparison_text"]]
        .dropna()
        .drop_duplicates(subset=["comparison"])
        .reset_index(drop=True)
    )
    clips = clips.sample(n=min(limit, len(clips)), random_state=2025).reset_index(drop=True)
    g2p = make_g2p()
    references = {
        row.comparison: tokenize_phoneme_string(
            tokenizer, " ".join(text_to_phonemes(g2p, str(row.comparison_text)))
        )
        for row in clips.itertuples(index=False)
    }

    # --- 2. LibriPhrase, precomputed .npy: what the last probe read -------------
    def precomputed_pairs():
        for rel in clips["comparison"]:
            path = Path(
                _resolve_eval_fbank_path(
                    eval_paths["test_dir"], rel, fbank_dir=eval_paths["fbank_dir"]
                )
            )
            if not path.is_file():
                continue
            yield torch.from_numpy(np.load(path)), references[rel]

    score("2_libriphrase_precomputed_npy", precomputed_pairs())

    # --- 3. LibriPhrase, on-the-fly: Step A's feature path, same clips ----------
    def onthefly_pairs():
        for rel in clips["comparison"]:
            wav = Path(eval_paths["test_dir"]) / rel
            if not wav.is_file():
                continue
            try:
                waveform, sr = load_audio(str(wav), sample_rate=sample_rate)
            except Exception:
                continue
            yield train_extractor.extract(waveform, sr), references[rel]

    score("3_libriphrase_onthefly", onthefly_pairs())

    print("\n" + json.dumps({"step_a_reference_per": STEP_A_REFERENCE_PER, "results": results}))

    one = results.get("1_librispeech_dev_onthefly", {}).get("per")
    two = results.get("2_libriphrase_precomputed_npy", {}).get("per")
    three = results.get("3_libriphrase_onthefly", {}).get("per")
    print("\n--- reading ---")
    if one is None:
        print("INCOMPLETE: no LibriSpeech dev result; the probe path was not validated.")
    elif one > 3 * STEP_A_REFERENCE_PER:
        print(
            f"CONFOUND A: LibriSpeech dev PER {one} is far from {STEP_A_REFERENCE_PER}. "
            "The probe's model/decode path is wrong; its LibriPhrase verdict means nothing."
        )
    elif two is None or three is None:
        print("INCOMPLETE: one or both LibriPhrase feature paths produced no result.")
    elif two > three + 0.05:
        print(
            f"CONFOUND B: precomputed {two} vs on-the-fly {three}. The eval fbank files "
            "are not the feature space the encoder was trained on. Regenerate them."
        )
    elif three > two + 0.05:
        print(
            f"INCONCLUSIVE: precomputed {two} is unexpectedly better than on-the-fly {three}; "
            "the feature paths disagree, so the frozen-stack verdict is not validated."
        )
    else:
        print(
            f"NO CONFOUND: path validated at {one} on LibriSpeech dev, and both LibriPhrase "
            f"feature paths agree ({two} vs {three}). The shared LibriPhrase PER reflects "
            "frozen-stack behavior rather than a feature mismatch."
        )


if __name__ == "__main__":
    main()
