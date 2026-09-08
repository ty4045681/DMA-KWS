"""Exercise joint adaptation methods through training and export on CPU."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from dma_kws.config import compose_config, config_to_dict
from dma_kws.stage2.adapt import Stage2AdaptArgs, run_stage2_adaptation
from dma_kws.stage2.adapt_paths import adapt_exp_root
from dma_kws.stage2.collate import test_collate_fn
from dma_kws.stage2.module import Stage2LightningModule
from dma_kws.tokenizer import load_char_tokenizer, tokenize_phoneme_string
from dma_kws.training.checkpoint_io import (
    QBYT_READOUT_VERSION_KEY,
    STAGE2_BASE_FINGERPRINT_KEY,
    assert_qbyt_readout_version,
    fingerprint_stage2_base,
    stamp_qbyt_readout_version,
)


class _TinyEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, feat, lengths, **kwargs):
        mask = torch.arange(feat.shape[1], device=feat.device)[None] < lengths[:, None]
        return self.projection(feat), mask[:, None]

    def apply_stream_config(self, chunk_sizes, left_context_frames):
        pass  # The fixture has no time-dependent receptive field.

    def set_batch_count(self, batch_count):
        self.batch_count = float(batch_count)
        return 1

    def output_frames(self, num_frames):
        return num_frames


class _BackgroundFeatures:
    """Replace audio decoding only; real LibriPhrase sampling remains active."""

    def __init__(self, **kwargs):
        self.path = kwargs["audio_list_path"]

    def extract(self, *, rng):
        return torch.full((8, 8), rng.random() - 0.5, dtype=torch.float32)


def _write_corpus(root: Path) -> tuple[Path, Path, Path]:
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    generator = np.random.default_rng(11)
    phones = {1: "HH EY1", 0: "N OW1"}
    for phase in ("real", "tts"):
        for split in ("train", "eval"):
            rows = []
            for label in (0, 1):
                relative = f"raw/{phase}/{split}/{label}.wav"
                rows.append({
                    "audio_path": relative, "text": "hey" if label else "no",
                    "label": label, "keyword_phonemes": phones[1],
                    "text_variant_phonemes": phones[label],
                    "speaker_id": f"{phase}_{split}", "voice_id": "voice_a",
                })
                feature = root / "fbank" / phase / split / f"{label}.npy"
                feature.parent.mkdir(parents=True, exist_ok=True)
                np.save(feature, generator.normal(size=(8, 8)).astype(np.float32))
            with (manifests / f"{phase}_{split}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    replay_root = root / "replay"
    replay_root.mkdir()
    entries = []
    for index, label in enumerate((0, 1)):
        feature = replay_root / f"clip{index}.npy"
        np.save(feature, generator.normal(size=(8, 8)).astype(np.float32))
        clips = replay_root / f"clips-1-{index}.npy"
        distances = replay_root / f"distances-0-{index}.npy"
        np.save(clips, np.array([{"audio_path": f"clip{index}.wav"}], dtype=object))
        np.save(distances, np.array([], dtype=object))
        entries.append({
            "ngram": "hey" if label else "no", "ngram_g2p": phones[label],
            "clips_file": str(clips), "distances_file": str(distances),
        })
    parquet = replay_root / "phrases.parquet"
    pd.DataFrame(entries).to_parquet(parquet)

    # The audio decoder is replaced above, so the split validator needs only
    # distinct existing file identities, not a downloaded MUSAN corpus.
    for split in ("train", "eval"):
        audio = root / f"background_{split}.wav"
        audio.touch()
        (root / f"background_{split}.list").write_text(str(audio) + "\n", encoding="utf-8")
    return parquet, root / "background_train.list", root / "background_eval.list"


@pytest.mark.parametrize("background_validation", [False, True])
@pytest.mark.parametrize("method", ["lora", "qbyt_full", "encoder_qbyt_full"])
def test_joint_entry_point_trains_and_exports(
    tmp_path, monkeypatch, background_validation, method,
):
    import pytorch_lightning as pl
    from dma_kws.stage2 import adapt as adapt_module
    from dma_kws.stage2 import adapt_dataset, features, joint_validation
    from dma_kws.training import callbacks

    data_root = tmp_path / "data"
    parquet, background_train, background_eval = _write_corpus(data_root)
    experiments = (
        "+experiment=adapt_joint" if method == "lora" else
        f"+experiment=[icefall_zipformer_stage2_eps_softmin_v41,adapt_joint,adapt_{method}]"
    )
    config = config_to_dict(compose_config(overrides=[experiments]))
    config["paths"].update({
        "processed_root": str(data_root), "feature_root": str(data_root),
        "exp_root": str(tmp_path / "exp"),
    })
    config["stage1"].update({"input_dim": 8, "causal": method != "lora"})
    config["fbank"]["num_mel_bins"] = 8
    config["stage2"].update({
        "encoder_output_dim": 8, "qbyt_embed_dim": 8,
        "qbyt_layers": 0 if method == "lora" else 1,
        "parquet_file": str(parquet), "wav_dir": str(parquet.parent),
        "precision": "32-true", "accumulate_grad_batches": 1,
        "log_interval": 1,
    })
    config["stage2"]["qbyt_alignment"].update({
        "min_phone_duration_frames": 1, "max_phone_duration_frames": 3,
        "max_inter_phone_gap_frames": 1, "max_keyword_span_frames": 8,
        "local_context_kernel": 3,
    })
    config["stage2"]["phoneme_adapter"].update({
        "enabled": True, "freeze": True, "init_checkpoint": "", "ctc_weight": 0.0,
        "trunk": {"type": "linear", "output_dim": 8, "dropout": 0.0},
    })
    config["stage2"]["background_negative"]["audio_list_path"] = str(background_train)
    config["stage2"]["dataloader"]["pin_memory"] = False
    config["stage2"]["checkpoint"].update({"every_n_train_steps": 1, "save_top_k": 1})
    config["adapt"].update({
        "keyword": "hey", "data_root": str(data_root), "max_steps": 2,
        "sample_lens": 40, "batch_size_per_gpu": 20, "val_batch_size": 2,
        "num_workers": 0, "val_num_workers": 0, "rank": 2, "alpha": 4,
        "warmup_steps": 0, "learning_rate": 0.01,
    })
    if method == "encoder_qbyt_full":
        config["adapt"]["encoder_learning_rate"] = 0.001
    config["adapt"]["logging"]["backends"] = ["csv"]
    config["adapt"]["console"]["rich"] = False
    config["adapt"]["validation"].update({"val_check_interval": 1, "limit_val_batches": 1})
    config["adapt"]["joint"].update({
        "background_eval_list": str(background_eval) if background_validation else "",
        "background_val_samples": 2,
    })

    monkeypatch.setattr(
        "dma_kws.stage2.module.build_encoder",
        lambda stage1, *, output_dim: _TinyEncoder(stage1["input_dim"], output_dim),
    )
    monkeypatch.setattr(
        "dma_kws.nn.build_encoder",
        lambda stage1, *, output_dim: _TinyEncoder(stage1["input_dim"], output_dim),
    )
    monkeypatch.setattr(adapt_dataset, "make_g2p", lambda: None)
    monkeypatch.setattr(features, "TrainingBackgroundSampler", _BackgroundFeatures)
    monkeypatch.setattr(joint_validation, "TrainingBackgroundSampler", _BackgroundFeatures)
    monkeypatch.setattr(callbacks, "build_console_callbacks", lambda *args, **kwargs: [])
    transformers = ModuleType("transformers")
    transformers.get_cosine_schedule_with_warmup = (
        lambda optimizer, **kwargs: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    tokenizer = load_char_tokenizer(config["tokenizer"]["dict_path"])
    anchor = torch.tensor(tokenize_phoneme_string(tokenizer, "N OW1"), dtype=torch.long)
    replay_eval = [
        {"sample_id": index, "anchor_seq": anchor, "label": torch.tensor(1 - index),
         "feat": torch.from_numpy(np.load(parquet.parent / f"clip{index}.npy"))}
        for index in (0, 1)
    ]
    monkeypatch.setattr(
        adapt_module, "_build_val_dataloader",
        lambda *_: DataLoader(replay_eval, batch_size=2, collate_fn=test_collate_fn),
    )

    torch.manual_seed(17)
    base = Stage2LightningModule(config, vocab_size=len(tokenizer._symbol_table), freeze_encoder=True)
    base_state = {name: value.detach().clone() for name, value in base.state_dict().items()}
    base_path = tmp_path / "base.pt"
    torch.save(stamp_qbyt_readout_version({
        "model_state_dict": base_state, "config": base._checkpoint_config,
    }, alignment=base.qbyt_score), base_path)

    observed = {}
    atomic_destinations = []
    atomic_save = adapt_module._atomic_torch_save

    def observe_atomic_save(payload, destination):
        atomic_destinations.append(Path(destination))
        return atomic_save(payload, destination)

    monkeypatch.setattr(adapt_module, "_atomic_torch_save", observe_atomic_save)

    class Observe(pl.Callback):
        def on_train_start(self, trainer, model):
            observed["initial"] = {name: value.detach().clone() for name, value in model.state_dict().items()}
            observed["trainable"] = [name for name, value in model.named_parameters() if value.requires_grad]
            observed["batches"] = []
            observed["val_loaders"] = trainer.val_dataloaders

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            observed["batches"].append(batch["domain_source"].clone())

        def on_train_end(self, trainer, model):
            observed["trained"] = {name: value.detach().clone() for name, value in model.state_dict().items()}
            observed["step"] = trainer.global_step

    trainer_type = pl.Trainer

    def make_trainer(**kwargs):
        kwargs.update(enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
        kwargs["callbacks"].append(Observe())
        observed["trainer"] = trainer_type(**kwargs)
        return observed["trainer"]

    monkeypatch.setattr(pl, "Trainer", make_trainer)
    artifacts = run_stage2_adaptation(config, Stage2AdaptArgs(init_checkpoint=str(base_path), device="cpu"))

    # Check the artifact contract through the real runner for each method.
    # Stable aliases must be published atomically, while callers receive the
    # immutable per-run files that cannot be replaced by a subsequent run.
    exp_root = adapt_exp_root(config, "hey")
    stable_paths = {exp_root / "stage2_adapted.pt"}
    stable_paths.add(exp_root / "joint" / (
        "stage2_adapted.pt" if method != "lora" else "adapter_hey.pt"
    ))
    assert stable_paths <= set(atomic_destinations)
    assert set(artifacts.values()).isdisjoint(stable_paths)
    for artifact_path in artifacts.values():
        assert artifact_path in atomic_destinations
        assert artifact_path.is_file()
        assert "checkpoints" in artifact_path.relative_to(exp_root).parts

    assert observed["step"] == 2
    assert len(observed["batches"]) == 2
    for sources in observed["batches"]:
        assert torch.bincount(sources, minlength=4).tolist() == [8, 6, 4, 2]
    assert len(observed["val_loaders"]) == (4 if background_validation else 3)
    metrics = observed["trainer"].callback_metrics
    assert {"val/real_auc", "val/lph_auc", "val/tts_auc"} <= set(metrics)
    if background_validation:
        assert "val/musan_deploy_fpr" in metrics
        assert float(metrics["val/musan_num_neg"]) == 2

    assert observed["trainable"]
    changed = {
        name for name, tensor in observed["trained"].items()
        if not torch.equal(tensor, observed["initial"][name])
    }
    assert changed
    assert changed <= set(observed["trainable"])
    if method != "lora":
        expected_trainable = {f"qbyt.{name}" for name, _ in base.qbyt.named_parameters()}
        if method == "encoder_qbyt_full":
            expected_trainable |= {f"encoder.{name}" for name, _ in base.encoder.named_parameters()}
            assert "encoder.projection.weight" in changed
        assert expected_trainable == set(observed["trainable"])
        assert {"qbyt.audio_projection.weight", "qbyt.final_pos_fc.weight"} <= changed
        assert not any("parametrizations" in name or "lora_" in name for name in observed["trained"])
        assert "adapter" not in artifacts
        exported = torch.load(artifacts["merged"], map_location="cpu", weights_only=False)
        assert exported["phase"] == "joint"
        assert exported["method"] == method
        assert exported["dma_kws_run_context"]["run_id"].startswith(f"adapt_hey_joint_{method}/")
        assert exported[QBYT_READOUT_VERSION_KEY] == 4
        assert_qbyt_readout_version(exported, source="full QbyT CPU smoke", expected_alignment=base.qbyt_score)
        assert not any("parametrizations" in name or "lora_" in name for name in exported["model_state_dict"])
        frozen_prefixes = ("adapter.",) if method == "encoder_qbyt_full" else ("encoder.", "adapter.")
        for name, tensor in exported["model_state_dict"].items():
            if name.startswith(frozen_prefixes):
                torch.testing.assert_close(tensor, base_state[name], rtol=0, atol=0)
        if method == "encoder_qbyt_full":
            assert not torch.equal(exported["model_state_dict"]["encoder.projection.weight"], base_state["encoder.projection.weight"])

        # Compare the exported inference model with the selected Lightning
        # checkpoint, not the last training step: best-AUC selection may differ.
        best_path = observed["trainer"].checkpoint_callback.best_model_path
        selected = torch.load(best_path, map_location="cpu", weights_only=False)
        assert selected["checkpoint_kind"] == f"stage2_{method}"
        assert selected["method"] == method
        probe = next(iter(observed["val_loaders"][0]))
        base.load_state_dict(selected["state_dict"], strict=True)
        base.eval()
        with torch.no_grad():
            expected = base(probe["feat"], probe["feat_lengths"], probe["anchor"])[0]
        base.load_state_dict(exported["model_state_dict"], strict=True)
        with torch.no_grad():
            actual = base(probe["feat"], probe["feat_lengths"], probe["anchor"])[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        if method == "encoder_qbyt_full":
            from dma_kws.inference.stage2_verifier import Stage2Verifier
            monkeypatch.setattr(
                "dma_kws.inference.stage2_verifier.build_encoder",
                lambda stage1, *, output_dim: _TinyEncoder(stage1["input_dim"], output_dim),
            )
            # This comparison starts from precomputed features; waveform
            # extraction is outside its scope and requires optional Lhotse.
            monkeypatch.setattr(
                "dma_kws.inference.stage2_verifier.FbankExtractor", lambda **kwargs: None,
            )
            verifier = Stage2Verifier.from_config(
                config, {"stage2_ckpt": str(artifacts["merged"])}, torch.device("cpu"),
            )
            clips = [feat[:int(length)] for feat, length in zip(probe["feat"], probe["feat_lengths"])]
            keywords = [anchor[anchor != 0].tolist() for anchor in probe["anchor"]]
            deployed = verifier.score_clip_feats_with_logits(clips, keywords)
            torch.testing.assert_close(torch.tensor([raw for raw, _ in deployed]), expected, rtol=0, atol=0)

        from dma_kws.training.checkpoint_convert import convert_checkpoint
        converted_path = tmp_path / "converted-full.pt"
        convert_checkpoint(best_path, converted_path)
        converted = torch.load(converted_path, map_location="cpu", weights_only=False)
        assert_qbyt_readout_version(converted, source=converted_path, expected_alignment=base.qbyt_score)
        base.load_state_dict(converted["model_state_dict"], strict=True)
        with torch.no_grad():
            converted_logits = base(probe["feat"], probe["feat_lengths"], probe["anchor"])[0]
        torch.testing.assert_close(converted_logits, expected, rtol=0, atol=0)
        exp_root = adapt_exp_root(config, "hey")
        assert (exp_root / "joint" / "stage2_adapted.pt").is_file()
        assert (exp_root / "stage2_adapted.pt").is_file()
        assert not (exp_root / "joint" / "adapter_hey.pt").exists()
        return

    assert all(name.endswith((".lora_A", ".lora_B")) for name in observed["trainable"])
    assert fingerprint_stage2_base(observed["trained"]) == fingerprint_stage2_base(base_state)

    adapter = torch.load(artifacts["adapter"], map_location="cpu", weights_only=False)
    merged = torch.load(artifacts["merged"], map_location="cpu", weights_only=False)
    for payload in (adapter, merged):
        assert payload["phase"] == "joint"
        assert payload["dma_kws_run_context"]["run_id"].startswith("adapt_hey_joint/")
        assert payload[QBYT_READOUT_VERSION_KEY] == 7
        assert_qbyt_readout_version(payload, source="CPU joint smoke", expected_alignment=base.qbyt_score)
    assert adapter["step"] == merged["step"]
    assert adapter["step"] in (1, 2)
    assert adapter[STAGE2_BASE_FINGERPRINT_KEY] == fingerprint_stage2_base(base_state)
    assert all(name.endswith((".lora_A", ".lora_B")) for name in adapter["lora_state_dict"])
    assert any(torch.count_nonzero(value) for name, value in adapter["lora_state_dict"].items() if name.endswith("lora_B"))
    assert not any("parametrizations" in name for name in merged["model_state_dict"])
    base.load_state_dict(merged["model_state_dict"], strict=True)
    for name, value in merged["model_state_dict"].items():
        if name not in {"qbyt.audio_key.weight", "qbyt.text_query.weight"}:
            torch.testing.assert_close(value, base_state[name], rtol=0, atol=0)
    for target in ("audio_key", "text_query"):
        prefix = f"{target}.parametrizations.weight.0"
        state = adapter["lora_state_dict"]
        delta = (adapter["alpha"] / adapter["rank"]) * (state[f"{prefix}.lora_B"] @ state[f"{prefix}.lora_A"])
        name = f"qbyt.{target}.weight"
        torch.testing.assert_close(merged["model_state_dict"][name], base_state[name] + delta)
    exp_root = adapt_exp_root(config, "hey")
    assert (exp_root / "joint" / "adapter_hey.pt").is_file()
    assert (exp_root / "stage2_adapted.pt").is_file()
