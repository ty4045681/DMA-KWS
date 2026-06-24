import ast
import sys
from pathlib import Path

import pytest

from dma_kws.g2p import clean_phoneme_tokens, text_to_phonemes
from scripts import prepare_stage1_librispeech as prepare_stage1
from scripts import run_two_stage_demo as demo


def test_qbyt_forward_accepts_lengths_and_masks_padded_frames():
    source = Path("qbyt/model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    qbyt_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QbyT")
    forward = next(node for node in qbyt_class.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    arg_names = [arg.arg for arg in forward.args.args]

    assert "speech_lengths" in arg_names
    assert any(
        isinstance(node, ast.Call)
        and any(keyword.arg == "src_key_padding_mask" for keyword in node.keywords)
        for node in ast.walk(forward)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "gather"
        for node in ast.walk(forward)
    )


def test_demo_load_model_state_requires_exact_checkpoint_keys():
    class FakeModel:
        def __init__(self):
            self.loaded_state = None
            self.strict = None

        def load_state_dict(self, state, *, strict):
            self.loaded_state = state
            self.strict = strict

    model = FakeModel()

    returned = demo.load_model_state(
        model,
        "checkpoint.pt",
        load_fn=lambda path, map_location: {"model_state_dict": {"weight": 1}},
    )

    assert returned is model
    assert model.loaded_state == {"weight": 1}
    assert model.strict is True


def test_stage1_hf_parquet_requires_non_empty_dev_source(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  librispeech_root: {tmp_path / 'missing-librispeech'}",
                f"  processed_root: {tmp_path / 'processed'}",
                "stage1:",
                "  train_splits: [train-clean-100]",
                "  dev_splits: [dev-clean]",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(prepare_stage1, "make_g2p", lambda: object())
    monkeypatch.setattr(prepare_stage1, "prepare_parquet_split", lambda **kwargs: ["HH"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_stage1_librispeech.py",
            "--config",
            str(config_path),
            "--input-format",
            "hf-parquet",
            "--parquet-root",
            str(tmp_path / "train.parquet"),
        ],
    )

    with pytest.raises(SystemExit, match="--dev-parquet-root"):
        prepare_stage1.main()


def test_stage1_hf_parquet_can_prepare_dev_from_parquet(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  librispeech_root: {tmp_path / 'missing-librispeech'}",
                f"  processed_root: {tmp_path / 'processed'}",
                "stage1:",
                "  train_splits: [train-clean-100]",
                "  dev_splits: [dev-clean]",
            ]
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_prepare_parquet_split(**kwargs):
        calls.append(kwargs)
        return ["HH"]

    monkeypatch.setattr(prepare_stage1, "make_g2p", lambda: object())
    monkeypatch.setattr(prepare_stage1, "prepare_parquet_split", fake_prepare_parquet_split)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_stage1_librispeech.py",
            "--config",
            str(config_path),
            "--input-format",
            "hf-parquet",
            "--parquet-root",
            str(tmp_path / "train.parquet"),
            "--dev-parquet-root",
            str(tmp_path / "dev.parquet"),
            "--dev-parquet-split",
            "dev-clean",
        ],
    )

    prepare_stage1.main()

    assert [call["output_path"].name for call in calls] == ["train.jsonl", "dev.jsonl"]
    assert calls[1]["split"] == "dev-clean"


def test_demo_stage2_candidate_requires_enough_fbank_frames():
    sample_rate = 16_000
    min_frames = 7

    assert demo.num_fbank_frames(399, sample_rate=sample_rate) == 0
    assert demo.min_samples_for_fbank_frames(min_frames, sample_rate=sample_rate) == 1360
    assert not demo.has_min_fbank_frames(1359, min_frames=min_frames, sample_rate=sample_rate)
    assert demo.has_min_fbank_frames(1360, min_frames=min_frames, sample_rate=sample_rate)


def test_stage2_text_to_phonemes_filters_tokens_that_become_empty():
    class FakeG2P:
        def __call__(self, text):
            assert text == "hello"
            return ["HH0", " ", "", "1", "OW1"]

    assert text_to_phonemes(FakeG2P(), "HELLO") == ["HH", "OW"]
    assert clean_phoneme_tokens(["AH0", "", "2", "  ", "L"]) == ["AH", "L"]
