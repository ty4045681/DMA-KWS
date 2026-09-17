from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dma_kws.configs.schema import StreamPolicy
from dma_kws.inference import stage2_onnx
from dma_kws.inference.stage2_onnx import (
    QbyTOnnx,
    Stage2EncoderOnnx,
    Stage2FullOnnx,
    Stage2OnnxExportError,
    export_stage2_onnx,
    load_stage2_pt,
)
from dma_kws.stage2.readout import resolve_qbyt_score_spec


class _FakeQbyT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_projection = torch.nn.Linear(2, 2)

    def forward(
        self,
        speech,
        anchors,
        speech_lengths=None,
        text_lengths=None,
    ):
        projected = self.audio_projection(speech)
        positions = torch.arange(speech.size(1)).unsqueeze(0)
        speech_mask = positions < speech_lengths.unsqueeze(1)
        raw = (projected * speech_mask.unsqueeze(2)).sum(dim=(1, 2)) / (
            speech_lengths * projected.size(2)
        )
        sequence = anchors.to(dtype=speech.dtype)
        if text_lengths is not None:
            text_positions = torch.arange(anchors.size(1)).unsqueeze(0)
            sequence = sequence * (text_positions < text_lengths.unsqueeze(1))
        return raw, sequence


class _FakeEncoder(torch.nn.Identity):
    def output_frames(self, num_input_frames: int) -> int:
        return num_input_frames


class _FakeStage2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_scale = torch.nn.Parameter(torch.tensor(1.25))
        self.qbyt = _FakeQbyT()
        self.encoder = _FakeEncoder()
        self.adapter = None

    def encode_for_qbyt(self, feats, feat_lengths):
        return feats[..., :2] * self.encoder_scale, feat_lengths.to(torch.int64)

    def encode_and_score(self, feats, feat_lengths, anchors, anchor_lengths):
        speech, speech_lengths = self.encode_for_qbyt(feats, feat_lengths)
        return self.qbyt(
            speech,
            anchors,
            speech_lengths=speech_lengths,
            text_lengths=anchor_lengths,
        )

    def forward(self, feats, feat_lengths, anchors, anchor_lengths):
        return self.encode_and_score(
            feats, feat_lengths, anchors, anchor_lengths
        )[0]


def _score_spec():
    return resolve_qbyt_score_spec(
        {
            "qbyt_readout_version": 4,
            "qbyt_readout": {
                "mode": "eps_softmin",
                "temperature": 1.0,
            },
        }
    )


def _config() -> dict:
    return {
        "stage1": {"encoder_type": "conformer", "input_dim": 2},
        "stage2": {
            "encoder_output_dim": 2,
            "qbyt_readout_version": 4,
            "qbyt_readout": {
                "mode": "eps_softmin",
                "temperature": 1.0,
            },
            "phoneme_adapter": {"enabled": False},
        },
    }


def _write_fake_pt(path: Path) -> _FakeStage2:
    model = _FakeStage2()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": _config(),
            "step": 12,
            "vocab_size": 8,
            "qbyt_readout_version": 4,
        },
        path,
    )
    return model


def _patch_model_builder(monkeypatch):
    monkeypatch.setattr(
        stage2_onnx,
        "build_stage2_inference_model",
        lambda **_kwargs: (
            _FakeStage2(),
            _score_spec(),
            StreamPolicy(backend="conformer", enabled=False),
        ),
    )
    monkeypatch.setattr(stage2_onnx, "assert_stream_policy_matches", lambda *a, **k: None)
    monkeypatch.setattr(stage2_onnx, "assert_qbyt_readout_version", lambda *a, **k: None)


def test_onnx_wrappers_preserve_the_deployment_boundaries():
    model = _FakeStage2().eval()
    feats = torch.randn(2, 7, 3)
    feat_lengths = torch.tensor([7, 5])
    anchors = torch.tensor([[1, 2, 0], [3, 4, 5]])
    anchor_lengths = torch.tensor([2, 3])

    encoder = Stage2EncoderOnnx(model)
    speech, speech_lengths = encoder(feats, feat_lengths)
    assert speech.shape == (2, 7, 2)
    assert speech_lengths.dtype == torch.int64

    qbyt = QbyTOnnx(model.qbyt, include_sequence_logits=True)
    raw, sequence = qbyt(speech, speech_lengths, anchors, anchor_lengths)
    full = Stage2FullOnnx(model, include_sequence_logits=True)
    full_raw, full_sequence = full(
        feats, feat_lengths, anchors, anchor_lengths
    )
    torch.testing.assert_close(raw, full_raw)
    torch.testing.assert_close(sequence, full_sequence)


def test_load_stage2_pt_rebuilds_and_strict_loads(monkeypatch, tmp_path):
    source = tmp_path / "stage2.pt"
    expected = _write_fake_pt(source)
    _patch_model_builder(monkeypatch)

    loaded = load_stage2_pt(source)

    assert loaded.vocab_size == 8
    assert loaded.qbyt_score.version == 4
    assert loaded.model.training is False
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], tensor)


def test_load_stage2_pt_rejects_adapter_only_payload(tmp_path):
    source = tmp_path / "adapter.pt"
    torch.save(
        {
            "lora_state_dict": {"qbyt.x.lora_A": torch.ones(1)},
            "config": _config(),
            "vocab_size": 8,
        },
        source,
    )

    with pytest.raises(Stage2OnnxExportError, match="Adapter-only"):
        load_stage2_pt(source)


def test_opset_below_17_is_rejected():
    with pytest.raises(Stage2OnnxExportError, match="opset 17"):
        stage2_onnx._validate_export_options(
            layout="split",
            exporter="legacy",
            batch_size=1,
            feature_frames=300,
            anchor_tokens=16,
            opset_version=16,
            dynamic_batch=False,
            qbyt_score=_score_spec(),
        )


def test_non_v4_readout_is_rejected():
    score = resolve_qbyt_score_spec({"qbyt_readout_version": 7})
    with pytest.raises(Stage2OnnxExportError, match="only QbyT pooling v4/v4.1"):
        stage2_onnx._validate_export_options(
            layout="split",
            exporter="legacy",
            batch_size=1,
            feature_frames=300,
            anchor_tokens=16,
            opset_version=17,
            dynamic_batch=False,
            qbyt_score=score,
        )


def test_export_orchestration_is_atomic_and_writes_manifest(monkeypatch, tmp_path):
    source = tmp_path / "stage2.pt"
    _write_fake_pt(source)
    _patch_model_builder(monkeypatch)
    loaded = load_stage2_pt(source)
    monkeypatch.setattr(stage2_onnx, "load_stage2_pt", lambda _path: loaded)
    monkeypatch.setattr(
        stage2_onnx,
        "_require_onnx_dependencies",
        lambda **_kwargs: object(),
    )

    def fake_export(plan, destination, **_kwargs):
        destination.write_bytes(f"fake-{plan.kind}".encode())

    monkeypatch.setattr(stage2_onnx, "_export_graph", fake_export)
    monkeypatch.setattr(stage2_onnx, "_stamp_and_check_onnx", lambda *a, **k: None)
    monkeypatch.setattr(
        stage2_onnx,
        "_verify_graphs",
        lambda *a, **k: {"split_pipeline.raw_logit": 2.5e-6},
    )

    output_dir = tmp_path / "onnx"
    result = export_stage2_onnx(
        source,
        output_dir,
        layout="split",
        exporter="dynamo",
        feature_frames=12,
        anchor_tokens=4,
    )

    assert [path.name for path in result.artifacts] == [
        "stage2_encoder.onnx",
        "qbyt.onnx",
    ]
    assert all(path.is_file() for path in result.artifacts)
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["model"]["qbyt_readout_version"] == 4
    assert manifest["model"]["qbyt_family"] == "pooling"
    assert manifest["verification"]["onnxruntime_cpu"] is True
    assert set(manifest["artifacts"]) == {"stage2_encoder.onnx", "qbyt.onnx"}
    assert not list(output_dir.glob(".*.tmp*"))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_stage2_onnx(
            source,
            output_dir,
            layout="split",
            feature_frames=12,
            anchor_tokens=4,
        )


def test_missing_export_dependencies_have_actionable_error(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name in {"onnx", "onnxscript", "onnxruntime"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(Stage2OnnxExportError, match="uv pip install"):
        stage2_onnx._require_onnx_dependencies(exporter="dynamo", verify=True)


def test_real_legacy_export_and_onnx_checker(tmp_path):
    onnx = pytest.importorskip("onnx")
    model = _FakeStage2().eval()
    loaded = stage2_onnx.LoadedStage2Model(
        source=tmp_path / "unused.pt",
        payload={},
        config=_config(),
        model=model,
        qbyt_score=_score_spec(),
        stream_policy=StreamPolicy(backend="conformer", enabled=False),
        vocab_size=8,
    )
    dummy = stage2_onnx._dummy_inputs(
        loaded,
        batch_size=2,
        feature_frames=12,
        anchor_tokens=4,
    )
    plans = stage2_onnx._graph_plans(
        loaded,
        dummy,
        layout="both",
        include_sequence_logits=True,
    )

    for plan in plans:
        destination = tmp_path / plan.filename
        stage2_onnx._export_graph(
            plan,
            destination,
            opset_version=17,
            exporter="legacy",
            dynamic_batch=False,
        )
        stage2_onnx._stamp_and_check_onnx(
            onnx,
            destination,
            {"dma_kws.artifact_kind": plan.kind},
        )
        exported = onnx.load(destination, load_external_data=False)
        assert [value.name for value in exported.graph.input] == list(plan.input_names)
        assert [value.name for value in exported.graph.output] == list(plan.output_names)
        assert {
            entry.key: entry.value for entry in exported.metadata_props
        }["dma_kws.artifact_kind"] == plan.kind


@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_real_v41_qbyt_graph_exports_with_lengths_and_relative_bias(
    tmp_path,
    dynamic_batch,
):
    onnx = pytest.importorskip("onnx")
    from onnx.reference import ReferenceEvaluator
    from qbyt.pooling import QbyT

    qbyt = QbyT(
        encoder_output_size=6,
        num_embeds=8,
        embed_dim=8,
        post_num_layers=1,
        readout_mode="eps_softmin",
        readout_temperature=1.0,
        sink_token=True,
        text_position="learned",
        audio_position="relative_bias",
    ).eval()
    wrapper = QbyTOnnx(qbyt, include_sequence_logits=True).eval()
    speech = torch.randn(2, 12, 6)
    speech_lengths = torch.tensor([12, 9], dtype=torch.int64)
    anchors = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=torch.int64)
    anchor_lengths = torch.tensor([3, 4], dtype=torch.int64)
    plan = stage2_onnx.OnnxGraphPlan(
        kind="qbyt",
        filename="qbyt.onnx",
        module=wrapper,
        args=(speech, speech_lengths, anchors, anchor_lengths),
        input_names=("speech", "speech_lengths", "anchors", "anchor_lengths"),
        output_names=("raw_logit", "sequence_logits"),
        dynamic_axes={
            "speech": {0: "batch", 1: "encoder_frames"},
            "speech_lengths": {0: "batch"},
            "anchors": {0: "batch", 1: "anchor_tokens"},
            "anchor_lengths": {0: "batch"},
            "raw_logit": {0: "batch"},
            "sequence_logits": {0: "batch", 1: "anchor_tokens"},
        },
        dynamic_bounds={
            "batch": (1, None),
            "encoder_frames": (1, None),
            "anchor_tokens": (1, 128),
        },
    )
    destination = tmp_path / plan.filename

    stage2_onnx._export_graph(
        plan,
        destination,
        opset_version=17,
        exporter="legacy",
        dynamic_batch=dynamic_batch,
    )
    stage2_onnx._stamp_and_check_onnx(
        onnx,
        destination,
        {"dma_kws.qbyt_readout_version": "4"},
    )

    exported = onnx.load(destination, load_external_data=False)
    assert [value.name for value in exported.graph.input] == list(plan.input_names)
    assert [value.name for value in exported.graph.output] == list(plan.output_names)
    reference = ReferenceEvaluator(exported)
    expected = wrapper(speech, speech_lengths, anchors, anchor_lengths)
    actual = reference.run(
        None,
        {
            "speech": speech.detach().numpy(),
            "speech_lengths": speech_lengths.numpy(),
            "anchors": anchors.numpy(),
            "anchor_lengths": anchor_lengths.numpy(),
        },
    )
    torch.testing.assert_close(
        torch.from_numpy(actual[0]), expected[0], atol=1.0e-4, rtol=1.0e-4
    )
    torch.testing.assert_close(
        torch.from_numpy(actual[1]), expected[1], atol=1.0e-4, rtol=1.0e-4
    )
    if dynamic_batch:
        assert exported.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
        assert exported.graph.input[0].type.tensor_type.shape.dim[1].dim_value == 12
        alt_speech = torch.randn(1, 12, 6)
        alt_speech_lengths = torch.tensor([7], dtype=torch.int64)
        alt_anchors = torch.tensor([[1, 3, 5, 0]], dtype=torch.int64)
        alt_anchor_lengths = torch.tensor([3], dtype=torch.int64)
        alt_expected = wrapper(
            alt_speech,
            alt_speech_lengths,
            alt_anchors,
            alt_anchor_lengths,
        )
        alt_actual = reference.run(
            None,
            {
                "speech": alt_speech.numpy(),
                "speech_lengths": alt_speech_lengths.numpy(),
                "anchors": alt_anchors.numpy(),
                "anchor_lengths": alt_anchor_lengths.numpy(),
            },
        )
        torch.testing.assert_close(
            torch.from_numpy(alt_actual[0]),
            alt_expected[0],
            atol=1.0e-4,
            rtol=1.0e-4,
        )
        torch.testing.assert_close(
            torch.from_numpy(alt_actual[1]),
            alt_expected[1],
            atol=1.0e-4,
            rtol=1.0e-4,
        )
