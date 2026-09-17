from __future__ import annotations

import json
from dataclasses import replace
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


class _ScriptedDouble(torch.nn.Module):
    def forward(self, value):
        return value * 2.0


class _PythonParentWithScriptedChild(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.child = torch.jit.script(_ScriptedDouble())
        self.projection = torch.nn.Linear(3, 3)

    def forward(self, value):
        return self.projection(self.child(value))


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


def _loaded_fake_model(tmp_path, *, encoder_type="conformer"):
    config = _config()
    config["stage1"]["encoder_type"] = encoder_type
    return stage2_onnx.LoadedStage2Model(
        source=tmp_path / "unused.pt",
        payload={},
        config=config,
        model=_FakeStage2().eval(),
        qbyt_score=_score_spec(),
        stream_policy=StreamPolicy(backend=encoder_type, enabled=False),
        vocab_size=8,
    )


def test_prepare_model_for_onnx_skips_non_icefall(monkeypatch, tmp_path):
    monkeypatch.setattr(
        stage2_onnx,
        "_load_icefall_onnx_converter",
        lambda: pytest.fail("converter must not load for conformer"),
    )

    assert stage2_onnx._prepare_model_for_onnx(_loaded_fake_model(tmp_path)) is None


def test_prepare_model_for_onnx_uses_icefall_onnx_conversion(monkeypatch, tmp_path):
    loaded = _loaded_fake_model(tmp_path, encoder_type="icefall_zipformer")
    calls = []

    def fake_converter(model, **kwargs):
        calls.append((model, kwargs))
        return model

    monkeypatch.setattr(
        stage2_onnx,
        "_load_icefall_onnx_converter",
        lambda: fake_converter,
    )

    preparation = stage2_onnx._prepare_model_for_onnx(loaded)

    assert preparation == "icefall.convert_scaled_to_non_scaled(is_onnx=True)"
    assert calls == [(loaded.model, {"inplace": True, "is_onnx": True})]
    assert loaded.model.training is False


def test_prepare_model_for_onnx_preserves_streaming_chunk_convolution(
    monkeypatch,
    tmp_path,
):
    class FakeChunkCausalDepthwiseConv1d(torch.nn.Module):
        def forward(self, value, chunk_size=-1):
            del chunk_size
            return value

    class FakeConvolutionModule(torch.nn.Module):
        def __init__(self, chunk):
            super().__init__()
            self.depthwise_conv = chunk

    class FakeLayer(torch.nn.Module):
        def __init__(self, chunk):
            super().__init__()
            self.conv_module1 = FakeConvolutionModule(chunk)

    class FakeStack(torch.nn.Module):
        def __init__(self, chunk):
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeLayer(chunk)])

    class FakeZipformer(torch.nn.Module):
        def __init__(self, chunk):
            super().__init__()
            self.encoders = torch.nn.ModuleList([FakeStack(chunk)])
            self.downsampling_factor = (1,)
            self.encoder_dim = (2,)
            self.downsample_output = torch.nn.Identity()

    class FakeIcefallEncoder(torch.nn.Module):
        def __init__(self, chunk):
            super().__init__()
            self.encoder = FakeZipformer(chunk)

    loaded = _loaded_fake_model(tmp_path, encoder_type="icefall_zipformer")
    chunk_module = FakeChunkCausalDepthwiseConv1d()
    loaded.model.encoder = FakeIcefallEncoder(chunk_module)
    loaded = replace(
        loaded,
        stream_policy=StreamPolicy(
            backend="icefall_zipformer",
            enabled=True,
            chunk_size=16,
            left_context_frames=64,
        ),
    )

    def fake_converter(model, **_kwargs):
        model.encoder.encoder.encoders[0].layers[
            0
        ].conv_module1.depthwise_conv = torch.nn.Identity()
        return model

    monkeypatch.setitem(
        fake_converter.__globals__,
        "ChunkCausalDepthwiseConv1d",
        FakeChunkCausalDepthwiseConv1d,
    )
    monkeypatch.setattr(
        stage2_onnx,
        "_load_icefall_onnx_converter",
        lambda: fake_converter,
    )

    preparation = stage2_onnx._prepare_model_for_onnx(loaded)

    zipformer = loaded.model.encoder.encoder
    assert isinstance(zipformer, stage2_onnx._FixedChunkZipformer2)
    adapted = zipformer.module.encoders[0].layers[0].conv_module1.depthwise_conv
    assert isinstance(adapted, stage2_onnx._FixedChunkDepthwiseConv1d)
    assert adapted.module is chunk_module
    assert adapted.chunk_size == 16
    assert preparation.endswith("fixed_chunk_policy=16/64")


def test_prepare_model_for_onnx_rejects_non_inplace_converter_result(
    monkeypatch,
    tmp_path,
):
    loaded = _loaded_fake_model(tmp_path, encoder_type="icefall_zipformer")
    monkeypatch.setattr(
        stage2_onnx,
        "_load_icefall_onnx_converter",
        lambda: (lambda *_args, **_kwargs: _FakeStage2()),
    )

    with pytest.raises(Stage2OnnxExportError, match="returned a different model"):
        stage2_onnx._prepare_model_for_onnx(loaded)


def test_fixed_chunk_zipformer_trace_preserves_attention_policy_and_batch():
    class FakeStack(torch.nn.Module):
        def forward(
            self,
            value,
            *,
            chunk_size,
            feature_mask,
            src_key_padding_mask,
            attn_mask,
        ):
            del feature_mask, src_key_padding_mask
            allowed = (~attn_mask).sum(dim=-1).to(value.dtype).view(-1, 1, 1)
            return value + allowed + float(chunk_size)

    class FakeZipformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = torch.nn.ModuleList([FakeStack()])
            self.encoder_dim = (2,)
            self.downsampling_factor = (1,)
            self.downsample_output = torch.nn.Identity()

    wrapper = stage2_onnx._FixedChunkZipformer2(
        FakeZipformer(),
        chunk_size=4,
        left_context_frames=4,
    ).eval()
    value = torch.zeros(8, 1, 2)
    lengths = torch.tensor([8], dtype=torch.int64)
    padding_mask = torch.zeros(1, 8, dtype=torch.bool)

    eager, eager_lengths = wrapper(value, lengths, padding_mask)
    traced = torch.jit.trace(wrapper, (value, lengths, padding_mask))
    value_batch2 = torch.zeros(8, 2, 2)
    lengths_batch2 = torch.tensor([8, 6], dtype=torch.int64)
    padding_batch2 = torch.arange(8).unsqueeze(0) >= lengths_batch2.unsqueeze(1)
    actual, actual_lengths = traced(value_batch2, lengths_batch2, padding_batch2)

    assert eager[:, 0, 0].tolist() == [8.0] * 4 + [12.0] * 4
    torch.testing.assert_close(actual, wrapper(value_batch2, lengths_batch2, padding_batch2)[0])
    torch.testing.assert_close(eager_lengths, torch.tensor([4]))
    torch.testing.assert_close(actual_lengths, torch.tensor([4, 3]))

    widened = wrapper._convert_num_channels(
        value_batch2,
        current_dim=2,
        target_dim=4,
    )
    assert widened.shape == (8, 2, 4)
    torch.testing.assert_close(widened[..., :2], value_batch2)
    assert torch.count_nonzero(widened[..., 2:]) == 0


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
    assert manifest["export"]["model_preparation"] is None
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


def test_real_legacy_export_pretraces_scripted_submodules(tmp_path):
    onnx = pytest.importorskip("onnx")
    model = _PythonParentWithScriptedChild().eval()
    value = torch.randn(2, 3)
    plan = stage2_onnx.OnnxGraphPlan(
        kind="scripted_probe",
        filename="scripted_probe.onnx",
        module=model,
        args=(value,),
        input_names=("value",),
        output_names=("output",),
        dynamic_axes={"value": {0: "batch"}, "output": {0: "batch"}},
        dynamic_bounds={"batch": (1, None)},
    )
    destination = tmp_path / plan.filename

    stage2_onnx._export_graph(
        plan,
        destination,
        opset_version=17,
        exporter="legacy",
        dynamic_batch=True,
    )
    stage2_onnx._stamp_and_check_onnx(onnx, destination, {})

    exported = onnx.load(destination, load_external_data=False)
    assert exported.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"


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
