"""Export a deployable Stage II QbyT v4.1 ``.pt`` checkpoint to ONNX.

The default split export mirrors multi-query inference: run the acoustic encoder
once, then reuse its output for one or more QbyT query batches.  A monolithic
graph is also available for integrations that prefer a single call.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from dma_kws.config import fbank_kwargs, get_eval_fbank_config
from dma_kws.inference.score_calibration import PositiveAffineCalibrator
from dma_kws.inference.stage2_verifier import build_stage2_inference_model
from dma_kws.nn import min_input_frames_for_encoder
from dma_kws.stage2.readout import QbyTScoreSpec
from dma_kws.training.checkpoint_io import (
    QBYT_ALIGNMENT_SPEC_KEY,
    QBYT_READOUT_VERSION_KEY,
    assert_qbyt_readout_version,
    assert_stream_policy_matches,
)


ExportLayout = Literal["split", "full", "both"]
ExporterKind = Literal["dynamo", "legacy"]
MANIFEST_SCHEMA_VERSION = 1


class Stage2OnnxExportError(RuntimeError):
    """Raised when a Stage II artifact cannot be exported safely."""


@dataclass(frozen=True)
class LoadedStage2Model:
    source: Path
    payload: Mapping[str, Any]
    config: dict[str, Any]
    model: torch.nn.Module
    qbyt_score: QbyTScoreSpec
    stream_policy: Any
    vocab_size: int


@dataclass(frozen=True)
class DummyInputs:
    feats: torch.Tensor
    feat_lengths: torch.Tensor
    anchors: torch.Tensor
    anchor_lengths: torch.Tensor
    speech: torch.Tensor
    speech_lengths: torch.Tensor


@dataclass(frozen=True)
class OnnxGraphPlan:
    kind: str
    filename: str
    module: torch.nn.Module
    args: tuple[torch.Tensor, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    dynamic_axes: Mapping[str, Mapping[int, str]]
    dynamic_bounds: Mapping[str, tuple[int | None, int | None]]


@dataclass(frozen=True)
class Stage2OnnxExportResult:
    output_dir: Path
    artifacts: tuple[Path, ...]
    manifest: Path
    verification_max_abs_error: Mapping[str, float]


class Stage2EncoderOnnx(torch.nn.Module):
    """Expose the encoder/optional-adapter boundary used by QbyT."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        speech, speech_lengths = self.model.encode_for_qbyt(feats, feat_lengths)
        return speech, speech_lengths.to(dtype=torch.int64)


class QbyTOnnx(torch.nn.Module):
    """Expose QbyT independently so encoder output can serve many queries."""

    def __init__(self, qbyt: torch.nn.Module, *, include_sequence_logits: bool) -> None:
        super().__init__()
        self.qbyt = qbyt
        self.include_sequence_logits = bool(include_sequence_logits)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        anchors: torch.Tensor,
        anchor_lengths: torch.Tensor,
    ):
        raw_logit, sequence_logits = self.qbyt(
            speech,
            anchors,
            speech_lengths=speech_lengths,
            text_lengths=anchor_lengths,
        )
        if self.include_sequence_logits:
            return raw_logit, sequence_logits
        return raw_logit


class Stage2FullOnnx(torch.nn.Module):
    """Expose the complete fbank-to-QbyT score path as one graph."""

    def __init__(self, model: torch.nn.Module, *, include_sequence_logits: bool) -> None:
        super().__init__()
        self.model = model
        self.include_sequence_logits = bool(include_sequence_logits)

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        anchors: torch.Tensor,
        anchor_lengths: torch.Tensor,
    ):
        raw_logit, sequence_logits = self.model.encode_and_score(
            feats,
            feat_lengths,
            anchors,
            anchor_lengths,
        )
        if self.include_sequence_logits:
            return raw_logit, sequence_logits
        return raw_logit


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        return torch.load(path, map_location="cpu")


def load_stage2_pt(path: str | Path) -> LoadedStage2Model:
    """Strictly restore a full Stage II inference ``.pt`` artifact."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Stage II .pt not found: {source}")
    if source.suffix != ".pt":
        raise Stage2OnnxExportError(
            f"Input must be the converted full-model .pt artifact, got: {source}"
        )

    raw = _torch_load(source)
    if not isinstance(raw, Mapping):
        raise Stage2OnnxExportError(f"Checkpoint payload must be a mapping: {source}")
    if "model_state_dict" not in raw:
        if "lora_state_dict" in raw:
            raise Stage2OnnxExportError(
                "Adapter-only LoRA artifacts cannot be exported. Convert the Lightning "
                "checkpoint with --lora-output merged first."
            )
        raise Stage2OnnxExportError(
            "Checkpoint has no model_state_dict. Run scripts/convert_stage2_checkpoints.py "
            "before ONNX export."
        )
    state = raw["model_state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise Stage2OnnxExportError("model_state_dict must be a non-empty mapping")

    config_value = raw.get("config")
    if not isinstance(config_value, Mapping):
        raise Stage2OnnxExportError(
            "Stage II .pt must embed the resolved training config"
        )
    config = dict(config_value)
    stage1 = config.get("stage1")
    stage2 = config.get("stage2")
    if not isinstance(stage1, Mapping) or not isinstance(stage2, Mapping):
        raise Stage2OnnxExportError(
            "Embedded config must contain stage1 and stage2 mappings"
        )

    vocab_value = raw.get("vocab_size")
    if isinstance(vocab_value, bool):
        raise Stage2OnnxExportError("vocab_size must be an integer")
    try:
        vocab_size = int(vocab_value)
    except (TypeError, ValueError) as exc:
        raise Stage2OnnxExportError(
            "Stage II .pt must record vocab_size"
        ) from exc
    if vocab_size < 2:
        raise Stage2OnnxExportError(f"vocab_size must be at least 2, got {vocab_size}")

    try:
        model, qbyt_score, stream_policy = build_stage2_inference_model(
            stage1_cfg=stage1,
            stage2_cfg=stage2,
            vocab_size=vocab_size,
        )
        assert_stream_policy_matches(raw, stream_policy, source=source)
        assert_qbyt_readout_version(
            raw,
            source=source,
            expected_alignment=qbyt_score,
        )
        model.load_state_dict(dict(state), strict=True)
    except Stage2OnnxExportError:
        raise
    except Exception as exc:
        raise Stage2OnnxExportError(
            f"Stage II checkpoint is incompatible with its embedded config: {exc}"
        ) from exc

    model = model.cpu().float().eval()
    return LoadedStage2Model(
        source=source,
        payload=raw,
        config=config,
        model=model,
        qbyt_score=qbyt_score,
        stream_policy=stream_policy,
        vocab_size=vocab_size,
    )


def _validate_export_options(
    *,
    layout: ExportLayout,
    exporter: ExporterKind,
    batch_size: int,
    feature_frames: int,
    anchor_tokens: int,
    opset_version: int,
    dynamic_batch: bool,
    qbyt_score: QbyTScoreSpec,
) -> None:
    del dynamic_batch
    if layout not in {"split", "full", "both"}:
        raise Stage2OnnxExportError(f"Unsupported export layout: {layout}")
    if exporter not in {"dynamo", "legacy"}:
        raise Stage2OnnxExportError(f"Unsupported ONNX exporter: {exporter}")
    for name, value in (
        ("batch_size", batch_size),
        ("feature_frames", feature_frames),
        ("anchor_tokens", anchor_tokens),
    ):
        if isinstance(value, bool) or int(value) < 1:
            raise Stage2OnnxExportError(f"{name} must be a positive integer")
    if int(opset_version) < 17:
        raise Stage2OnnxExportError("Stage II export requires ONNX opset 17 or newer")
    # The v4.1 architecture is checkpointed as readout version 4; its extended
    # sink/position fields live in the qbyt_readout compatibility mapping.
    if qbyt_score.version != 4 or qbyt_score.family != "pooling":
        raise Stage2OnnxExportError(
            "Stage II ONNX export currently supports only QbyT pooling v4/v4.1 "
            f"checkpoints, got readout version {qbyt_score.version} "
            f"({qbyt_score.family})"
        )


def _load_icefall_onnx_converter():
    """Load Icefall's recipe-local ONNX module converter.

    The GigaSpeech KWS recipe commonly symlinks ``scaling.py`` from the
    LibriSpeech Zipformer recipe without symlinking ``scaling_converter.py``.
    Resolve both locations instead of relying on an ambient ``PYTHONPATH``.
    """

    from dma_kws.pathing import ensure_icefall_on_path

    try:
        recipe = ensure_icefall_on_path()
    except SystemExit as exc:
        raise Stage2OnnxExportError(str(exc)) from exc
    scaling_path = (recipe / "scaling.py").resolve()
    candidates = (
        recipe / "scaling_converter.py",
        scaling_path.with_name("scaling_converter.py"),
    )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        searched = ", ".join(str(path) for path in candidates)
        raise Stage2OnnxExportError(
            "Icefall scaling_converter.py is required for Zipformer ONNX export; "
            f"searched: {searched}"
        )

    module_name = "_dma_kws_icefall_scaling_converter"
    module = sys.modules.get(module_name)
    if (
        module is None
        or Path(getattr(module, "__file__", "")).resolve() != source.resolve()
    ):
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise Stage2OnnxExportError(
                f"Could not load Icefall ONNX converter from {source}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise Stage2OnnxExportError(
                f"Could not import Icefall ONNX converter from {source}: {exc}"
            ) from exc

    converter = getattr(module, "convert_scaled_to_non_scaled", None)
    if not callable(converter):
        raise Stage2OnnxExportError(
            f"Icefall converter {source} has no convert_scaled_to_non_scaled()"
        )
    return converter


def _prepare_model_for_onnx(loaded: LoadedStage2Model) -> str | None:
    """Apply backend-specific, semantics-preserving ONNX graph rewrites."""

    encoder_type = str(
        loaded.config["stage1"].get("encoder_type", "conformer")
    ).lower()
    if encoder_type != "icefall_zipformer":
        return None

    converter = _load_icefall_onnx_converter()
    preserve_chunk_convolution = bool(
        getattr(loaded.stream_policy, "enabled", False)
        and int(getattr(loaded.stream_policy, "chunk_size", -1)) > 0
    )
    chunk_modules: list[tuple[str, torch.nn.Module]] = []
    if preserve_chunk_convolution:
        chunk_type = converter.__globals__.get("ChunkCausalDepthwiseConv1d")
        if not isinstance(chunk_type, type):
            raise Stage2OnnxExportError(
                "Icefall ONNX converter does not expose "
                "ChunkCausalDepthwiseConv1d; cannot preserve the checkpoint's "
                "chunked evaluation semantics"
            )
        chunk_modules = [
            (name, module)
            for name, module in loaded.model.named_modules()
            if name and isinstance(module, chunk_type)
        ]
    try:
        converted = converter(loaded.model, inplace=True, is_onnx=True)
    except Exception as exc:
        raise Stage2OnnxExportError(
            "Icefall Zipformer ONNX preparation failed while converting scaled "
            f"modules: {exc}"
        ) from exc
    if converted is not loaded.model:
        raise Stage2OnnxExportError(
            "Icefall convert_scaled_to_non_scaled(inplace=True) returned a different model"
        )
    # Icefall's ONNX converter scripts SimpleDownsample (required for causal
    # inputs whose time length is not divisible by the stack downsampling
    # factor), but it also replaces every chunk convolution with a full-context
    # implementation.  Restore those modules for fixed chunked checkpoints so
    # export keeps the same eval operating point instead of silently changing
    # 16/64-style models to non-streaming inference.
    for name, module in chunk_modules:
        if "." in name:
            parent_name, child_name = name.rsplit(".", maxsplit=1)
            parent = loaded.model.get_submodule(parent_name)
        else:
            parent, child_name = loaded.model, name
        setattr(parent, child_name, module)
    loaded.model.eval()
    if preserve_chunk_convolution:
        return (
            "icefall.convert_scaled_to_non_scaled(is_onnx=True), "
            "preserve_chunk_convolution=true"
        )
    return "icefall.convert_scaled_to_non_scaled(is_onnx=True)"


def _dummy_inputs(
    loaded: LoadedStage2Model,
    *,
    batch_size: int,
    feature_frames: int,
    anchor_tokens: int,
) -> DummyInputs:
    input_dim = int(loaded.config["stage1"].get("input_dim", 80))
    total_values = batch_size * feature_frames * input_dim
    feats = torch.linspace(-1.0, 1.0, steps=total_values, dtype=torch.float32).reshape(
        batch_size, feature_frames, input_dim
    )

    # Vary lengths in multi-row probes so masking is exercised without making a
    # row so short that encoder subsampling removes every frame.
    feat_lengths = torch.full((batch_size,), feature_frames, dtype=torch.int64)
    if batch_size > 1:
        decrement = max(1, feature_frames // 20)
        for row in range(1, batch_size):
            feat_lengths[row] = max(feature_frames // 2, feature_frames - row * decrement)

    anchors = torch.zeros((batch_size, anchor_tokens), dtype=torch.int64)
    anchor_lengths = torch.empty((batch_size,), dtype=torch.int64)
    for row in range(batch_size):
        length = max(1, anchor_tokens - (row % max(1, anchor_tokens)))
        anchor_lengths[row] = length
        anchors[row, :length] = 1 + (
            torch.arange(length, dtype=torch.int64) % (loaded.vocab_size - 1)
        )

    try:
        with torch.inference_mode():
            speech, speech_lengths = loaded.model.encode_for_qbyt(
                feats, feat_lengths
            )
    except Exception as exc:
        raise Stage2OnnxExportError(
            "Example feature shape cannot pass the Stage II encoder; increase "
            f"--feature-frames (currently {feature_frames}): {exc}"
        ) from exc
    if speech.ndim != 3 or speech.size(1) < 1:
        raise Stage2OnnxExportError(
            f"Encoder produced an invalid QbyT input shape: {tuple(speech.shape)}"
        )
    return DummyInputs(
        feats=feats,
        feat_lengths=feat_lengths,
        anchors=anchors,
        anchor_lengths=anchor_lengths,
        speech=speech,
        speech_lengths=speech_lengths.to(dtype=torch.int64),
    )


def _output_names(include_sequence_logits: bool) -> tuple[str, ...]:
    return (
        ("raw_logit", "sequence_logits")
        if include_sequence_logits
        else ("raw_logit",)
    )


def _graph_plans(
    loaded: LoadedStage2Model,
    dummy: DummyInputs,
    *,
    layout: ExportLayout,
    include_sequence_logits: bool,
) -> list[OnnxGraphPlan]:
    score_outputs = _output_names(include_sequence_logits)
    min_feature_frames = min_input_frames_for_encoder(loaded.model.encoder, 1)
    max_anchor_tokens = (
        128 if getattr(loaded.model.qbyt, "text_pos_emb", None) is not None else None
    )
    shared_bounds = {
        "batch": (1, None),
        "feature_frames": (min_feature_frames, None),
        "encoder_frames": (1, None),
        "anchor_tokens": (1, max_anchor_tokens),
    }
    plans: list[OnnxGraphPlan] = []
    if layout in {"split", "both"}:
        plans.append(
            OnnxGraphPlan(
                kind="encoder",
                filename="stage2_encoder.onnx",
                module=Stage2EncoderOnnx(loaded.model).eval(),
                args=(dummy.feats, dummy.feat_lengths),
                input_names=("feats", "feat_lengths"),
                output_names=("speech", "speech_lengths"),
                dynamic_axes={
                    "feats": {0: "batch", 1: "feature_frames"},
                    "feat_lengths": {0: "batch"},
                    "speech": {0: "batch", 1: "encoder_frames"},
                    "speech_lengths": {0: "batch"},
                },
                dynamic_bounds=shared_bounds,
            )
        )
        qbyt_axes: dict[str, dict[int, str]] = {
            "speech": {0: "batch", 1: "encoder_frames"},
            "speech_lengths": {0: "batch"},
            "anchors": {0: "batch", 1: "anchor_tokens"},
            "anchor_lengths": {0: "batch"},
            "raw_logit": {0: "batch"},
        }
        if include_sequence_logits:
            qbyt_axes["sequence_logits"] = {0: "batch", 1: "anchor_tokens"}
        plans.append(
            OnnxGraphPlan(
                kind="qbyt",
                filename="qbyt.onnx",
                module=QbyTOnnx(
                    loaded.model.qbyt,
                    include_sequence_logits=include_sequence_logits,
                ).eval(),
                args=(
                    dummy.speech,
                    dummy.speech_lengths,
                    dummy.anchors,
                    dummy.anchor_lengths,
                ),
                input_names=(
                    "speech",
                    "speech_lengths",
                    "anchors",
                    "anchor_lengths",
                ),
                output_names=score_outputs,
                dynamic_axes=qbyt_axes,
                dynamic_bounds=shared_bounds,
            )
        )

    if layout in {"full", "both"}:
        full_axes: dict[str, dict[int, str]] = {
            "feats": {0: "batch", 1: "feature_frames"},
            "feat_lengths": {0: "batch"},
            "anchors": {0: "batch", 1: "anchor_tokens"},
            "anchor_lengths": {0: "batch"},
            "raw_logit": {0: "batch"},
        }
        if include_sequence_logits:
            full_axes["sequence_logits"] = {0: "batch", 1: "anchor_tokens"}
        plans.append(
            OnnxGraphPlan(
                kind="full",
                filename="stage2.onnx",
                module=Stage2FullOnnx(
                    loaded.model,
                    include_sequence_logits=include_sequence_logits,
                ).eval(),
                args=(
                    dummy.feats,
                    dummy.feat_lengths,
                    dummy.anchors,
                    dummy.anchor_lengths,
                ),
                input_names=(
                    "feats",
                    "feat_lengths",
                    "anchors",
                    "anchor_lengths",
                ),
                output_names=score_outputs,
                dynamic_axes=full_axes,
                dynamic_bounds=shared_bounds,
            )
        )
    return plans


def _require_onnx_dependencies(*, exporter: ExporterKind, verify: bool):
    missing: list[str] = []
    try:
        import onnx
    except ImportError:
        onnx = None
        missing.append("onnx")
    if exporter == "dynamo":
        try:
            import onnxscript  # noqa: F401
        except ImportError:
            missing.append("onnxscript")
    if verify:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            missing.append("onnxruntime")
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise Stage2OnnxExportError(
            f"Missing ONNX export dependencies: {names}. Install them with "
            "`uv pip install --python .venv/bin/python onnx onnxscript "
            "onnxruntime` before exporting."
        )
    return onnx


def _batch_dynamic_axes(plan: OnnxGraphPlan) -> dict[str, dict[int, str]]:
    selected: dict[str, dict[int, str]] = {}
    for name, mapping in plan.dynamic_axes.items():
        batch_axes = {axis: symbol for axis, symbol in mapping.items() if symbol == "batch"}
        if batch_axes:
            selected[name] = batch_axes
    return selected


def _dynamo_dynamic_shapes(plan: OnnxGraphPlan):
    dimensions: dict[str, Any] = {}

    def dim(name: str):
        if name not in dimensions:
            minimum, maximum = plan.dynamic_bounds.get(name, (1, None))
            dimensions[name] = torch.export.Dim(
                name,
                min=minimum,
                max=maximum,
            )
        return dimensions[name]

    result: list[dict[int, Any]] = []
    selected = _batch_dynamic_axes(plan)
    for input_name in plan.input_names:
        axes = selected.get(input_name, {})
        result.append({axis: dim(name) for axis, name in axes.items()})
    return tuple(result)


def _temporary_onnx_path(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.onnx"
    )


def _export_graph(
    plan: OnnxGraphPlan,
    destination: Path,
    *,
    opset_version: int,
    exporter: ExporterKind,
    dynamic_batch: bool,
) -> None:
    export_module = plan.module
    if exporter == "legacy" and any(
        isinstance(module, torch.jit.ScriptModule)
        for module in plan.module.modules()
    ):
        # Icefall's ONNX preparation scripts SimpleDownsample and compact
        # positional encodings.  Feeding a Python parent with scripted children
        # directly to torch.onnx.export fails with "not part of the active
        # trace"; Icefall's own exporter first traces the complete wrapper.
        try:
            # Scripted Icefall blocks feed ordinary Conv modules whose parameters
            # still require gradients. inference_mode creates tensors those Conv
            # modules cannot save, even while tracing an eval model; no_grad has
            # the desired export behavior without that restriction.
            with torch.no_grad():
                export_module = torch.jit.trace(plan.module, plan.args)
        except Exception as exc:
            raise Stage2OnnxExportError(
                f"Failed to pre-trace {plan.kind} graph after Icefall ONNX "
                f"preparation: {exc}"
            ) from exc

    kwargs: dict[str, Any] = {
        "input_names": list(plan.input_names),
        "output_names": list(plan.output_names),
        "opset_version": int(opset_version),
        "verbose": False,
    }
    if exporter == "dynamo":
        kwargs.update(
            {
                "dynamo": True,
                "external_data": False,
                "dynamic_shapes": (
                    _dynamo_dynamic_shapes(plan) if dynamic_batch else None
                ),
            }
        )
    else:
        kwargs.update(
            {
                "dynamo": False,
                "dynamic_axes": (
                    _batch_dynamic_axes(plan) if dynamic_batch else None
                ),
            }
        )
    try:
        with torch.inference_mode():
            torch.onnx.export(
                export_module,
                plan.args,
                str(destination),
                **kwargs,
            )
    except Exception as exc:
        raise Stage2OnnxExportError(
            f"Failed to export {plan.kind} graph with the {exporter} exporter: {exc}"
        ) from exc
    if not destination.is_file() or destination.stat().st_size == 0:
        raise Stage2OnnxExportError(
            f"ONNX exporter did not create a non-empty {plan.kind} graph"
        )


def _stamp_and_check_onnx(onnx_module, path: Path, metadata: Mapping[str, str]) -> None:
    model = onnx_module.load(str(path), load_external_data=False)
    existing = {entry.key: entry.value for entry in model.metadata_props}
    existing.update({str(key): str(value) for key, value in metadata.items()})
    del model.metadata_props[:]
    for key, value in sorted(existing.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx_module.checker.check_model(model)
    onnx_module.save_model(model, str(path), save_as_external_data=False)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _as_output_tuple(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _assert_close(
    name: str,
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> float:
    if expected.shape != actual.shape:
        raise Stage2OnnxExportError(
            f"ONNX verification shape mismatch for {name}: "
            f"PyTorch={expected.shape}, ONNX={actual.shape}"
        )
    if not np.isfinite(actual).all():
        raise Stage2OnnxExportError(
            f"ONNX verification produced non-finite values for {name}"
        )
    difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    max_error = float(difference.max(initial=0.0))
    if not np.allclose(expected, actual, atol=atol, rtol=rtol):
        raise Stage2OnnxExportError(
            f"ONNX verification failed for {name}: max_abs_error={max_error:.6g}, "
            f"atol={atol:.6g}, rtol={rtol:.6g}"
        )
    return max_error


def _ort_session(ort_module, path: Path):
    try:
        return ort_module.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise Stage2OnnxExportError(
            f"ONNX Runtime could not load {path.name}: {exc}"
        ) from exc


def _run_ort(session, names: Sequence[str], tensors: Sequence[Any]) -> list[np.ndarray]:
    feed = {
        name: (_to_numpy(value) if isinstance(value, torch.Tensor) else value)
        for name, value in zip(names, tensors)
    }
    return session.run(None, feed)


def _verify_graphs(
    plans: Sequence[OnnxGraphPlan],
    temporary_paths: Mapping[str, Path],
    dummy: DummyInputs,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # Guarded earlier; keeps direct callers explicit.
        raise Stage2OnnxExportError("onnxruntime is required for verification") from exc

    by_kind = {plan.kind: plan for plan in plans}
    errors: dict[str, float] = {}
    with torch.inference_mode():
        if "encoder" in by_kind:
            plan = by_kind["encoder"]
            expected = _as_output_tuple(plan.module(*plan.args))
            session = _ort_session(ort, temporary_paths["encoder"])
            actual = _run_ort(session, plan.input_names, plan.args)
            for name, expected_value, actual_value in zip(
                plan.output_names, expected, actual
            ):
                errors[f"encoder.{name}"] = _assert_close(
                    f"encoder.{name}",
                    _to_numpy(expected_value),
                    actual_value,
                    atol=atol,
                    rtol=rtol,
                )

        if "qbyt" in by_kind:
            plan = by_kind["qbyt"]
            expected = _as_output_tuple(plan.module(*plan.args))
            session = _ort_session(ort, temporary_paths["qbyt"])
            actual = _run_ort(session, plan.input_names, plan.args)
            for name, expected_value, actual_value in zip(
                plan.output_names, expected, actual
            ):
                errors[f"qbyt.{name}"] = _assert_close(
                    f"qbyt.{name}",
                    _to_numpy(expected_value),
                    actual_value,
                    atol=atol,
                    rtol=rtol,
                )

        if "full" in by_kind:
            plan = by_kind["full"]
            expected = _as_output_tuple(plan.module(*plan.args))
            session = _ort_session(ort, temporary_paths["full"])
            actual = _run_ort(session, plan.input_names, plan.args)
            for name, expected_value, actual_value in zip(
                plan.output_names, expected, actual
            ):
                errors[f"full.{name}"] = _assert_close(
                    f"full.{name}",
                    _to_numpy(expected_value),
                    actual_value,
                    atol=atol,
                    rtol=rtol,
                )

        # The split pair is a deployment pipeline, not merely two isolated graphs:
        # verify that encoder ORT outputs feed QbyT ORT and match the PyTorch score.
        if "encoder" in by_kind and "qbyt" in by_kind:
            encoder_plan = by_kind["encoder"]
            qbyt_plan = by_kind["qbyt"]
            encoder_session = _ort_session(ort, temporary_paths["encoder"])
            qbyt_session = _ort_session(ort, temporary_paths["qbyt"])
            encoded = _run_ort(
                encoder_session, encoder_plan.input_names, encoder_plan.args
            )
            split_args = (
                encoded[0],
                encoded[1],
                _to_numpy(dummy.anchors),
                _to_numpy(dummy.anchor_lengths),
            )
            split_actual = _run_ort(
                qbyt_session, qbyt_plan.input_names, split_args
            )
            with torch.inference_mode():
                full_expected = _as_output_tuple(
                    Stage2FullOnnx(
                        encoder_plan.module.model,
                        include_sequence_logits=len(qbyt_plan.output_names) == 2,
                    )(
                        dummy.feats,
                        dummy.feat_lengths,
                        dummy.anchors,
                        dummy.anchor_lengths,
                    )
                )
            for name, expected_value, actual_value in zip(
                qbyt_plan.output_names, full_expected, split_actual
            ):
                errors[f"split_pipeline.{name}"] = _assert_close(
                    f"split_pipeline.{name}",
                    _to_numpy(expected_value),
                    actual_value,
                    atol=atol,
                    rtol=rtol,
                )
    return errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_path = str(payload.get("tokenizer_dict_path", "") or "")
    result: dict[str, Any] = {"dict_path": raw_path or None}
    if raw_path:
        path = Path(raw_path)
        result["dict_sha256"] = _sha256(path) if path.is_file() else None
    else:
        result["dict_sha256"] = None
    return result


def _calibration_manifest(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        calibrator = PositiveAffineCalibrator(slope=1.0, bias=0.0)
        return {
            "source": None,
            "source_sha256": None,
            "slope": calibrator.slope,
            "bias": calibrator.bias,
            "formula": "sigmoid(slope * raw_logit + bias)",
        }
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Calibration JSON not found: {source}")
    calibrator = PositiveAffineCalibrator.load_json(source)
    return {
        "source": str(source),
        "source_sha256": _sha256(source),
        "slope": calibrator.slope,
        "bias": calibrator.bias,
        "formula": "sigmoid(slope * raw_logit + bias)",
    }


def _artifact_interface(plan: OnnxGraphPlan, dynamic_batch: bool) -> dict[str, Any]:
    selected = _batch_dynamic_axes(plan) if dynamic_batch else {}
    axes = {
        name: {str(axis): symbol for axis, symbol in mapping.items()}
        for name, mapping in selected.items()
        if name in plan.input_names or name in plan.output_names
    }
    return {
        "kind": plan.kind,
        "inputs": list(plan.input_names),
        "outputs": list(plan.output_names),
        "dynamic_axes": axes,
        "dynamic_bounds": (
            {
                name: {"min": bounds[0], "max": bounds[1]}
                for name, bounds in plan.dynamic_bounds.items()
                if name == "batch"
            }
            if dynamic_batch
            else {}
        ),
    }


def _build_manifest(
    loaded: LoadedStage2Model,
    plans: Sequence[OnnxGraphPlan],
    temporary_paths: Mapping[str, Path],
    *,
    layout: ExportLayout,
    exporter: ExporterKind,
    opset_version: int,
    dynamic_batch: bool,
    batch_size: int,
    feature_frames: int,
    anchor_tokens: int,
    include_sequence_logits: bool,
    model_preparation: str | None,
    verification: Mapping[str, float],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    stage1 = loaded.config["stage1"]
    stage2 = loaded.config["stage2"]
    tokenizer = _tokenizer_manifest(loaded.payload)
    tokenizer["vocab_size"] = loaded.vocab_size
    tokenizer["padding_and_blank_id"] = 0
    readout = loaded.qbyt_score.as_dict()
    adapter_cfg = stage2.get("phoneme_adapter", {}) or {}
    artifacts: dict[str, Any] = {}
    for plan in plans:
        path = temporary_paths[plan.kind]
        artifacts[plan.filename] = {
            "sha256": _sha256(path),
            "interface": _artifact_interface(plan, dynamic_batch),
        }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": {
            "checkpoint": str(loaded.source),
            "checkpoint_sha256": _sha256(loaded.source),
            "step": int(loaded.payload.get("step", 0)),
        },
        "export": {
            "layout": layout,
            "exporter": exporter,
            "opset_version": int(opset_version),
            "dtype": "float32",
            "dynamic_batch": bool(dynamic_batch),
            "feature_frames_dynamic": False,
            "anchor_tokens_dynamic": False,
            "example_shapes": {
                "batch_size": batch_size,
                "feature_frames": feature_frames,
                "anchor_tokens": anchor_tokens,
            },
            "include_sequence_logits": bool(include_sequence_logits),
            "model_preparation": model_preparation,
        },
        "model": {
            "encoder_type": str(stage1.get("encoder_type", "conformer")),
            "encoder_output_dim": int(stage2.get("encoder_output_dim", 144)),
            "qbyt_input_dim": int(loaded.model.qbyt.audio_projection.in_features),
            "qbyt_family": loaded.qbyt_score.family,
            "qbyt_score": readout,
            QBYT_READOUT_VERSION_KEY: int(loaded.qbyt_score.version),
            QBYT_ALIGNMENT_SPEC_KEY: loaded.payload.get(QBYT_ALIGNMENT_SPEC_KEY),
            "stream_policy": asdict(loaded.stream_policy),
            "phoneme_adapter_enabled": bool(adapter_cfg.get("enabled", False)),
        },
        "preprocessing": {
            "input": "fbank",
            "fbank": fbank_kwargs(get_eval_fbank_config(loaded.config)),
            "tokenizer": tokenizer,
        },
        "calibration": dict(calibration),
        "artifacts": artifacts,
        "verification": {
            "onnxruntime_cpu": bool(verification),
            "max_abs_error": dict(verification),
        },
    }


def export_stage2_onnx(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    layout: ExportLayout = "split",
    exporter: ExporterKind = "legacy",
    opset_version: int = 17,
    batch_size: int = 1,
    feature_frames: int = 300,
    anchor_tokens: int = 16,
    dynamic_batch: bool = False,
    include_sequence_logits: bool = False,
    calibration_path: str | Path | None = None,
    verify: bool = True,
    atol: float = 1.0e-4,
    rtol: float = 1.0e-4,
    overwrite: bool = False,
) -> Stage2OnnxExportResult:
    """Export and optionally verify Stage II ONNX artifacts.

    All graphs are written to temporary files, checked, and (when requested)
    executed with ONNX Runtime before they replace their final destinations.
    """

    loaded = load_stage2_pt(checkpoint)
    _validate_export_options(
        layout=layout,
        exporter=exporter,
        batch_size=batch_size,
        feature_frames=feature_frames,
        anchor_tokens=anchor_tokens,
        opset_version=opset_version,
        dynamic_batch=dynamic_batch,
        qbyt_score=loaded.qbyt_score,
    )
    text_pos_emb = getattr(loaded.model.qbyt, "text_pos_emb", None)
    if text_pos_emb is not None and int(anchor_tokens) > int(text_pos_emb.num_embeddings):
        raise Stage2OnnxExportError(
            "--anchor-tokens exceeds the learned v4.1 text-position table: "
            f"{anchor_tokens} > {text_pos_emb.num_embeddings}"
        )
    if not np.isfinite(atol) or atol < 0 or not np.isfinite(rtol) or rtol < 0:
        raise Stage2OnnxExportError("atol and rtol must be finite and non-negative")

    onnx_module = _require_onnx_dependencies(exporter=exporter, verify=verify)
    calibration = _calibration_manifest(calibration_path)
    model_preparation = _prepare_model_for_onnx(loaded)
    dummy = _dummy_inputs(
        loaded,
        batch_size=int(batch_size),
        feature_frames=int(feature_frames),
        anchor_tokens=int(anchor_tokens),
    )
    plans = _graph_plans(
        loaded,
        dummy,
        layout=layout,
        include_sequence_logits=include_sequence_logits,
    )

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        plan.kind: destination_dir / plan.filename for plan in plans
    }
    manifest_path = destination_dir / "stage2_onnx_manifest.json"
    existing = [path for path in (*destinations.values(), manifest_path) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing ONNX artifacts: {joined}; pass --overwrite"
        )

    temporary_paths = {
        kind: _temporary_onnx_path(destination)
        for kind, destination in destinations.items()
    }
    manifest_temp = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        common_metadata = {
            "dma_kws.schema_version": str(MANIFEST_SCHEMA_VERSION),
            "dma_kws.checkpoint_sha256": _sha256(loaded.source),
            "dma_kws.qbyt_readout_version": str(loaded.qbyt_score.version),
            "dma_kws.qbyt_family": loaded.qbyt_score.family,
            "dma_kws.qbyt_score": json.dumps(
                loaded.qbyt_score.as_dict(), sort_keys=True, separators=(",", ":")
            ),
            "dma_kws.stream_policy": loaded.stream_policy.describe(),
            "dma_kws.output_semantics": "raw_qbyt_logit",
        }
        for plan in plans:
            temporary = temporary_paths[plan.kind]
            _export_graph(
                plan,
                temporary,
                opset_version=int(opset_version),
                exporter=exporter,
                dynamic_batch=dynamic_batch,
            )
            _stamp_and_check_onnx(
                onnx_module,
                temporary,
                {**common_metadata, "dma_kws.artifact_kind": plan.kind},
            )

        verification = (
            _verify_graphs(
                plans,
                temporary_paths,
                dummy,
                atol=float(atol),
                rtol=float(rtol),
            )
            if verify
            else {}
        )
        manifest = _build_manifest(
            loaded,
            plans,
            temporary_paths,
            layout=layout,
            exporter=exporter,
            opset_version=int(opset_version),
            dynamic_batch=dynamic_batch,
            batch_size=int(batch_size),
            feature_frames=int(feature_frames),
            anchor_tokens=int(anchor_tokens),
            include_sequence_logits=include_sequence_logits,
            model_preparation=model_preparation,
            verification=verification,
            calibration=calibration,
        )
        with manifest_temp.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        for kind, destination in destinations.items():
            os.replace(temporary_paths[kind], destination)
        os.replace(manifest_temp, manifest_path)
    finally:
        for path in (*temporary_paths.values(), manifest_temp):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    artifact_paths = tuple(destinations[plan.kind] for plan in plans)
    return Stage2OnnxExportResult(
        output_dir=destination_dir,
        artifacts=artifact_paths,
        manifest=manifest_path,
        verification_max_abs_error=verification,
    )


__all__ = [
    "LoadedStage2Model",
    "QbyTOnnx",
    "Stage2EncoderOnnx",
    "Stage2FullOnnx",
    "Stage2OnnxExportError",
    "Stage2OnnxExportResult",
    "export_stage2_onnx",
    "load_stage2_pt",
]
