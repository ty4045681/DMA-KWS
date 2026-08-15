"""Composable waveform augmentation for clip evaluation."""

from __future__ import annotations

from typing import Any, Mapping, Protocol


class WaveformTransform(Protocol):
    """A pickle-friendly transform accepted by :class:`ClipFeatureDataset`."""

    def __call__(self, index: int, waveform: Any, sample_rate: int) -> Any: ...


def _component_enabled(component: object | None) -> bool:
    return component is not None and bool(getattr(component, "enabled", True))


def _component_summary(component: object | None) -> dict[str, Any]:
    if component is None:
        return {"enabled": False}
    summary = getattr(component, "summary", None)
    if not callable(summary):
        return {"enabled": _component_enabled(component)}
    value = summary()
    if not isinstance(value, Mapping):
        raise TypeError("waveform augmentation summary() must return a mapping")
    return dict(value)


def _component_metadata(component: object, index: int) -> dict[str, Any]:
    metadata = getattr(component, "recipe_metadata", None)
    if not callable(metadata):
        return {"row_index": index}
    value = metadata(index)
    if not isinstance(value, Mapping):
        raise TypeError(
            "waveform augmentation recipe_metadata() must return a mapping"
        )
    return dict(value)


def _shape(waveform: Any, *, stage: str) -> tuple[int, ...]:
    try:
        return tuple(waveform.shape)
    except (AttributeError, TypeError) as exc:
        raise TypeError(f"{stage} must return a waveform tensor with a shape") from exc


def _require_mono_waveform(
    waveform: Any,
    *,
    stage: str,
) -> Any:
    actual_shape = _shape(waveform, stage=stage)
    if len(actual_shape) != 2 or actual_shape[0] != 1 or actual_shape[1] <= 0:
        raise ValueError(
            f"{stage} must return a non-empty mono 2-D waveform with shape "
            f"(1, samples), got {actual_shape}"
        )
    return waveform


def _require_additive_shape(
    waveform: Any,
    clean_shape: tuple[int, ...],
    *,
    stage: str,
) -> Any:
    _require_mono_waveform(waveform, stage=stage)
    actual_shape = tuple(waveform.shape)
    if actual_shape != clean_shape:
        raise ValueError(
            f"{stage} must match the clean waveform shape: "
            f"clean={clean_shape}, additive={actual_shape}"
        )
    return waveform


def _clone_waveform(waveform: Any) -> Any:
    clone = getattr(waveform, "clone", None)
    if callable(clone):
        return clone()
    copy = getattr(waveform, "copy", None)
    if callable(copy):
        return copy()
    raise TypeError("waveform must provide clone() or copy() for additive mixing")


class WaveformAugmentationPipeline:
    """Compose AudioAug phases with MUSAN against one shared clean reference.

    The order is ``AudioAug pre-mix`` -> optional ``MUSAN pre-mix`` ->
    ``AudioAug and MUSAN additive deltas`` -> ``AudioAug post-mix``. The MUSAN
    pre-mix phase is used by clean-only effects such as volume variation, so
    every additive branch uses the same transformed clean reference. Pre/post
    phases may change the sample count (for example, faithful speed change),
    while every additive delta must match the final pre-mix clean shape. The
    object is a regular top-level callable so a ``DataLoader`` can pickle it for
    spawned workers.
    """

    def __init__(
        self,
        *,
        audio_aug: object | None = None,
        musan_mixer: WaveformTransform | None = None,
    ) -> None:
        self._audio_aug = audio_aug
        self._musan_mixer = musan_mixer

    @property
    def enabled(self) -> bool:
        return _component_enabled(self._audio_aug) or _component_enabled(
            self._musan_mixer
        )

    @property
    def changes_duration(self) -> bool:
        """Whether the configured source transform can change sample count."""

        return bool(getattr(self._audio_aug, "changes_duration", False))

    def summary(self) -> dict[str, Any]:
        """Return fields that can be merged into the evaluation summary."""

        return {
            "audio_aug": _component_summary(self._audio_aug),
            "musan_mix": _component_summary(self._musan_mixer),
        }

    def recipe_metadata(self, index: int) -> dict[str, Any]:
        """Return enabled component fields that can be merged into one result."""

        result: dict[str, Any] = {}
        if _component_enabled(self._audio_aug):
            result["audio_aug"] = _component_metadata(self._audio_aug, index)
        if _component_enabled(self._musan_mixer):
            result["musan_mix"] = _component_metadata(self._musan_mixer, index)
        return result

    def __call__(self, index: int, waveform: Any, sample_rate: int) -> Any:
        if not self.enabled:
            return waveform

        clean = _require_mono_waveform(
            waveform,
            stage="waveform augmentation input",
        )
        audio_enabled = _component_enabled(self._audio_aug)
        musan_enabled = _component_enabled(self._musan_mixer)

        # Preserve the historical MUSAN-only numerical path exactly. Rewriting
        # it as clean + (mixed - clean) can introduce an avoidable float32 ULP
        # difference and extra copies when no parallel AudioAug delta exists.
        if musan_enabled and not audio_enabled:
            return _require_mono_waveform(
                self._musan_mixer(index, clean, sample_rate),  # type: ignore[misc]
                stage="musan_mixer",
            )

        if audio_enabled:
            pre_mix = getattr(self._audio_aug, "apply_pre_mix", None)
            if not callable(pre_mix):
                raise TypeError("enabled audio_aug must define apply_pre_mix()")
            clean = _require_mono_waveform(
                pre_mix(index, clean, sample_rate),
                stage="audio_aug.apply_pre_mix",
            )

        musan_pre_mix = None
        musan_additive = None
        if musan_enabled:
            musan_pre_mix = getattr(self._musan_mixer, "apply_pre_mix", None)
            musan_additive = getattr(
                self._musan_mixer,
                "apply_additive_delta",
                None,
            )
            has_musan_pre_mix = callable(musan_pre_mix)
            has_musan_additive = callable(musan_additive)
            if has_musan_pre_mix != has_musan_additive:
                raise TypeError(
                    "phased musan_mixer must define both apply_pre_mix() and "
                    "apply_additive_delta()"
                )
            if has_musan_pre_mix:
                clean = _require_mono_waveform(
                    musan_pre_mix(index, clean, sample_rate),
                    stage="musan_mixer.apply_pre_mix",
                )

        clean_shape = tuple(clean.shape)

        additive_deltas = []
        if audio_enabled:
            additive = getattr(self._audio_aug, "apply_additive_delta", None)
            if not callable(additive):
                raise TypeError(
                    "enabled audio_aug must define apply_additive_delta()"
                )
            delta = additive(index, _clone_waveform(clean), sample_rate)
            additive_deltas.append(
                _require_additive_shape(
                    delta,
                    clean_shape,
                    stage="audio_aug.apply_additive_delta",
                )
            )

        if musan_enabled:
            if callable(musan_additive):
                delta = musan_additive(
                    index,
                    _clone_waveform(clean),
                    sample_rate,
                )
                additive_deltas.append(
                    _require_additive_shape(
                        delta,
                        clean_shape,
                        stage="musan_mixer.apply_additive_delta",
                    )
                )
            else:
                musan_clean = _clone_waveform(clean)
                musan_mixed = _require_additive_shape(
                    self._musan_mixer(  # type: ignore[misc]
                        index,
                        musan_clean,
                        sample_rate,
                    ),
                    clean_shape,
                    stage="musan_mixer",
                )
                additive_deltas.append(musan_mixed - musan_clean)

        mixed = clean
        for delta in additive_deltas:
            mixed = mixed + delta

        if audio_enabled:
            post_mix = getattr(self._audio_aug, "apply_post_mix", None)
            if not callable(post_mix):
                raise TypeError("enabled audio_aug must define apply_post_mix()")
            mixed = _require_mono_waveform(
                post_mix(index, mixed, sample_rate),
                stage="audio_aug.apply_post_mix",
            )
        return mixed


__all__ = ["WaveformAugmentationPipeline", "WaveformTransform"]
