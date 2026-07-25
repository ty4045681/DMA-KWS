"""Hyperparameter resolution for Stage II LoRA adaptation.

Kept free of torch imports so config resolution and console summaries can use it
without loading the training stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_ADAPT_LR = 4e-4

#: ``adapt.lr`` is an alias kept for CLI convenience; ``adapt.learning_rate`` is
#: canonical and carries the default. The alias defaults to ``None`` so that an
#: explicitly set value always wins over the canonical default.
ADAPT_LR_KEYS = ("lr", "learning_rate")

#: Keys emitted by the Optuna sweep that are not valid ``adapt`` config entries.
_SWEEP_ONLY_KEYS = ("alpha_ratio", "score")


def resolve_adapt_lr(adapt: dict[str, Any]) -> tuple[float, str]:
    """Return the effective adaptation learning rate and the key it came from."""
    for key in ADAPT_LR_KEYS:
        value = adapt.get(key)
        if value is not None and value != "":
            return float(value), key
    return DEFAULT_ADAPT_LR, "default"


def normalize_adapt_params(params: dict[str, Any], *, rank: int | None = None) -> dict[str, Any]:
    """Map sweep trial / ``best_params.yaml`` entries onto ``adapt`` config keys.

    Optuna searches ``alpha_ratio`` rather than ``alpha``, so the absolute alpha
    is derived here; bookkeeping keys such as ``score`` or ``_trial_number`` must
    not leak into the config. ``rank`` supplies the current config value when the
    params only carry ``alpha_ratio``.
    """
    normalized = {
        key: value
        for key, value in params.items()
        if not key.startswith("_") and key not in _SWEEP_ONLY_KEYS
    }

    if "alpha" not in normalized:
        alpha_ratio = params.get("alpha_ratio")
        effective_rank = normalized.get("rank", rank)
        if alpha_ratio is not None and effective_rank is not None:
            normalized["alpha"] = int(round(float(alpha_ratio) * int(effective_rank)))

    return normalized


def merge_adapt_params(adapt: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Merge sweep/params-file entries into an ``adapt`` section, in place.

    Returns the normalized overrides that were applied. A learning rate coming
    from ``params`` clears the ``lr`` alias so it cannot shadow the override.
    """
    normalized = normalize_adapt_params(params, rank=adapt.get("rank"))
    if "learning_rate" in normalized and "lr" not in normalized:
        adapt.pop("lr", None)
    adapt.update(normalized)
    return normalized


def load_adapt_params_file(params_file: str | Path) -> dict[str, Any]:
    """Load a YAML file of ``adapt`` overrides (e.g. sweep ``best_params.yaml``)."""
    import yaml

    path = Path(params_file)
    if not path.exists():
        raise FileNotFoundError(f"adapt params file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"adapt params file must be a mapping: {path}")
    return data
