"""Convert Step A phoneme-adapter Lightning checkpoints to the ``.pt`` format.

The Step A trainer exports ``adapter_best_step*.pt`` / ``adapter_final_step*.pt``
itself when ``trainer.fit`` returns. This module exists for the checkpoints that
export never covers: interrupted runs, or a non-best ``.ckpt`` that should become
a loadable adapter artifact after the fact.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import torch

from dma_kws.config import require_sections
from dma_kws.pathing import resolve_dict_path
from dma_kws.training.checkpoint_io import export_model_pt
from dma_kws.training.run_context import RUN_CONTEXT_KEY, checkpoint_run_context

ADAPTER_PREFIX = "adapter."
BLANK_ID = 0

# FreshValidationModelCheckpoint saves as ``adapter_{step:07d}_{val_per:.4f}``.
_STEP_FROM_FILENAME = re.compile(r"^adapter_(\d+)_")


class PhonemeAdapterConversionError(Exception):
    """A Step A ``.ckpt`` cannot be converted to the adapter ``.pt`` format."""


def _adapter_state(
    state: Mapping[str, Any],
    *,
    source: Path,
) -> dict[str, torch.Tensor]:
    """Strip the Lightning module's ``adapter.`` prefix; the export stores the
    bare adapter state dict, which is what ``load_adapter_weights`` strict-loads."""
    adapter_state = {
        key[len(ADAPTER_PREFIX) :]: value
        for key, value in state.items()
        if key.startswith(ADAPTER_PREFIX)
    }
    if not adapter_state:
        raise PhonemeAdapterConversionError(
            f"{source} carries no 'adapter.' weights; this script only converts "
            "Step A checkpoints written by scripts/train_ctc_adapter.py."
        )
    return adapter_state


def _resolve_step(
    checkpoint: Mapping[str, Any],
    source: Path,
    step_override: int | None,
) -> int:
    if step_override is not None:
        return int(step_override)
    global_step = checkpoint.get("global_step")
    if isinstance(global_step, int):
        return global_step
    match = _STEP_FROM_FILENAME.match(source.name)
    if match:
        return int(match.group(1))
    raise PhonemeAdapterConversionError(
        f"Cannot determine the training step of {source}: the checkpoint has no "
        "'global_step' entry and the filename does not match 'adapter_<step>_*'. "
        "Pass --step explicitly."
    )


def _resolve_vocab_and_blank(
    checkpoint: Mapping[str, Any],
    config: dict[str, Any],
    source: Path,
) -> tuple[int, int]:
    """Recover the training-time ``vocab_size``/``blank_id``.

    ``PhonemeAdapterCtcModule.save_hyperparameters`` records both, so current
    checkpoints are self-describing. Older ones are recomputed from the config's
    tokenizer dict, the same way the training runner derives them.
    """
    hparams = checkpoint.get("hyper_parameters")
    if isinstance(hparams, Mapping):
        vocab_size = hparams.get("vocab_size")
        blank_id = hparams.get("blank_id")
        if vocab_size is not None and blank_id is not None:
            return int(vocab_size), int(blank_id)

    dict_path = resolve_dict_path(config)
    if not dict_path.is_file():
        raise PhonemeAdapterConversionError(
            f"{source} has no saved hyperparameters and the tokenizer dict from the "
            f"config is unavailable at {dict_path}, so vocab_size/blank_id cannot "
            "be reconstructed."
        )
    from dma_kws.config import get_tokenizer_config
    from dma_kws.tokenizer import load_char_tokenizer

    tokenizer_cfg = get_tokenizer_config(config)
    tokenizer = load_char_tokenizer(
        dict_path, split_with_space=tokenizer_cfg.get("split_with_space", " ")
    )
    return len(tokenizer._symbol_table), int(tokenizer.symbol_table.get("<blank>", BLANK_ID))


def convert_phoneme_adapter_checkpoint(
    source: Path | str,
    output: Path | str,
    *,
    config: dict[str, Any],
    step: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert one Step A Lightning ``.ckpt`` to the adapter ``.pt`` export.

    The payload matches what ``run_phoneme_adapter_training`` writes after
    ``trainer.fit``, so both ``load_adapter_weights`` (Step A warm start) and
    Stage II's ``_load_adapter_checkpoint`` accept it. ``config`` must be the
    resolved training config: Step A ``.ckpt`` files do not embed it, and the
    export must carry it so ``assert_stream_policy_matches`` can verify the
    streaming operating point at load time.
    """
    source = Path(source)
    output = Path(output)
    if output.suffix != ".pt":
        raise PhonemeAdapterConversionError(f"Output must end in .pt: {output}")
    if output.exists() and not overwrite:
        raise PhonemeAdapterConversionError(
            f"Output already exists: {output} (use --overwrite to replace it)"
        )
    try:
        require_sections(config, ["stage1", "phoneme_adapter", "tokenizer"])
    except ValueError as exc:
        raise PhonemeAdapterConversionError(str(exc)) from exc

    raw = torch.load(source, map_location="cpu")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("state_dict"), Mapping):
        raise PhonemeAdapterConversionError(
            f"{source} is not a Lightning checkpoint: expected a mapping with a "
            "'state_dict' entry."
        )

    adapter_state = _adapter_state(raw["state_dict"], source=source)
    vocab_size, blank_id = _resolve_vocab_and_blank(raw, config, source)
    resolved_step = _resolve_step(raw, source, step)

    # Rebuild the adapter and strict-load rather than trusting the raw tensors:
    # a config from the wrong experiment fails here instead of producing a .pt
    # that only blows up inside Stage II.
    from dma_kws.phoneme_adapter.module import build_phoneme_adapter

    stage1 = config["stage1"]
    adapter = build_phoneme_adapter(
        config["phoneme_adapter"],
        input_dim=int(stage1.get("encoder_output_dim", 144)),
        vocab_size=vocab_size,
        causal=bool(stage1.get("causal", False)),
        blank_id=blank_id,
    )
    try:
        adapter.load_state_dict(adapter_state, strict=True)
    except RuntimeError as exc:
        raise PhonemeAdapterConversionError(
            f"{source} does not match the supplied config (stage1.encoder_output_dim, "
            f"stage1.causal, phoneme_adapter trunk, vocab_size={vocab_size}): {exc}"
        ) from exc

    extra: dict[str, Any] = {"checkpoint_kind": "phoneme_adapter"}
    run_context = checkpoint_run_context(raw)
    if run_context is not None:
        extra[RUN_CONTEXT_KEY] = run_context
    return export_model_pt(
        adapter,
        output,
        config=config,
        dict_path=resolve_dict_path(config),
        vocab_size=vocab_size,
        step=resolved_step,
        blank_id=blank_id,
        extra=extra,
    )
