"""Keep post-fit shared artifacts single-writer under externally launched DDP."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _function(path: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _assert_post_fit_artifacts_are_rank_zero(
    path: str,
    function_name: str,
    *,
    protected_calls: set[str],
    barrier_name: str,
) -> None:
    function = _function(path, function_name)
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    fit_call = next(call for call in calls if _call_name(call) == "fit")
    rank_zero_if = next(
        node
        for node in function.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "is_global_zero"
        and node.lineno > fit_call.lineno
    )

    guarded_calls = {
        _call_name(call)
        for call in ast.walk(rank_zero_if)
        if isinstance(call, ast.Call)
    }
    assert protected_calls <= guarded_calls

    barrier_calls = [
        call
        for call in calls
        if _call_name(call) == "barrier"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == barrier_name
    ]
    assert len(barrier_calls) == 1
    assert barrier_calls[0].lineno > rank_zero_if.end_lineno

    # A future refactor must not accidentally move a protected write/load or a
    # completion print back outside the single-writer block.
    guarded_nodes = set(ast.walk(rank_zero_if))
    unguarded = [
        call
        for call in calls
        if call.lineno > fit_call.lineno
        and _call_name(call) in protected_calls | {"print"}
        and call not in guarded_nodes
    ]
    assert not unguarded


@pytest.mark.parametrize(
    ("path", "function_name", "protected_calls", "barrier_name"),
    [
        (
            "dma_kws/stage2/train.py",
            "run_stage2_training",
            {"append_wide_row", "save"},
            "stage2_training_artifacts_saved",
        ),
        (
            "dma_kws/phoneme_adapter/runner.py",
            "run_phoneme_adapter_training",
            {"restore_best_checkpoint_weights", "export_model_pt"},
            "phoneme_adapter_artifacts_saved",
        ),
        (
            "dma_kws/stage1/runner.py",
            "run_stage1_training",
            {"load", "select_and_average_checkpoints", "export_stage1_encoder_pt"},
            "stage1_training_artifacts_saved",
        ),
        (
            "dma_kws/stage2/adapt.py",
            "run_stage2_adaptation",
            {"_atomic_torch_save", "append_wide_row", "merge_lora"},
            "stage2_adaptation_artifacts_saved",
        ),
    ],
)
def test_post_fit_artifacts_are_rank_zero(
    path: str,
    function_name: str,
    protected_calls: set[str],
    barrier_name: str,
):
    _assert_post_fit_artifacts_are_rank_zero(
        path,
        function_name,
        protected_calls=protected_calls,
        barrier_name=barrier_name,
    )
