"""Training checkpoints and exports must follow the immutable run identity."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch


RUNNERS = (
    ("dma_kws/stage1/runner.py", "run_stage1_training"),
    ("dma_kws/phoneme_adapter/runner.py", "run_phoneme_adapter_training"),
    ("dma_kws/stage2/train.py", "run_stage2_training"),
    ("dma_kws/stage2/adapt.py", "run_stage2_adaptation"),
)


def _function(path: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _assigned_call_line(function: ast.FunctionDef, target: str, call: str) -> int:
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(item, ast.Name) and item.id == target for item in node.targets):
            continue
        if isinstance(node.value, ast.Call):
            name = (
                node.value.func.id
                if isinstance(node.value.func, ast.Name)
                else getattr(node.value.func, "attr", "")
            )
            if name == call:
                return node.lineno
    raise AssertionError(f"No {target} = {call}(...) assignment")


@pytest.mark.parametrize(("path", "function_name"), RUNNERS)
def test_checkpoint_directory_is_scoped_after_run_context(
    path: str,
    function_name: str,
) -> None:
    function = _function(path, function_name)
    run_context_line = _assigned_call_line(
        function,
        "run_context",
        "build_run_context",
    )
    checkpoint_assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "checkpoint_dir"
            for target in node.targets
        )
    )

    assert checkpoint_assignment.lineno > run_context_line
    assert ast.unparse(checkpoint_assignment.value) == (
        "checkpoint_root / run_context.run_id"
    )


@pytest.mark.parametrize(("path", "function_name"), RUNNERS)
def test_run_record_contains_actual_primary_artifact_path(
    path: str,
    function_name: str,
) -> None:
    function = _function(path, function_name)
    string_literals = {
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "checkpoint_dir" in string_literals
    assert "primary_artifact_path" in string_literals


def test_adapt_compatibility_aliases_are_atomic_and_not_returned_as_primary() -> None:
    source = Path("dma_kws/stage2/adapt.py").read_text(encoding="utf-8")

    assert '_atomic_torch_save(adapter_payload, compatibility_aliases["phase_adapter"])' in source
    assert '_atomic_torch_save(merged_payload, compatibility_aliases["merged"])' in source
    assert '"adapter": adapter_out' in source
    assert '"merged": merged_out' in source
    assert '"compatibility_alias": json.dumps(' in source


def test_atomic_torch_save_publishes_one_complete_concurrent_payload(
    tmp_path: Path,
) -> None:
    from dma_kws.stage2.adapt import _atomic_torch_save

    destination = tmp_path / "adapter.pt"
    payloads = [
        {"writer": writer, "values": torch.full((128,), writer)}
        for writer in range(8)
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(
            executor.map(
                lambda payload: _atomic_torch_save(payload, destination),
                payloads,
            )
        )

    published = torch.load(destination, map_location="cpu", weights_only=False)
    assert torch.equal(
        published["values"],
        torch.full((128,), published["writer"]),
    )
    assert not list(tmp_path.glob(".*.tmp"))
