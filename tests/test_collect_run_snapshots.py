"""Tests for scripts/collect_run_snapshots.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_run_snapshots.py"


def _run(dest: Path, outputs: Path, exp: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--outputs-root",
            str(outputs),
            "--exp-root",
            str(exp),
            "--dest",
            str(dest),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    outputs = tmp_path / "outputs"
    exp = tmp_path / "exp"
    hydra = outputs / "2026-10-03" / "23-00-00" / ".hydra"
    hydra.mkdir(parents=True)
    for name in ("config.yaml", "hydra.yaml", "overrides.yaml"):
        (hydra / name).write_text(f"{name}: 1\n", encoding="utf-8")
    smoke = outputs / "smoke" / "logs" / "smoke" / "version_0"
    smoke.mkdir(parents=True)
    (smoke / "hparams.yaml").write_text("run_id: smoke\n", encoding="utf-8")
    run = exp / "stage2_qbyt" / "SS-x" / "logs" / "SS-x" / "version_0"
    run.mkdir(parents=True)
    (run / "hparams.yaml").write_text("run_id: SS-x\n", encoding="utf-8")
    return outputs, exp, tmp_path / "run_snapshots"


def test_collect_mirrors_hydra_and_lightning_snapshots(tmp_path):
    outputs, exp, dest = _fixture(tmp_path)

    result = _run(dest, outputs, exp)

    assert result.returncode == 0, result.stderr
    assert (dest / "hydra" / "2026-10-03" / "23-00-00" / "config.yaml").read_text() == "config.yaml: 1\n"
    assert (dest / "hydra" / "2026-10-03" / "23-00-00" / "hydra.yaml").is_file()
    assert (dest / "hydra" / "2026-10-03" / "23-00-00" / "overrides.yaml").is_file()
    # The duplicated logs/<run> segment is collapsed for Lightning runs.
    assert (dest / "lightning" / "outputs" / "smoke" / "version_0" / "hparams.yaml").is_file()
    assert (dest / "lightning" / "exp" / "stage2_qbyt" / "SS-x" / "version_0" / "hparams.yaml").is_file()
    index = (dest / "index.tsv").read_text().splitlines()
    assert index[0] == "snapshot\tsource\tsha256\tbytes"
    assert len(index) == 6  # header + 3 hydra + 2 hparams


def test_check_reports_drift_and_recovers(tmp_path):
    outputs, exp, dest = _fixture(tmp_path)
    assert _run(dest, outputs, exp).returncode == 0
    assert _run(dest, outputs, exp, "--check").returncode == 0

    (outputs / "smoke" / "logs" / "smoke" / "version_0" / "hparams.yaml").write_text(
        "run_id: smoke-changed\n", encoding="utf-8"
    )
    drift = _run(dest, outputs, exp, "--check")
    assert drift.returncode == 1
    assert "STALE" in drift.stderr

    assert _run(dest, outputs, exp).returncode == 0
    assert _run(dest, outputs, exp, "--check").returncode == 0
