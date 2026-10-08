"""Tests for scripts/checkpoint_release.py."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from scripts.checkpoint_release import (
    COLUMNS,
    asset_name,
    format_readout,
    hydra_overrides,
    iter_model_files,
    pending_rows,
    read_manifest,
    readout_spec,
    snapshot_candidates,
    stage_asset,
    unique_asset_names,
    write_manifest,
)


def test_asset_name_encodes_the_data_relative_path():
    name = asset_name("exp/stage2_qbyt/checkpoints/v41/last.ckpt")
    assert name == "exp__stage2_qbyt__checkpoints__v41__last.ckpt"
    assert asset_name("exp/x/step_step=001500.ckpt") == "exp__x__step_step-001500.ckpt"
    assert len(asset_name("a/" + "b" * 300 + ".pt")) <= 200


def test_unique_asset_names_disambiguates_sanitised_collisions():
    # Sanitising turns b=c into b-c, which collides with the real b-c file.
    names = unique_asset_names(["a/b=c.pt", "a/b-c.pt"])
    assert len(set(names.values())) == 2
    assert names["a/b=c.pt"] != names["a/b-c.pt"]


def test_manifest_round_trip(tmp_path):
    row = {
        "asset": "a.pt",
        "relpath": "exp/a.pt",
        "bytes": 3,
        "sha256": "abc",
        "readout": "",
        "run": "exp",
        "snapshot": "",
    }
    path = tmp_path / "manifest.tsv"
    write_manifest(path, [row])
    assert path.read_text(encoding="utf-8").splitlines()[0].split("\t") == list(COLUMNS)
    assert read_manifest(path) == [{column: str(row[column]) for column in COLUMNS}]


def test_snapshot_candidates_collapse_the_repeated_run_directory():
    run, candidates = snapshot_candidates(
        "exp/stage2_qbyt/sinkhead_stage2/SS-x/checkpoints/SS-x/version_0/last.ckpt"
    )
    assert run == "stage2_qbyt/sinkhead_stage2/SS-x"
    assert "stage2_qbyt/sinkhead_stage2/SS-x/version_0" in candidates

    _, plain = snapshot_candidates(
        "exp/stage2_qbyt/final/P1/checkpoints/P1/version_0/last.ckpt"
    )
    assert "stage2_qbyt/final/P1/version_0" in plain
    assert snapshot_candidates("exp/author_v1/stage1_v1.pt") == ("author_v1", [])


def test_iter_model_files_skips_non_weights(tmp_path):
    run = tmp_path / "exp" / "run"
    run.mkdir(parents=True)
    (run / "last.ckpt").write_bytes(b"x")
    (run / "hparams.yaml").write_text("y", encoding="utf-8")
    assert iter_model_files(tmp_path) == ["exp/run/last.ckpt"]


def test_pending_rows_skips_uploaded_sizes_and_filters():
    rows = [
        {"asset": "a", "relpath": "exp/run/last.ckpt", "bytes": "1", "sha256": "d", "run": "r", "snapshot": ""},
        {"asset": "b", "relpath": "exp/run/other.pt", "bytes": "2", "sha256": "e", "run": "r", "snapshot": ""},
    ]
    assert [row["asset"] for row in pending_rows(rows, {})] == ["a", "b"]
    assert [row["asset"] for row in pending_rows(rows, {"a": 1})] == ["b"]
    # A size mismatch means the asset has to be re-uploaded.
    assert [row["asset"] for row in pending_rows(rows, {"a": 99})] == ["a", "b"]
    assert [row["asset"] for row in pending_rows(rows, {}, only="other")] == ["b"]
    assert [row["asset"] for row in pending_rows(rows, {}, limit=1)] == ["a"]


@dataclass
class _Pooling:
    mode: str = "eps_softmin"
    sink_readout: str = "additive"
    sink_zero_init: bool = True
    score_temperature: float | None = None


def test_hydra_overrides_list_every_pooling_field():
    spec = SimpleNamespace(version=4, value=_Pooling())
    lines = hydra_overrides(spec)
    assert lines[0] == "stage2.qbyt_readout_version=4"
    assert "stage2.qbyt_readout.mode=eps_softmin" in lines
    assert "stage2.qbyt_readout.sink_readout=additive" in lines
    assert "stage2.qbyt_readout.sink_zero_init=true" in lines
    assert "stage2.qbyt_readout.score_temperature=null" in lines


def test_hydra_overrides_for_v1_only_sets_the_version():
    spec = SimpleNamespace(version=1, value=SimpleNamespace(readout="gru_last_padded"))
    assert hydra_overrides(spec) == ["stage2.qbyt_readout_version=1"]


class _Multiline:
    def __str__(self) -> str:
        return "a b\n  c\td"


def test_format_readout_is_empty_for_no_spec_and_collapses_whitespace():
    assert format_readout(None) == ""
    assert format_readout(_Multiline()) == "a b c d"


def test_readout_spec_returns_none_for_a_non_checkpoint(tmp_path):
    bogus = tmp_path / "encoder.pt"
    bogus.write_bytes(b"not a torch checkpoint")
    assert readout_spec(bogus) is None


def test_stage_asset_names_the_symlink_after_the_asset(tmp_path):
    target = tmp_path / "deep" / "last.ckpt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    row = {"asset": "exp__deep__last.ckpt", "relpath": "deep/last.ckpt", "bytes": "1", "sha256": "d"}
    link = stage_asset(row, tmp_path, tmp_path / "staging")
    assert link.name == "exp__deep__last.ckpt"
    assert link.resolve() == target.resolve()
