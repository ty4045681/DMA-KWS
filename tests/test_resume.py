from pathlib import Path

import pytest

from dma_kws.training.resume import (
    resolve_resume_path,
    resolve_versioned_resume_path,
)


def test_empty_string_returns_none(tmp_path: Path) -> None:
    assert resolve_resume_path("", tmp_path) is None


def test_last_resolves_to_checkpoint_dir_last_ckpt(tmp_path: Path) -> None:
    last_ckpt = tmp_path / "last.ckpt"
    last_ckpt.write_bytes(b"")

    result = resolve_resume_path("last", tmp_path)

    assert result == last_ckpt


def test_explicit_existing_path_returned_as_is(tmp_path: Path) -> None:
    ckpt = tmp_path / "stage1_epoch000_step000020.ckpt"
    ckpt.write_bytes(b"")

    result = resolve_resume_path(str(ckpt), tmp_path)

    assert result == Path(str(ckpt))


def test_missing_explicit_path_raises_system_exit(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.ckpt"

    with pytest.raises(SystemExit):
        resolve_resume_path(str(missing), tmp_path)


def test_last_without_last_ckpt_raises_system_exit(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        resolve_resume_path("last", tmp_path)


def test_versioned_last_selects_newest_run_version(tmp_path: Path) -> None:
    old = tmp_path / "qbyt" / "version_2" / "last.ckpt"
    newest = tmp_path / "qbyt" / "version_7" / "last.ckpt"
    old.parent.mkdir(parents=True)
    newest.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    newest.write_bytes(b"new")
    (tmp_path / "qbyt" / "version_9").mkdir()

    assert resolve_versioned_resume_path("last", tmp_path, "qbyt") == newest


def test_versioned_last_falls_back_to_legacy_root_checkpoint(tmp_path: Path) -> None:
    legacy = tmp_path / "last.ckpt"
    legacy.write_bytes(b"legacy")

    assert resolve_versioned_resume_path("last", tmp_path, "qbyt") == legacy


def test_versioned_resume_keeps_explicit_path_unchanged(tmp_path: Path) -> None:
    explicit = tmp_path / "elsewhere" / "selected.ckpt"
    explicit.parent.mkdir()
    explicit.write_bytes(b"checkpoint")

    assert (
        resolve_versioned_resume_path(str(explicit), tmp_path, "qbyt")
        == explicit
    )
