from pathlib import Path

import pytest

from dma_kws.training.resume import resolve_resume_path


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
