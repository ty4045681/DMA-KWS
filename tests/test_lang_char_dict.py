from pathlib import Path

import pytest

from dma_kws.tokenizer import CANONICAL_DICT_PATH, validate_lang_char_dict

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_lang_char_dict_passes_validation():
    validate_lang_char_dict(CANONICAL_DICT_PATH)


def test_validate_lang_char_dict_rejects_broken_dict(tmp_path):
    broken = tmp_path / "lang_char.txt"
    broken.write_text(
        "\n".join(
            [
                "<blank> 0",
                "<unk> 1",
                "AA 2",
                # Missing remaining phones and special tokens; wrong size and ids.
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        validate_lang_char_dict(broken)


def test_validate_lang_char_dict_rejects_non_contiguous_ids(tmp_path):
    lines = CANONICAL_DICT_PATH.read_text(encoding="utf-8").splitlines()
    lines[10] = "D 11"  # canonical id for D is 10; creates duplicate/gap

    broken = tmp_path / "lang_char.txt"
    broken.write_text("\n".join(lines), encoding="utf-8")

    with pytest.raises(ValueError, match="contiguous"):
        validate_lang_char_dict(broken)


def test_validate_lang_char_dict_rejects_missing_special_tokens(tmp_path):
    canonical = CANONICAL_DICT_PATH.read_text(encoding="utf-8").splitlines()
    broken_lines = [line for line in canonical if not line.startswith("<sos/eos>")]
    broken = tmp_path / "lang_char.txt"
    broken.write_text("\n".join(broken_lines), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required special tokens"):
        validate_lang_char_dict(broken)


def test_validate_lang_char_dict_rejects_missing_arpabet_phone(tmp_path):
    canonical = CANONICAL_DICT_PATH.read_text(encoding="utf-8").splitlines()
    broken_lines = [line for line in canonical if not line.startswith("ZH ")]
    broken = tmp_path / "lang_char.txt"
    broken.write_text("\n".join(broken_lines), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required ARPAbet phones"):
        validate_lang_char_dict(broken)
