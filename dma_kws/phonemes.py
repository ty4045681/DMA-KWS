"""Text normalization and phoneme vocabulary utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

_TEXT_CLEANUP_RE = re.compile(r"[^a-z0-9'\s]+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_english_text(text: str) -> str:
    """Normalize English transcript/keyword text before G2P conversion."""
    lowered = text.lower()
    without_punctuation = _TEXT_CLEANUP_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", without_punctuation).strip()


@dataclass(frozen=True)
class PhonemeVocabulary:
    """Bidirectional phoneme-token vocabulary."""

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    unk_token: str = "<unk>"

    @classmethod
    def build(
        cls,
        phonemes: Iterable[str],
        reserved: Sequence[str] = ("<blank>", "<unk>"),
        unk_token: str = "<unk>",
    ) -> "PhonemeVocabulary":
        tokens: list[str] = []
        seen: set[str] = set()
        for token in reserved:
            if token not in seen:
                tokens.append(token)
                seen.add(token)
        for token in phonemes:
            if token not in seen:
                tokens.append(token)
                seen.add(token)
        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        id_to_token = {idx: token for token, idx in token_to_id.items()}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token, unk_token=unk_token)

    @classmethod
    def read(cls, path: str | Path, unk_token: str = "<unk>") -> "PhonemeVocabulary":
        token_to_id: dict[str, int] = {}
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid vocab line {line_number}: {stripped!r}")
                token, raw_idx = parts
                token_to_id[token] = int(raw_idx)
        id_to_token = {idx: token for token, idx in token_to_id.items()}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token, unk_token=unk_token)

    def write(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for idx in sorted(self.id_to_token):
                handle.write(f"{self.id_to_token[idx]} {idx}\n")

    def encode(self, phonemes: Sequence[str]) -> list[int]:
        unk_id = self.token_to_id[self.unk_token]
        return [self.token_to_id.get(token, unk_id) for token in phonemes]

    def decode(self, ids: Sequence[int]) -> list[str]:
        return [self.id_to_token[idx] for idx in ids]
