# Two-Stage Real Demo Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first implementation slice for the two-stage real demo: repository hygiene, root ignore rules, shared config loading, phoneme vocabulary/text utilities, and Stage I candidate matching utilities.

**Architecture:** Add a small `dma_kws` Python package that is independent of PyTorch for foundational functionality, so it can be tested locally even before GPU dependencies are installed. Later Stage I/Stage II training scripts will import these utilities instead of duplicating path, vocabulary, and candidate-search logic.

**Tech Stack:** Python 3.10+, pytest, PyYAML, standard library dataclasses/pathlib/json/re.

---

## File Structure

- Create: `/Users/e4/Documents/dma-kws/.gitignore` — root ignore rules for Python caches, local data, experiments, checkpoints, and worktrees.
- Create: `/Users/e4/Documents/dma-kws/dma_kws/__init__.py` — package marker and version.
- Create: `/Users/e4/Documents/dma-kws/dma_kws/config.py` — YAML config loader with path expansion and required section validation.
- Create: `/Users/e4/Documents/dma-kws/dma_kws/phonemes.py` — text normalization, optional G2P wrapper, phoneme vocabulary read/write/encode/decode.
- Create: `/Users/e4/Documents/dma-kws/dma_kws/stage1/__init__.py` — Stage I package marker.
- Create: `/Users/e4/Documents/dma-kws/dma_kws/stage1/candidates.py` — approximate keyword phoneme subsequence matching and candidate span expansion.
- Create: `/Users/e4/Documents/dma-kws/configs/demo_librispeech100.yaml` — configurable small-scale real-training config for 2× V100.
- Create: `/Users/e4/Documents/dma-kws/tests/test_config.py` — config loader tests.
- Create: `/Users/e4/Documents/dma-kws/tests/test_phonemes.py` — vocabulary/text utility tests.
- Create: `/Users/e4/Documents/dma-kws/tests/test_stage1_candidates.py` — minimal candidate matching smoke test.

---

### Task 1: Root `.gitignore` and cache cleanup

**Files:**
- Create: `/Users/e4/Documents/dma-kws/.gitignore`
- Delete untracked cache directories under `/Users/e4/Documents/dma-kws/qbyt/**/__pycache__`

- [ ] **Step 1: Write root ignore rules**

Create `/Users/e4/Documents/dma-kws/.gitignore` with:

```gitignore
# Python caches
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Virtual environments
.venv/
venv/
env/

# macOS / editor files
.DS_Store
.vscode/
.idea/

# Local data and experiment outputs
data/
exp/
outputs/
checkpoints/
*.ckpt
*.pt
*.pth
*.onnx

# TensorBoard and Lightning logs
lightning_logs/
runs/
qbyt/logs/

# Local worktrees
.worktrees/
worktrees/
```

- [ ] **Step 2: Clean untracked cache directories**

Run:

```bash
find /Users/e4/Documents/dma-kws -type d -name __pycache__ -prune -exec rm -rf {} +
```

Expected: command exits 0.

- [ ] **Step 3: Verify ignore behavior**

Run:

```bash
git check-ignore -q qbyt/__pycache__/x.pyc && echo ignored-pycache
git check-ignore -q data/example && echo ignored-data
git check-ignore -q .worktrees/example && echo ignored-worktree
```

Expected output includes:

```text
ignored-pycache
ignored-data
ignored-worktree
```

- [ ] **Step 4: Commit hygiene change**

Run:

```bash
git add /Users/e4/Documents/dma-kws/.gitignore
git commit -m "chore: add root gitignore"
```

Expected: commit succeeds and only `.gitignore` is committed.

---

### Task 2: Config loader foundation

**Files:**
- Create: `/Users/e4/Documents/dma-kws/tests/test_config.py`
- Create: `/Users/e4/Documents/dma-kws/dma_kws/__init__.py`
- Create: `/Users/e4/Documents/dma-kws/dma_kws/config.py`

- [ ] **Step 1: Write failing config tests**

Create `/Users/e4/Documents/dma-kws/tests/test_config.py`:

```python
from pathlib import Path

import pytest

from dma_kws.config import load_config, require_sections


def test_load_config_expands_user_and_env_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("DMA_KWS_TEST_ROOT", str(tmp_path))
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "paths:\n"
        "  data_root: ${DMA_KWS_TEST_ROOT}/data\n"
        "  cache_root: ~/dma-kws-cache\n"
        "stage1:\n"
        "  batch_size_per_gpu: 48\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config["paths"]["data_root"] == str(tmp_path / "data")
    assert config["paths"]["cache_root"].startswith(str(Path.home()))
    assert config["stage1"]["batch_size_per_gpu"] == 48


def test_require_sections_reports_missing_sections():
    with pytest.raises(ValueError, match="Missing required config sections: stage2, demo"):
        require_sections({"paths": {}, "stage1": {}}, ["paths", "stage1", "stage2", "demo"])
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest /Users/e4/Documents/dma-kws/tests/test_config.py -q
```

Expected: FAIL because `dma_kws.config` does not exist.

- [ ] **Step 3: Implement minimal config loader**

Create `/Users/e4/Documents/dma-kws/dma_kws/__init__.py`:

```python
"""Utilities for the DMA-KWS two-stage demo."""

__version__ = "0.1.0"
```

Create `/Users/e4/Documents/dma-kws/dma_kws/config.py`:

```python
"""Configuration helpers for DMA-KWS scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import yaml


def _expand_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_value(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and expand env/user markers in string values."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at top level: {config_path}")
    return _expand_value(data)


def require_sections(config: dict[str, Any], sections: Iterable[str]) -> None:
    """Raise a readable error when required top-level config sections are absent."""
    missing = [section for section in sections if section not in config]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required config sections: {joined}")
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
pytest /Users/e4/Documents/dma-kws/tests/test_config.py -q
```

Expected: PASS.

---

### Task 3: Phoneme vocabulary and text utilities

**Files:**
- Create: `/Users/e4/Documents/dma-kws/tests/test_phonemes.py`
- Create: `/Users/e4/Documents/dma-kws/dma_kws/phonemes.py`

- [ ] **Step 1: Write failing phoneme tests**

Create `/Users/e4/Documents/dma-kws/tests/test_phonemes.py`:

```python
from dma_kws.phonemes import PhonemeVocabulary, normalize_english_text


def test_normalize_english_text_lowercases_and_removes_punctuation():
    assert normalize_english_text("Hello, WORLD! It's 2026.") == "hello world it's 2026"


def test_phoneme_vocabulary_round_trip(tmp_path):
    vocab = PhonemeVocabulary.build(["HH", "AH", "L", "OW", "HH"], reserved=("<blank>", "<unk>"))

    assert vocab.token_to_id["<blank>"] == 0
    assert vocab.token_to_id["<unk>"] == 1
    assert vocab.encode(["HH", "MISSING", "OW"]) == [2, 1, 5]
    assert vocab.decode([2, 1, 5]) == ["HH", "<unk>", "OW"]

    path = tmp_path / "phoneme_vocab.txt"
    vocab.write(path)
    loaded = PhonemeVocabulary.read(path)

    assert loaded.token_to_id == vocab.token_to_id
    assert loaded.id_to_token == vocab.id_to_token
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest /Users/e4/Documents/dma-kws/tests/test_phonemes.py -q
```

Expected: FAIL because `dma_kws.phonemes` does not exist.

- [ ] **Step 3: Implement phoneme utilities**

Create `/Users/e4/Documents/dma-kws/dma_kws/phonemes.py`:

```python
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
        for token in sorted(set(phonemes)):
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
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
pytest /Users/e4/Documents/dma-kws/tests/test_phonemes.py -q
```

Expected: PASS.

---

### Task 4: Stage I candidate matching utility

**Files:**
- Create: `/Users/e4/Documents/dma-kws/tests/test_stage1_candidates.py`
- Create: `/Users/e4/Documents/dma-kws/dma_kws/stage1/__init__.py`
- Create: `/Users/e4/Documents/dma-kws/dma_kws/stage1/candidates.py`

- [ ] **Step 1: Write failing candidate tests**

Create `/Users/e4/Documents/dma-kws/tests/test_stage1_candidates.py`:

```python
from dma_kws.stage1.candidates import PhonemeFrame, find_keyword_candidates


def test_find_keyword_candidates_returns_time_span_with_margin():
    frames = [
        PhonemeFrame("SIL", 0.00, 0.10, -0.1),
        PhonemeFrame("HH", 0.10, 0.20, -0.2),
        PhonemeFrame("AH", 0.20, 0.30, -0.3),
        PhonemeFrame("L", 0.30, 0.40, -0.4),
        PhonemeFrame("OW", 0.40, 0.50, -0.5),
        PhonemeFrame("SIL", 0.50, 0.60, -0.1),
    ]

    candidates = find_keyword_candidates(frames, ["HH", "AH", "L", "OW"], margin_sec=0.05)

    assert len(candidates) == 1
    assert candidates[0].start_sec == 0.05
    assert candidates[0].end_sec == 0.55
    assert candidates[0].phonemes == ["HH", "AH", "L", "OW"]


```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest /Users/e4/Documents/dma-kws/tests/test_stage1_candidates.py -q
```

Expected: FAIL because `dma_kws.stage1.candidates` does not exist.

- [ ] **Step 3: Implement candidate matching**

Create `/Users/e4/Documents/dma-kws/dma_kws/stage1/__init__.py`:

```python
"""Stage I phoneme CTC utilities."""
```

Create `/Users/e4/Documents/dma-kws/dma_kws/stage1/candidates.py`:

```python
"""Candidate generation from decoded phoneme frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PhonemeFrame:
    """A decoded phoneme with an approximate time span and log score."""

    phoneme: str
    start_sec: float
    end_sec: float
    log_score: float = 0.0


@dataclass(frozen=True)
class KeywordCandidate:
    """A Stage I candidate region for Stage II verification."""

    start_sec: float
    end_sec: float
    stage1_score: float
    phonemes: list[str]


def _candidate_from_window(
    frames: Sequence[PhonemeFrame],
    start_index: int,
    end_index: int,
    margin_sec: float,
) -> KeywordCandidate:
    window = list(frames[start_index : end_index + 1])
    start_sec = max(0.0, window[0].start_sec - margin_sec)
    end_sec = window[-1].end_sec + margin_sec
    score = sum(frame.log_score for frame in window)
    return KeywordCandidate(
        start_sec=round(start_sec, 6),
        end_sec=round(end_sec, 6),
        stage1_score=score,
        phonemes=[frame.phoneme for frame in window],
    )


def find_keyword_candidates(
    decoded_frames: Sequence[PhonemeFrame],
    keyword_phonemes: Sequence[str],
    *,
    margin_sec: float = 0.0,
    max_insertions: int = 0,
) -> list[KeywordCandidate]:
    """Find approximate keyword subsequence matches in decoded phoneme frames.

    `max_insertions` allows extra decoded phones between keyword phones. The
    first version uses a deterministic left-to-right scan so demo behavior is
    easy to reason about and test.
    """
    if not keyword_phonemes:
        return []

    candidates: list[KeywordCandidate] = []
    frame_count = len(decoded_frames)
    keyword_count = len(keyword_phonemes)

    for start_index in range(frame_count):
        if decoded_frames[start_index].phoneme != keyword_phonemes[0]:
            continue

        keyword_index = 1
        insertions = 0
        end_index = start_index
        cursor = start_index + 1

        while cursor < frame_count and keyword_index < keyword_count:
            if decoded_frames[cursor].phoneme == keyword_phonemes[keyword_index]:
                keyword_index += 1
                end_index = cursor
            else:
                insertions += 1
                if insertions > max_insertions:
                    break
                end_index = cursor
            cursor += 1

        if keyword_index == keyword_count:
            candidates.append(_candidate_from_window(decoded_frames, start_index, end_index, margin_sec))

    return candidates
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
pytest /Users/e4/Documents/dma-kws/tests/test_stage1_candidates.py -q
```

Expected: PASS.

---

### Task 5: Demo config file

**Files:**
- Create: `/Users/e4/Documents/dma-kws/configs/demo_librispeech100.yaml`

- [ ] **Step 1: Create demo config**

Create `/Users/e4/Documents/dma-kws/configs/demo_librispeech100.yaml`:

```yaml
paths:
  data_root: /data/dma-kws
  librispeech_root: /data/dma-kws/raw/LibriSpeech
  libriphrase100_root: /data/dma-kws/raw/LibriPhrase-100
  processed_root: /data/dma-kws/processed
  feature_root: /data/dma-kws/features
  exp_root: /data/dma-kws/exp

stage1:
  input_dim: 80
  encoder_output_dim: 144
  attention_heads: 4
  linear_units: 576
  num_blocks: 6
  batch_size_per_gpu: 48
  max_epochs: 30
  learning_rate: 0.001
  checkpoint_dir: /data/dma-kws/exp/stage1_phoneme_ctc/checkpoints

stage2:
  encoder_output_dim: 144
  qbyt_embed_dim: 128
  qbyt_layers: 2
  batch_size_per_gpu: 128
  max_steps: 50000
  learning_rate: 0.001
  checkpoint_dir: /data/dma-kws/exp/stage2_qbyt/checkpoints

demo:
  stage1_candidate_margin_sec: 0.15
  max_stage1_insertions: 2
  qbyt_threshold: 0.5
```

- [ ] **Step 2: Verify config loads**

Run:

```bash
python - <<'PY'
from dma_kws.config import load_config, require_sections
config = load_config('/Users/e4/Documents/dma-kws/configs/demo_librispeech100.yaml')
require_sections(config, ['paths', 'stage1', 'stage2', 'demo'])
print(config['stage1']['encoder_output_dim'])
PY
```

Expected output:

```text
144
```

---

### Task 6: Run foundation test suite and commit

**Files:**
- Stage all files from Tasks 2–5.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest /Users/e4/Documents/dma-kws/tests/test_config.py /Users/e4/Documents/dma-kws/tests/test_phonemes.py /Users/e4/Documents/dma-kws/tests/test_stage1_candidates.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Check git status**

Run:

```bash
git status --short
```

Expected: new `dma_kws/`, `tests/`, `configs/`, and plan file are present; no `__pycache__` files are untracked.

- [ ] **Step 3: Commit foundation implementation**

Run:

```bash
git add /Users/e4/Documents/dma-kws/dma_kws /Users/e4/Documents/dma-kws/tests /Users/e4/Documents/dma-kws/configs /Users/e4/Documents/dma-kws/docs/superpowers/plans/2026-06-17-two-stage-real-demo-foundation.md
git commit -m "feat: add two-stage demo foundation utilities"
```

Expected: commit succeeds.
