# Stage II Decoded Parquet Audio Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stage II prep/train consume audio directly from LibriPhrase-100 `LP-100-decoded-*.parquet` shards, materializing only the referenced clips as `.npy` files — no loose `.wav` files required.

**Architecture:** The prep script builds anchors and pairs as today, collects the set of referenced clip paths, normalizes each clip path (`clips` value minus the `LP-100/` prefix == decoded parquet `audio_rel`), streams the decoded shards extracting only matching rows, and writes each as a float32 `.npy` under `<processed_root>/stage2_qbyt/audio/<audio_rel>.npy`. The JSONL `wav_path` points at the `.npy` relative path. The train script's Dataset loads via `np.load` instead of `torchaudio.load`, dropping channel-merge/resample since the decoded audio is already 16 kHz mono.

**Tech Stack:** Python 3.9, pandas/pyarrow (parquet), numpy, torch/torchaudio (`torchaudio.compliance.kaldi` for fbank). All already in `requirements.txt`.

## Global Constraints

- No new dependencies — only `numpy`, `pandas`, `pyarrow`, `torch`, `torchaudio` (all already in `requirements.txt`).
- `clips` path → `audio_rel` key: strip a leading `LP-100/` prefix (use `str.removeprefix` — Python 3.9 lacks it, so use a helper that slices the prefix if present; see Task 1).
- `.npy` files are float32, mono, 16 kHz, saved under `<processed_root>/stage2_qbyt/audio/<audio_rel>.npy`.
- Unmatched clips must be counted and warned at end of prep — never silently dropped.
- Follow existing patterns: parquet reading helpers live in `dma_kws/stage1/librispeech.py` style (small monkeypatchable functions); pure data logic lives in `dma_kws/stage2/pairs.py`.

---

### Task 1: Add clip-path normalization helper + audio_rel matching to `pairs.py`

**Files:**
- Modify: `dma_kws/stage2/pairs.py`
- Test: `tests/test_stage2_pairs.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `def clip_to_audio_rel(clip_path: str) -> str` — strips a leading `"LP-100/"` segment if present, returns the remainder (the decoded-parquet `audio_rel` key). Idempotent if no prefix.
  - `PairRecord` gains a field `sample_rate: int` (defaults not allowed in frozen dataclass ordering — add it as a required field after `label`). `to_json_dict()` continues to use `asdict`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage2_pairs.py`:

```python
from dma_kws.stage2.pairs import AnchorExample, PairRecord, clip_to_audio_rel, make_pair_records


def test_clip_to_audio_rel_strips_lp100_prefix():
    assert clip_to_audio_rel("LP-100/missus_rachel/103-1240-0000_000.wav") == "missus_rachel/103-1240-0000_000.wav"


def test_clip_to_audio_rel_without_prefix_is_identity():
    assert clip_to_audio_rel("missus_rachel/103-1240-0000_000.wav") == "missus_rachel/103-1240-0000_000.wav"


def test_pair_record_carries_sample_rate():
    record = PairRecord(
        anchor_text="hi",
        anchor_phonemes=["HH", "AY"],
        wav_path="audio/a.npy",
        label=1,
        sample_rate=16000,
    )
    assert record.to_json_dict()["sample_rate"] == 16000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stage2_pairs.py -v`
Expected: FAIL — `ImportError: cannot import name 'clip_to_audio_rel'` and `PairRecord.__init__` missing `sample_rate`.

- [ ] **Step 3: Write minimal implementation**

In `dma_kws/stage2/pairs.py`, add the helper near the top (after imports):

```python
def clip_to_audio_rel(clip_path: str) -> str:
    """Map a LibriPhrase `clips` path to a decoded-parquet `audio_rel` key.

    The aggregated parquet stores clip paths like `LP-100/<ngram>/<id>.wav`
    while the decoded shards key audio by `audio_rel` = `<ngram>/<id>.wav`.
    """
    prefix = "LP-100/"
    if clip_path.startswith(prefix):
        return clip_path[len(prefix):]
    return clip_path
```

Update `PairRecord`:

```python
@dataclass(frozen=True)
class PairRecord:
    """One QbyT training/eval pair."""

    anchor_text: str
    anchor_phonemes: list[str]
    wav_path: str
    label: int
    sample_rate: int

    def to_json_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Update `make_pair_records` to set `sample_rate`**

`make_pair_records` does not know the real sample rate (audio is materialized later), so add a `sample_rate: int = 16000` keyword param and pass it into every `PairRecord(...)`:

```python
def make_pair_records(
    anchors: Sequence[AnchorExample],
    *,
    negatives_per_anchor: int = 1,
    seed: int = 2025,
    sample_rate: int = 16000,
) -> list[PairRecord]:
```

Add `sample_rate=sample_rate` to both `PairRecord(...)` constructions (positive at the top of the loop, negative inside the inner loop).

- [ ] **Step 5: Fix the existing test for the new required field**

The existing `test_make_pair_records_creates_positive_and_negative_pair` constructs no `PairRecord` directly, so it still passes — but it asserts `pairs[0].wav_path == "a.wav"`. That stays valid. No change needed; just confirm.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_stage2_pairs.py -v`
Expected: PASS (all tests).

- [ ] **Step 7: Commit**

```bash
git add dma_kws/stage2/pairs.py tests/test_stage2_pairs.py
git commit -m "feat(stage2): add clip_to_audio_rel + sample_rate on PairRecord"
```

---

### Task 2: Add decoded-parquet audio extraction helper to `pairs.py`

**Files:**
- Modify: `dma_kws/stage2/pairs.py`
- Test: `tests/test_stage2_pairs.py`

**Interfaces:**
- Consumes: `clip_to_audio_rel` (Task 1).
- Produces:
  - `def iter_decoded_audio_rows(parquet_paths, needed_keys, *, read_parquet)` — a generator yielding `(audio_rel: str, audio: list[float], sampling_rate: int)` tuples for rows whose `audio_rel` is in `needed_keys`. `read_parquet` is an injected callable `(Path) -> DataFrame` (default wired in the prep script) so the unit test can monkeypatch without pyarrow. Iterates shards lazily; does not hold all rows in memory.

This is pure iteration logic — kept in `pairs.py` next to the other Stage II data helpers so it is unit-testable with a fake `read_parquet`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage2_pairs.py`:

```python
from pathlib import Path

from dma_kws.stage2.pairs import iter_decoded_audio_rows


class _FakeDF:
    def __init__(self, rows):
        self._rows = rows

    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, row


def test_iter_decoded_audio_rows_filters_to_needed_keys():
    shard_rows = {
        Path("0000.parquet"): [
            {"audio_rel": "a/x.wav", "audio": [0.1, 0.2], "sampling_rate": 16000},
            {"audio_rel": "a/y.wav", "audio": [0.3], "sampling_rate": 16000},
        ],
        Path("0001.parquet"): [
            {"audio_rel": "b/z.wav", "audio": [0.4, 0.5], "sampling_rate": 16000},
        ],
    }

    def fake_read(path):
        return _FakeDF(shard_rows[path])

    needed = {"a/x.wav", "b/z.wav"}
    got = list(
        iter_decoded_audio_rows(
            [Path("0000.parquet"), Path("0001.parquet")],
            needed,
            read_parquet=fake_read,
        )
    )

    assert [(rel, sr) for rel, _, sr in got] == [("a/x.wav", 16000), ("b/z.wav", 16000)]
    assert got[0][1] == [0.1, 0.2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stage2_pairs.py::test_iter_decoded_audio_rows_filters_to_needed_keys -v`
Expected: FAIL — `ImportError: cannot import name 'iter_decoded_audio_rows'`.

- [ ] **Step 3: Write minimal implementation**

Add to `dma_kws/stage2/pairs.py`:

```python
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


def iter_decoded_audio_rows(
    parquet_paths: Iterable[Path],
    needed_keys: set[str],
    *,
    read_parquet: Callable[[Path], Any],
) -> Iterator[tuple[str, Any, int]]:
    """Yield (audio_rel, audio, sampling_rate) for rows whose audio_rel is needed.

    Streams one parquet shard at a time via the injected ``read_parquet``
    callable so the full ~5 GB of decoded audio never materializes at once.
    """
    remaining = set(needed_keys)
    for parquet_path in parquet_paths:
        if not remaining:
            break
        frame = read_parquet(parquet_path)
        for _, row in frame.iterrows():
            audio_rel = str(row["audio_rel"])
            if audio_rel not in remaining:
                continue
            remaining.discard(audio_rel)
            yield audio_rel, row["audio"], int(row["sampling_rate"])
```

Note: `set[str]` and `tuple[...]` subscripting work at runtime here because `from __future__ import annotations` is already at the top of `pairs.py` for the annotations, and the runtime `set(needed_keys)` call uses the builtin. Confirm `from __future__ import annotations` is present (it is).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_stage2_pairs.py::test_iter_decoded_audio_rows_filters_to_needed_keys -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dma_kws/stage2/pairs.py tests/test_stage2_pairs.py
git commit -m "feat(stage2): stream decoded parquet rows filtered to needed clips"
```

---

### Task 3: Wire decoded-parquet extraction + `.npy` materialization into prep script

**Files:**
- Modify: `scripts/prepare_stage2_libriphrase.py`
- Test: `tests/test_prepare_stage2_libriphrase.py` (create)

**Interfaces:**
- Consumes: `clip_to_audio_rel`, `iter_decoded_audio_rows`, `make_pair_records`, `AnchorExample` (Tasks 1–2).
- Produces:
  - `def materialize_pairs(pairs, *, decoded_parquet_paths, audio_dir, read_parquet) -> tuple[list[PairRecord], int]` — extracts referenced clips to `.npy`, rewrites each `PairRecord.wav_path` to the `.npy` path relative to `audio_dir.parent` (i.e. `audio/<audio_rel>.npy`), and returns `(rewritten_pairs, unmatched_count)`.
  - New CLI arg `--decoded-parquet-root` (default `""` → falls back to `paths.libriphrase100_root`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_prepare_stage2_libriphrase.py`:

```python
import json
from pathlib import Path

import numpy as np

from dma_kws.stage2.pairs import PairRecord
from scripts import prepare_stage2_libriphrase as prep


class _FakeDF:
    def __init__(self, rows):
        self._rows = rows

    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, row


def test_materialize_pairs_writes_npy_and_rewrites_paths(tmp_path):
    audio_dir = tmp_path / "stage2_qbyt" / "audio"
    pairs = [
        PairRecord(
            anchor_text="hi",
            anchor_phonemes=["HH", "AY"],
            wav_path="LP-100/hi/1-2-3_000.wav",
            label=1,
            sample_rate=16000,
        ),
        PairRecord(
            anchor_text="hi",
            anchor_phonemes=["HH", "AY"],
            wav_path="LP-100/bye/4-5-6_000.wav",
            label=0,
            sample_rate=16000,
        ),
    ]
    rows = [
        {"audio_rel": "hi/1-2-3_000.wav", "audio": [0.1, 0.2, 0.3], "sampling_rate": 16000},
        {"audio_rel": "bye/4-5-6_000.wav", "audio": [0.4, 0.5], "sampling_rate": 16000},
    ]

    def fake_read(path):
        return _FakeDF(rows)

    rewritten, unmatched = prep.materialize_pairs(
        pairs,
        decoded_parquet_paths=[Path("0000.parquet")],
        audio_dir=audio_dir,
        read_parquet=fake_read,
    )

    assert unmatched == 0
    assert rewritten[0].wav_path == "audio/hi/1-2-3_000.wav"
    saved = np.load(audio_dir / "hi" / "1-2-3_000.wav.npy")
    assert saved.dtype == np.float32
    np.testing.assert_allclose(saved, np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_materialize_pairs_counts_unmatched(tmp_path):
    audio_dir = tmp_path / "stage2_qbyt" / "audio"
    pairs = [
        PairRecord(
            anchor_text="hi",
            anchor_phonemes=["HH", "AY"],
            wav_path="LP-100/missing/9-9-9_000.wav",
            label=1,
            sample_rate=16000,
        )
    ]

    def fake_read(path):
        return _FakeDF([])

    rewritten, unmatched = prep.materialize_pairs(
        pairs,
        decoded_parquet_paths=[Path("0000.parquet")],
        audio_dir=audio_dir,
        read_parquet=fake_read,
    )

    assert unmatched == 1
    assert rewritten == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prepare_stage2_libriphrase.py -v`
Expected: FAIL — `AttributeError: module 'scripts.prepare_stage2_libriphrase' has no attribute 'materialize_pairs'`.

- [ ] **Step 3: Write `materialize_pairs` + helpers in the prep script**

Add imports near the top of `scripts/prepare_stage2_libriphrase.py`:

```python
import numpy as np

from dma_kws.stage2.pairs import (
    AnchorExample,
    PairRecord,
    clip_to_audio_rel,
    iter_decoded_audio_rows,
    make_pair_records,
)
```

Add a default parquet reader and `materialize_pairs`:

```python
def _read_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


def find_decoded_parquets(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    matches = sorted(root.rglob("LP-100-decoded-*.parquet"))
    if not matches:
        matches = sorted(root.rglob("*.parquet"))
    if not matches:
        raise SystemExit(f"No decoded parquet shards found under {root}")
    return matches


def materialize_pairs(
    pairs: list[PairRecord],
    *,
    decoded_parquet_paths: list[Path],
    audio_dir: Path,
    read_parquet=_read_parquet,
) -> tuple[list[PairRecord], int]:
    """Extract referenced clips to .npy and rewrite each pair's wav_path.

    Returns (rewritten_pairs, unmatched_count). Pairs whose audio could not
    be found in any decoded shard are dropped from the returned list.
    """
    needed = {clip_to_audio_rel(pair.wav_path) for pair in pairs}
    audio_dir.mkdir(parents=True, exist_ok=True)
    rel_for_key: dict[str, str] = {}
    for audio_rel, audio, _sr in iter_decoded_audio_rows(
        decoded_parquet_paths, needed, read_parquet=read_parquet
    ):
        out_path = audio_dir / f"{audio_rel}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, np.asarray(audio, dtype=np.float32))
        # wav_path is relative to audio_dir.parent (the stage2_qbyt dir)
        rel_for_key[audio_rel] = f"{audio_dir.name}/{audio_rel}.npy"

    rewritten: list[PairRecord] = []
    unmatched = 0
    for pair in pairs:
        key = clip_to_audio_rel(pair.wav_path)
        rel = rel_for_key.get(key)
        if rel is None:
            unmatched += 1
            continue
        rewritten.append(
            PairRecord(
                anchor_text=pair.anchor_text,
                anchor_phonemes=pair.anchor_phonemes,
                wav_path=rel,
                label=pair.label,
                sample_rate=pair.sample_rate,
            )
        )
    return rewritten, unmatched
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prepare_stage2_libriphrase.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_stage2_libriphrase.py tests/test_prepare_stage2_libriphrase.py
git commit -m "feat(stage2): materialize referenced clips from decoded parquet to npy"
```

---

### Task 4: Update prep `main()` flow + CLI arg

**Files:**
- Modify: `scripts/prepare_stage2_libriphrase.py`

**Interfaces:**
- Consumes: `materialize_pairs`, `find_decoded_parquets` (Task 3).
- Produces: end-to-end prep that writes `.npy` audio + JSONL with `.npy` `wav_path`s.

- [ ] **Step 1: Add the `--decoded-parquet-root` CLI arg**

In `parse_args()`, after the existing args:

```python
    parser.add_argument(
        "--decoded-parquet-root",
        default="",
        help="Directory or file with LP-100-decoded-*.parquet shards; defaults to libriphrase100_root",
    )
```

- [ ] **Step 2: Rewrite the tail of `main()`**

Replace the block that currently writes pairs (from `pairs = make_pair_records(...)` through the final `print`) with:

```python
    pairs = make_pair_records(
        anchors,
        negatives_per_anchor=args.negatives_per_anchor,
        seed=args.seed,
    )

    libriphrase_root = Path(paths["libriphrase100_root"])
    decoded_root = Path(args.decoded_parquet_root) if args.decoded_parquet_root else libriphrase_root
    decoded_parquet_paths = find_decoded_parquets(decoded_root)
    audio_dir = output_dir / "audio"

    rewritten, unmatched = materialize_pairs(
        pairs,
        decoded_parquet_paths=decoded_parquet_paths,
        audio_dir=audio_dir,
    )

    with output_path.open("w", encoding="utf-8") as writer:
        for pair in rewritten:
            writer.write(json.dumps(pair.to_json_dict(), ensure_ascii=False) + "\n")

    print(f"Read {len(anchors)} anchors from {input_path}")
    print(f"Materialized {len(rewritten)} pairs to {output_path} (audio under {audio_dir})")
    if unmatched:
        print(f"WARNING: {unmatched} pairs had clips not found in decoded parquet and were skipped")
```

Note: `output_dir`, `output_path`, `input_path`, `anchors` are already defined earlier in `main()`. The old inline `with output_path.open(...)` write loop is fully replaced by the block above — delete the original one.

- [ ] **Step 3: Verify the script imports and `--help` works**

Run: `python scripts/prepare_stage2_libriphrase.py --help`
Expected: help text including `--decoded-parquet-root`. No import errors.

- [ ] **Step 4: Run the full test suite for prep + pairs**

Run: `python -m pytest tests/test_stage2_pairs.py tests/test_prepare_stage2_libriphrase.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_stage2_libriphrase.py
git commit -m "feat(stage2): prep main() extracts decoded parquet audio to npy"
```

---

### Task 5: Switch train Dataset to `np.load`

**Files:**
- Modify: `scripts/train_stage2_qbyt.py:68-101`

**Interfaces:**
- Consumes: JSONL records with `wav_path` = `audio/<audio_rel>.npy` (relative to `<processed_root>/stage2_qbyt`) and `sample_rate`.
- Produces: training that reads `.npy` waveforms.

- [ ] **Step 1: Repoint `resolve_wav_path` base dir**

In `train()`, the current code (lines ~68-74):

```python
    libriphrase_root = Path(paths["libriphrase100_root"])

    def resolve_wav_path(raw_path: str) -> str:
        path = Path(raw_path)
        if path.is_absolute():
            return str(path)
        return str(libriphrase_root / path)
```

Replace with:

```python
    stage2_dir = processed_root / "stage2_qbyt"

    def resolve_wav_path(raw_path: str) -> str:
        path = Path(raw_path)
        if path.is_absolute():
            return str(path)
        return str(stage2_dir / path)
```

(`processed_root` is already defined just above as `Path(paths["processed_root"])`.)

- [ ] **Step 2: Replace `torchaudio.load` with `np.load` in `__getitem__`**

The current `__getitem__` (lines ~83-101):

```python
        def __getitem__(self, index: int) -> dict:
            record = self.records[index]
            wav_path = resolve_wav_path(record["wav_path"])
            waveform, sr = torchaudio.load(wav_path)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != sample_rate:
                waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
            feat = kaldi.fbank(
                ...
            )
```

Replace the load/resample lines with:

```python
        def __getitem__(self, index: int) -> dict:
            record = self.records[index]
            wav_path = resolve_wav_path(record["wav_path"])
            audio = np.load(wav_path).astype("float32")
            waveform = torch.from_numpy(audio).reshape(1, -1)
            feat = kaldi.fbank(
                waveform,
                num_mel_bins=num_mel_bins,
                frame_length=25,
                frame_shift=10,
                dither=0.1,
                sample_frequency=sample_rate,
            )
            anchor = torch.tensor(vocab.encode(record["anchor_phonemes"]), dtype=torch.long)
            label = torch.tensor(float(record["label"]), dtype=torch.float32)
            return {"feat": feat, "anchor": anchor, "label": label}
```

The decoded audio is already 16 kHz mono and float in [-1, 1] — identical range to what `torchaudio.load` returned — so channel-merge and resample are removed and `kaldi.fbank` behavior is unchanged.

- [ ] **Step 3: Add `numpy` import**

In the `try:` import block inside `train()` (alongside `import torch`, `import torchaudio`), add:

```python
        import numpy as np
```

`torchaudio` import stays — `torchaudio.compliance.kaldi` is still used for fbank.

- [ ] **Step 4: Smoke-check the script parses**

Run: `python -c "import ast; ast.parse(open('scripts/train_stage2_qbyt.py').read()); print('ok')"`
Expected: `ok` (no torch needed for a parse check).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_stage2_qbyt.py
git commit -m "feat(stage2): train Dataset loads npy waveforms instead of wav"
```

---

### Task 6: Update README Stage II docs

**Files:**
- Modify: `README.md` (section 3.2 around lines 191-249, and the Stage II prep/train command blocks around 373-401)

**Interfaces:** none (docs).

- [ ] **Step 1: Update the "expects a parquet containing" subsection**

In section 3.2, after the download instructions, replace the "The Stage II preparation script currently expects..." block to explain the two inputs:

```markdown
The Stage II preparation reads two kinds of files from the download:

1. An **aggregated** parquet for phrase metadata — at least columns:

   ```text
   ngram
   clips
   ```

   and optionally `ngram_g2p` (skips on-the-fly G2P if present). Recommended:
   `aggregated_segments_with_g2p.parquet`.

2. The **decoded** audio shards `LP-100-decoded-*.parquet` (columns
   `audio_rel`, `audio`, `sampling_rate`, ...). The script extracts only the
   clips actually referenced by the pairs and writes them as float32 `.npy`
   files under `<processed_root>/stage2_qbyt/audio/`. No loose `.wav` files
   are needed.
```

- [ ] **Step 2: Update the prep command block**

In the Stage II prep example (around line 382-401), show both inputs:

```bash
LP_AGG=/data/dma-kws/raw/LibriPhrase-100/aggregated_segments_with_g2p.parquet

python3 scripts/prepare_stage2_libriphrase.py \
  --config configs/demo_librispeech100.yaml \
  --input-parquet "$LP_AGG" \
  --decoded-parquet-root /data/dma-kws/raw/LibriPhrase-100 \
  --limit-anchors 50
```

Add a sentence: `--decoded-parquet-root` defaults to `paths.libriphrase100_root`, so it can be omitted if the decoded shards live there.

- [ ] **Step 3: Note the train input change**

Where the doc describes Stage II training inputs, add: the trainer now loads `.npy` waveforms produced by prep (resolved relative to `<processed_root>/stage2_qbyt`); it no longer reads `.wav` from `libriphrase100_root`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: Stage II reads decoded parquet, materializes npy audio"
```

---

## Self-Review

**Spec coverage:**
- Data flow / `clips`→`audio_rel` mapping → Task 1 (`clip_to_audio_rel`).
- Streaming extraction of only referenced clips → Task 2 (`iter_decoded_audio_rows`) + Task 3 (`materialize_pairs` collects `needed` first).
- `.npy` float32 16 kHz under `<processed_root>/stage2_qbyt/audio/<audio_rel>.npy` → Task 3.
- `--decoded-parquet-root` arg, default to `libriphrase100_root` → Tasks 3-4.
- `wav_path` rewritten to `.npy`, `sample_rate` field → Tasks 1, 3.
- Unmatched clips warned + counted → Tasks 3-4.
- Train `np.load`, drop channel/resample, keep `kaldi.fbank`, repoint base dir → Task 5.
- Unit tests (pairs matching, prep smoke) → Tasks 1-3. README → Task 6.
- No new deps → all tasks use numpy/pandas/pyarrow/torch already present.

**Placeholder scan:** No TBD/TODO; every code step shows full code.

**Type consistency:** `clip_to_audio_rel(str)->str`, `iter_decoded_audio_rows(paths, set, *, read_parquet)->Iterator[(str,audio,int)]`, `materialize_pairs(pairs,*,decoded_parquet_paths,audio_dir,read_parquet)->(list[PairRecord],int)`, `PairRecord(...,sample_rate:int)` — names and signatures match across Tasks 1-5. The `_FakeDF.iterrows()` matches the real pandas DataFrame interface used in `iter_decoded_audio_rows`.

**Note on Python 3.9:** `set[str]`/`tuple[...]`/`list[PairRecord]` annotations are safe because every touched module/script uses `from __future__ import annotations` (verify in `train_stage2_qbyt.py` — it has it at line 4; `pairs.py` line 3; prep script line 4). Runtime `set(...)` builtins are used, not subscripted generics.
