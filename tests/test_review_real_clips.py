import csv
import io
import json
import sqlite3
import threading
import urllib.error
import urllib.request
import wave
from pathlib import Path

import pytest

from scripts.review_real_clips import (
    VerdictStore,
    _parse_speaker_aliases,
    apply_verdicts,
    build_index,
    build_reviewed_view,
    load_clips,
    ReviewServer,
)


def _wav_bytes(frames: int = 160) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x01" * frames)
    return buffer.getvalue()


def _write_corpus(root: Path) -> Path:
    """Create a tiny corpus with one original + one augmentation per phrase."""

    speakers = [(1, "alice"), (2, "bob")]
    recordings = [
        # (speaker_id, category, wake_word, is_original, aug_type, group, file_path)
        (1, "positive", "Hey Eva", 1, "orig", "g1", "record_data/positive/Hey_Eva/alice/orig_A.wav"),
        (1, "positive", "Hey Eva", 0, "noise_mix", "g1", "record_data/positive/Hey_Eva/alice/aug_A.wav"),
        (2, "near_negative", "Hey Ava", 1, "orig", "g2", "record_data/near_negative/Hey_Ava/bob/orig_B.wav"),
        (2, "near_negative", "Hey Ava", 0, "volume", "g2", "record_data/near_negative/Hey_Ava/bob/aug_B.wav"),
    ]
    connection = sqlite3.connect(root / "app.db")
    connection.executescript(
        """
        CREATE TABLE speakers (id INTEGER PRIMARY KEY, name TEXT, gender TEXT, age TEXT,
                               env TEXT, mic_dist TEXT, created_at TEXT);
        CREATE TABLE recordings (id INTEGER PRIMARY KEY, speaker_id INTEGER, group_id TEXT,
                                 category TEXT, wake_word TEXT, is_original INTEGER,
                                 aug_type TEXT, aug_params TEXT, orig_path TEXT, file_path TEXT,
                                 duration REAL, peak_dbfs REAL, rms_dbfs REAL, recorded_at TEXT);
        """
    )
    connection.executemany("INSERT INTO speakers VALUES (?, ?, '', '', '', '', '')", speakers)
    originals = {group: relative for _, _, _, is_original, _, group, relative in recordings if is_original}
    for speaker_id, category, wake_word, is_original, aug_type, group, relative in recordings:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_wav_bytes())
        original = relative if is_original else originals[group]
        connection.execute(
            "INSERT INTO recordings (speaker_id, group_id, category, wake_word, is_original,"
            " aug_type, aug_params, orig_path, file_path, duration) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (speaker_id, group, category, wake_word, is_original, aug_type, "", original,
             "data/" + relative, 2.0),
        )
    connection.commit()
    connection.close()
    return root / "app.db"


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "hey_eva_real"
    root.mkdir()
    _write_corpus(root)
    index_path = root / "index.jsonl"
    build_index(root, root / "app.db", index_path)
    clips = load_clips(index_path, root)
    store = VerdictStore(root / "reviews.jsonl")
    store.load()
    server = ReviewServer(("127.0.0.1", 0), root=root, clips=clips, store=store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base, root, index_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get_json(url: str):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_build_index_orders_originals_first(corpus):
    _base, _root, index_path = corpus
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    assert all(row["is_original"] == 1 for row in rows[:2])
    assert rows[0]["category"] == "positive"
    assert {row["key"].split("/")[0] for row in rows} == {"record_data"}
    aug = next(row for row in rows if row["is_original"] == 0 and row["category"] == "positive")
    assert aug["orig_key"] == "record_data/positive/Hey_Eva/alice/orig_A.wav"
    assert aug["speaker"] == "alice"


def test_page_has_three_verdict_controls(corpus):
    base, _root, _index = corpus
    with urllib.request.urlopen(base + "/", timeout=10) as response:
        page = response.read().decode("utf-8")
    assert response.status == 200
    assert '<audio id="audio"' in page
    assert "说的是别的词" in page
    assert "不可用" in page
    assert "/api/verdict" in page


def test_clips_api_lists_metadata(corpus):
    base, _root, _index = corpus
    status, payload = _get_json(base + "/api/clips")
    assert status == 200
    assert len(payload["clips"]) == 4
    first = payload["clips"][0]
    assert first["is_original"] == 1
    assert first["text"] == "Hey Eva"
    assert first["audio_url"].startswith("/audio/")
    assert payload["verdicts"] == {}


def test_audio_route_returns_file_bytes(corpus):
    base, root, _index = corpus
    relative = "record_data/positive/Hey_Eva/alice/orig_A.wav"
    with urllib.request.urlopen(base + "/audio/" + relative, timeout=10) as response:
        body = response.read()
        assert response.status == 200
        assert response.headers["Content-Type"] == "audio/wav"
    assert body == (root / relative).read_bytes()


def test_audio_route_rejects_path_traversal(corpus):
    base, root, _index = corpus
    (root.parent / "secret.txt").write_text("nope", encoding="utf-8")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(base + "/audio/../secret.txt", timeout=10)
    assert error.value.code == 404


def test_verdict_post_persists_note(corpus):
    base, root, _index = corpus
    key = "record_data/positive/Hey_Eva/alice/orig_A.wav"
    status, payload = _post_json(base + "/api/verdict", {"key": key, "verdict": "wrong", "note": "Hey Ava"})
    assert status == 200
    assert payload["verdicts"][key]["verdict"] == "wrong"
    assert payload["verdicts"][key]["note"] == "Hey Ava"
    rows = (root / "reviews.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(rows[-1])["note"] == "Hey Ava"


def test_invalid_verdict_is_rejected(corpus):
    base, _root, _index = corpus
    with pytest.raises(urllib.error.HTTPError) as error:
        _post_json(base + "/api/verdict", {"key": "x", "verdict": "maybe"})
    assert error.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as error:
        _post_json(base + "/api/verdict", {"verdict": "ok"})
    assert error.value.code == 400


def _apply(tmp_path, corpus, entries, **kwargs):
    base, root, index_path = corpus
    verdict_path = root / "reviews.jsonl"
    with verdict_path.open("w", encoding="utf-8") as handle:
        for row in entries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output = tmp_path / "reviewed.csv"
    stats = apply_verdicts(
        index_path=index_path,
        verdict_path=verdict_path,
        output_path=output,
        **kwargs,
    )
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return stats, rows, output


OK = "record_data/positive/Hey_Eva/alice/orig_A.wav"
OK_AUG = "record_data/positive/Hey_Eva/alice/aug_A.wav"
NEG = "record_data/near_negative/Hey_Ava/bob/orig_B.wav"
NEG_AUG = "record_data/near_negative/Hey_Ava/bob/aug_B.wav"


def test_apply_drops_bad_and_keeps_rest(tmp_path, corpus):
    stats, rows, _output = _apply(tmp_path, corpus, [{"key": NEG, "verdict": "bad", "note": ""}])
    kept = {row["audio_path"] for row in rows}
    assert OK in kept and OK_AUG in kept
    assert NEG not in kept  # own verdict
    assert NEG_AUG not in kept  # inherited from its original
    assert stats["dropped_bad"] == 2
    assert stats["inherited_verdicts"] == 1
    assert stats["kept"] == 2


def test_apply_relabels_wrong_with_note(tmp_path, corpus):
    stats, rows, _output = _apply(
        tmp_path, corpus, [{"key": OK, "verdict": "wrong", "note": "Hey Ava"}]
    )
    by_key = {row["audio_path"]: row for row in rows}
    assert by_key[OK]["text"] == "Hey Ava"
    assert by_key[OK]["label"] == "0"
    assert by_key[OK_AUG]["text"] == "Hey Ava"  # inherited
    assert by_key[OK_AUG]["label"] == "0"
    assert by_key[OK_AUG]["verdict_source"] == "inherited"
    assert stats["kept_positive"] == 0
    assert stats["kept_negative"] == 4


def test_apply_wrong_keyword_becomes_positive(tmp_path, corpus):
    _stats, rows, _output = _apply(
        tmp_path, corpus, [{"key": NEG, "verdict": "wrong", "note": "hey eva!"}]
    )
    by_key = {row["audio_path"]: row for row in rows}
    assert by_key[NEG]["text"] == "Hey Eva"  # canonicalised
    assert by_key[NEG]["label"] == "1"
    assert by_key[NEG_AUG]["label"] == "1"


def test_apply_drops_wrong_without_note(tmp_path, corpus):
    stats, rows, _output = _apply(tmp_path, corpus, [{"key": OK, "verdict": "wrong", "note": ""}])
    kept = {row["audio_path"] for row in rows}
    assert OK not in kept and OK_AUG not in kept
    assert stats["dropped_wrong_without_note"] == 2


def test_apply_writes_speaker_disjoint_split(tmp_path, corpus):
    stats, rows, _output = _apply(
        tmp_path, corpus,
        [{"key": OK, "verdict": "ok", "note": ""}],
        eval_speakers=["bob"],
    )
    by_speaker: dict[str, set[str]] = {}
    for row in rows:
        by_speaker.setdefault(row["speaker_id"], set()).add(row["split"])
    assert by_speaker == {"alice": {"train"}, "bob": {"eval"}}
    assert stats["eval_speakers"] == ["bob"]
    with (tmp_path / "reviewed.csv").open("r", encoding="utf-8") as handle:
        assert handle.readline().strip().split(",")[:5] == ["audio_path", "text", "label", "phase", "split"]


def test_apply_without_eval_speakers_omits_split_column(tmp_path, corpus):
    _stats, _rows, output = _apply(tmp_path, corpus, [])
    with output.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert "split" not in header


def test_load_clips_skips_missing_audio(corpus):
    _base, root, index_path = corpus
    (root / NEG).unlink()
    clips = load_clips(index_path, root)
    assert NEG not in {clip["key"] for clip in clips}
    assert len(clips) == 3


def test_verdicts_reload_last_wins(tmp_path):
    path = tmp_path / "reviews.jsonl"
    path.write_text(
        json.dumps({"key": "a", "verdict": "ok", "note": "", "at": 1}) + "\n"
        + json.dumps({"key": "a", "verdict": "wrong", "note": "Hey Eve", "at": 2}) + "\n",
        encoding="utf-8",
    )
    store = VerdictStore(path)
    store.load()
    assert store.entries["a"]["verdict"] == "wrong"
    assert store.entries["a"]["note"] == "Hey Eve"


def _verdicts(root: Path, entries: list[dict]) -> Path:
    path = root / "reviews.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in entries),
        encoding="utf-8",
    )
    return path


def test_build_reviewed_view_moves_relabelled_clip(tmp_path, corpus):
    _base, root, index_path = corpus
    verdict_path = _verdicts(root, [{"key": NEG, "verdict": "wrong", "note": "Hey Eva"}])
    target = tmp_path / "view"
    stats = build_reviewed_view(
        index_path=index_path,
        verdict_path=verdict_path,
        source_root=root,
        target_root=target,
    )
    assert stats["kept"] == 4
    assert stats["kept_positive"] == 4
    assert stats["kept_negative"] == 0
    # The near-negative pair was actually the wake word, so it moved (aug included).
    assert (target / "record_data/positive/Hey_Eva/bob/orig_B.wav").is_file()
    assert (target / "record_data/positive/Hey_Eva/bob/aug_B.wav").is_file()
    assert not (target / "record_data/near_negative").exists()
    rows = list(csv.DictReader((target / "manifests/real_reviewed.csv").open(encoding="utf-8")))
    assert len(rows) == 4
    assert {row["text"] for row in rows} == {"Hey Eva"}
    assert {row["label"] for row in rows} == {"1"}
    absolute = list(csv.DictReader((target / "manifests/real_reviewed_abs.csv").open(encoding="utf-8")))
    assert all(Path(row["audio_path"]).is_file() for row in absolute)
    assert (target / "README.md").is_file()
    assert (target / "review_summary.json").is_file()
    # The pre-review backup is untouched.
    assert (root / NEG).is_file()


def test_build_reviewed_view_drops_bad_and_its_augmentations(tmp_path, corpus):
    _base, root, index_path = corpus
    verdict_path = _verdicts(root, [{"key": OK, "verdict": "bad"}])
    target = tmp_path / "view"
    stats = build_reviewed_view(
        index_path=index_path,
        verdict_path=verdict_path,
        source_root=root,
        target_root=target,
    )
    assert stats["kept"] == 2
    assert stats["dropped_bad"] == 2
    assert not (target / "record_data/positive/Hey_Eva/alice").exists()
    assert (target / "record_data/near_negative/Hey_Ava/bob/orig_B.wav").is_file()
    dropped = list(csv.DictReader((target / "manifests/dropped_clips.csv").open(encoding="utf-8")))
    assert len(dropped) == 2
    assert {row["reason"] for row in dropped} == {"bad"}


def test_build_reviewed_view_writes_split_for_eval_speakers(tmp_path, corpus):
    _base, root, index_path = corpus
    verdict_path = _verdicts(root, [{"key": OK, "verdict": "ok"}])
    target = tmp_path / "view"
    stats = build_reviewed_view(
        index_path=index_path,
        verdict_path=verdict_path,
        source_root=root,
        target_root=target,
        eval_speakers=["bob"],
    )
    assert stats["eval_speakers"] == ["bob"]
    rows = list(csv.DictReader((target / "manifests/real_reviewed.csv").open(encoding="utf-8")))
    by_speaker: dict[str, set[str]] = {}
    for row in rows:
        by_speaker.setdefault(row["speaker_id"], set()).add(row["split"])
    assert by_speaker == {"alice": {"train"}, "bob": {"eval"}}


def _alias_index(root: Path, index_path: Path, speakers=("解震", "解震2")) -> Path:
    """Two speaker directories that a caller wants to merge into one person."""

    rows = []
    for speaker in speakers:
        relative = (
            f"record_data/positive/Hey_Eva/{speaker}_male_26_35/"
            f"orig__Hey_Eva__{speaker}_male_26_35__1.wav"
        )
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_wav_bytes())
        rows.append(
            {
                "key": relative,
                "audio_path": relative,
                "text": "Hey Eva",
                "label": 1,
                "category": "positive",
                "wake_word": "Hey Eva",
                "speaker": speaker,
                "is_original": 1,
                "aug_type": "orig",
                "orig_key": relative,
                "duration": 2.0,
            }
        )
    index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return index_path


def test_parse_speaker_aliases(tmp_path):
    assert _parse_speaker_aliases("解震2=解震, a = b") == {"解震2": "解震", "a": "b"}
    assert _parse_speaker_aliases("") == {}
    with pytest.raises(SystemExit):
        _parse_speaker_aliases("解震2")
    with pytest.raises(SystemExit):
        _parse_speaker_aliases("=x")


def test_build_reviewed_view_merges_aliased_speaker_dirs(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    index_path = _alias_index(source, tmp_path / "index.jsonl")
    target = tmp_path / "view"
    stats = build_reviewed_view(
        index_path=index_path,
        verdict_path=tmp_path / "reviews.jsonl",
        source_root=source,
        target_root=target,
        speaker_aliases={"解震2": "解震"},
    )
    merged = target / "record_data/positive/Hey_Eva/解震_male_26_35"
    assert sorted(path.name for path in merged.iterdir()) == [
        "orig__Hey_Eva__解震2_male_26_35__1.wav",
        "orig__Hey_Eva__解震_male_26_35__1.wav",
    ]
    assert not (target / "record_data/positive/Hey_Eva/解震2_male_26_35").exists()
    rows = list(csv.DictReader((target / "manifests/real_reviewed.csv").open(encoding="utf-8")))
    assert {row["speaker_id"] for row in rows} == {"解震"}
    assert {row["source_speaker"] for row in rows} == {"解震", "解震2"}
    assert stats["speaker_aliases"] == ["解震2 -> 解震"]


def test_build_reviewed_view_alias_applies_before_eval_split(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    index_path = _alias_index(source, tmp_path / "index.jsonl")
    target = tmp_path / "view"
    build_reviewed_view(
        index_path=index_path,
        verdict_path=tmp_path / "reviews.jsonl",
        source_root=source,
        target_root=target,
        eval_speakers=["解震"],
        speaker_aliases={"解震2": "解震"},
    )
    rows = list(csv.DictReader((target / "manifests/real_reviewed.csv").open(encoding="utf-8")))
    assert {row["speaker_id"] for row in rows} == {"解震"}
    assert {row["split"] for row in rows} == {"eval"}


def test_build_reviewed_view_refuses_existing_target(tmp_path, corpus):
    _base, root, index_path = corpus
    target = tmp_path / "view"
    (target / "record_data").mkdir(parents=True)
    with pytest.raises(SystemExit):
        build_reviewed_view(
            index_path=index_path,
            verdict_path=root / "reviews.jsonl",
            source_root=root,
            target_root=target,
        )
