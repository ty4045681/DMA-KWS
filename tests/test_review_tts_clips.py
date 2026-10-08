import io
import json
import threading
import urllib.error
import urllib.request
import wave
from pathlib import Path

import pytest

from scripts.review_tts_clips import ReviewServer, VerdictStore, load_clips


def _wav_bytes(frames: int = 160) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x01" * frames)
    return buffer.getvalue()


def _write_index(root: Path, clips: int = 2) -> Path:
    rows = []
    for index in range(clips):
        relative = f"elevenlabs_output/voice-{index}/clip_{index}.wav"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = _wav_bytes(160 + index)
        target.write_bytes(payload)
        rows.append(
            {
                "key": f"key-{index}",
                "provider": "elevenlabs",
                "voice_id": f"voice-{index}",
                "voice_name": f"Voice {index}",
                "label": 1 if index == 0 else 0,
                "text": "Hey Eva" if index == 0 else "Hey Ava",
                "phonemes": "HH EY1 IY1 V AH0",
                "audio_path": relative,
                "bytes": len(payload),
            }
        )
    index_path = root / "index.jsonl"
    index_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return index_path


@pytest.fixture
def review(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    index_path = _write_index(root)
    clips = load_clips(index_path, root)
    store = VerdictStore(root / "reviews.jsonl")
    store.load()
    server = ReviewServer(("127.0.0.1", 0), root=root, clips=clips, store=store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base, root
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


def test_page_has_player_controls(review):
    base, _root = review
    with urllib.request.urlopen(base + "/", timeout=10) as response:
        page = response.read().decode("utf-8")
    assert response.status == 200
    assert '<audio id="audio"' in page
    assert "/api/verdict" in page


def test_clips_api_lists_playable_rows(review):
    base, _root = review
    status, payload = _get_json(base + "/api/clips")
    assert status == 200
    assert len(payload["clips"]) == 2
    first = payload["clips"][0]
    assert first["text"] == "Hey Eva"
    assert first["audio_url"].startswith("/audio/")
    assert payload["verdicts"] == {}


def test_audio_route_returns_the_file_bytes(review):
    base, root = review
    relative = "elevenlabs_output/voice-0/clip_0.wav"
    with urllib.request.urlopen(base + "/audio/" + relative, timeout=10) as response:
        body = response.read()
        assert response.status == 200
        assert response.headers["Content-Type"] == "audio/wav"
    assert body == (root / relative).read_bytes()


def test_audio_route_rejects_path_traversal(review):
    base, root = review
    (root.parent / "secret.txt").write_text("nope", encoding="utf-8")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(base + "/audio/../secret.txt", timeout=10)
    assert error.value.code == 404


def test_missing_audio_route_returns_404(review):
    base, _root = review
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(base + "/audio/elevenlabs_output/nope.wav", timeout=10)
    assert error.value.code == 404


def test_verdict_post_persists_and_returns_state(review):
    base, root = review
    status, payload = _post_json(base + "/api/verdict", {"key": "key-1", "verdict": "bad"})
    assert status == 200
    assert payload["verdicts"]["key-1"]["verdict"] == "bad"
    lines = (root / "reviews.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["verdict"] == "bad"

    status, payload = _post_json(base + "/api/verdict", {"key": "key-1", "verdict": ""})
    assert payload["verdicts"]["key-1"]["verdict"] == ""


def test_invalid_verdict_is_rejected(review):
    base, _root = review
    with pytest.raises(urllib.error.HTTPError) as error:
        _post_json(base + "/api/verdict", {"key": "key-0", "verdict": "maybe"})
    assert error.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as error:
        _post_json(base + "/api/verdict", {"verdict": "ok"})
    assert error.value.code == 400


def test_verdicts_reload_from_disk(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    _write_index(root)
    path = root / "reviews.jsonl"
    path.write_text(
        json.dumps({"key": "key-0", "verdict": "ok", "note": "", "at": 1}) + "\n"
        + json.dumps({"key": "key-0", "verdict": "bad", "note": "", "at": 3}) + "\n",
        encoding="utf-8",
    )
    store = VerdictStore(path)
    store.load()
    assert store.entries["key-0"]["verdict"] == "bad"


def test_load_clips_skips_missing_audio(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    index_path = _write_index(root)
    (root / "elevenlabs_output/voice-1/clip_1.wav").unlink()
    clips = load_clips(index_path, root)
    assert [clip["key"] for clip in clips] == ["key-0"]
