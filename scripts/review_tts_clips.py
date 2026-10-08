#!/usr/bin/env python3
"""Serve a local page for listening to generated clips and flagging bad ones.

The page reads <run-root>/index.jsonl, plays the audio under <run-root>, and
appends every verdict to <run-root>/reviews.jsonl. It binds 127.0.0.1 by
default.

    python3 scripts/review_tts_clips.py --run-root outputs/tts/hey_eva --port 8765

Keyboard: space plays or pauses, 1 passes the clip, 2 flags it, 0 clears the
verdict, and the arrow keys move between clips.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERDICTS = ("ok", "bad", "")


PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hey Eva 语料复核</title>
<style>
:root{--bg:#101319;--panel:#181d27;--line:#2a3140;--fg:#e7eaf1;--muted:#93a0b4;
--ok:#2fbf71;--bad:#e5484d;--accent:#4c8dff}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;display:flex;flex-direction:column;background:var(--bg);color:var(--fg);
font:14px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans SC",sans-serif}
header{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:10px 18px;
border-bottom:1px solid var(--line);background:var(--panel)}
#progress{font-variant-numeric:tabular-nums}
#progress b{color:var(--accent)}
button{background:#232a38;color:var(--fg);border:1px solid var(--line);border-radius:8px;
padding:6px 12px;cursor:pointer;font:inherit}
button:hover{border-color:var(--accent)}
button.active{border-color:var(--accent);color:var(--accent)}
button.ok{border-color:var(--ok)}
button.bad{border-color:var(--bad)}
main{flex:1;display:grid;grid-template-columns:minmax(0,1fr) 340px;min-height:0}
section{padding:22px;display:flex;flex-direction:column;gap:14px;min-width:0}
#clipText{font-size:34px;font-weight:600;letter-spacing:.5px}
#clipMeta{color:var(--muted);display:flex;gap:16px;flex-wrap:wrap}
#clipMeta code{color:var(--fg)}
audio{width:100%}
#actions{display:flex;gap:10px;flex-wrap:wrap}
#hint{color:var(--muted);font-size:12px}
aside{border-left:1px solid var(--line);overflow-y:auto;background:var(--panel)}
.row{display:flex;gap:10px;align-items:center;padding:8px 14px;cursor:pointer;
border-bottom:1px solid rgba(255,255,255,.04)}
.row:hover{background:rgba(255,255,255,.04)}
.row.current{background:rgba(76,141,255,.16)}
.row .text{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row .voice{color:var(--muted);font-size:12px;max-width:110px;overflow:hidden;
text-overflow:ellipsis;white-space:nowrap}
.dot{width:9px;height:9px;border-radius:50%;background:#3a4356;flex:none}
.dot.ok{background:var(--ok)}
.dot.bad{background:var(--bad)}
.tag{font-size:11px;color:var(--muted);border:1px solid var(--line);border-radius:6px;padding:0 6px}
</style>
</head>
<body>
<header>
  <div id="progress"></div>
  <div id="filters"></div>
  <label><input type="checkbox" id="autonext" checked> 自动下一条</label>
</header>
<main>
  <section>
    <div id="clipText">加载中…</div>
    <div id="clipMeta"></div>
    <audio id="audio" controls preload="auto"></audio>
    <div id="actions">
      <button id="ok" class="ok">通过 (1)</button>
      <button id="bad" class="bad">有问题 (2)</button>
      <button id="clear">清除 (0)</button>
      <button id="prev">上一条 (←)</button>
      <button id="next">下一条 (→)</button>
    </div>
    <div id="hint">空格播放或暂停。判定结果立即存到 reviews.jsonl，刷新页面不会丢。</div>
  </section>
  <aside id="list"></aside>
</main>
<script>
(function () {
  "use strict";
  var state = { clips: [], verdicts: {}, index: 0, filter: "all", autonext: true, played: false };
  var audio = document.getElementById("audio");
  var listEl = document.getElementById("list");
  var progressEl = document.getElementById("progress");
  var filtersEl = document.getElementById("filters");
  var textEl = document.getElementById("clipText");
  var metaEl = document.getElementById("clipMeta");
  var prefetch = null;
  var rowEls = [];

  var FILTERS = [["all", "全部"], ["pending", "未复核"], ["ok", "通过"], ["bad", "有问题"]];

  function esc(value) {
    return String(value).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function verdictOf(key) {
    var entry = state.verdicts[key];
    return entry ? entry.verdict : "";
  }

  function passes(clip) {
    var verdict = verdictOf(clip.key);
    if (state.filter === "all") { return true; }
    if (state.filter === "pending") { return verdict === ""; }
    return verdict === state.filter;
  }

  function visible() {
    var out = [];
    for (var i = 0; i < state.clips.length; i += 1) {
      if (passes(state.clips[i])) { out.push(i); }
    }
    return out;
  }

  function counts() {
    var ok = 0, bad = 0;
    for (var key in state.verdicts) {
      if (state.verdicts[key].verdict === "ok") { ok += 1; }
      if (state.verdicts[key].verdict === "bad") { bad += 1; }
    }
    return { ok: ok, bad: bad, total: state.clips.length };
  }

  function paintProgress() {
    var c = counts();
    progressEl.innerHTML = "已复核 <b>" + (c.ok + c.bad) + "</b> / " + c.total +
      " · 通过 " + c.ok + " · 有问题 <b>" + c.bad + "</b>";
  }

  function paintRow(el, clip) {
    var verdict = verdictOf(clip.key);
    el.querySelector(".dot").className = "dot " + verdict;
    el.classList.toggle("current", state.clips[state.index] === clip);
  }

  function paintRows() {
    for (var i = 0; i < rowEls.length; i += 1) {
      var entry = rowEls[i];
      var show = passes(entry.clip);
      entry.el.style.display = show ? "" : "none";
      paintRow(entry.el, entry.clip);
    }
  }

  function renderList() {
    var fragment = document.createDocumentFragment();
    rowEls = [];
    for (var i = 0; i < state.clips.length; i += 1) {
      var clip = state.clips[i];
      var row = document.createElement("div");
      row.className = "row";
      row.innerHTML = '<span class="dot"></span><span class="text">' + esc(clip.text) +
        '</span><span class="tag">' + (clip.label === 1 ? "正" : "负") +
        '</span><span class="voice">' + esc(clip.voice_name) + "</span>";
      row.dataset.index = String(i);
      row.addEventListener("click", function (event) {
        select(Number(event.currentTarget.dataset.index));
      });
      fragment.appendChild(row);
      rowEls.push({ el: row, clip: clip, index: i });
    }
    listEl.innerHTML = "";
    listEl.appendChild(fragment);
    paintRows();
  }

  function renderFilters() {
    filtersEl.innerHTML = "";
    FILTERS.forEach(function (pair) {
      var button = document.createElement("button");
      button.textContent = pair[1];
      button.className = state.filter === pair[0] ? "active" : "";
      button.addEventListener("click", function () {
        state.filter = pair[0];
        renderFilters();
        paintRows();
        var order = visible();
        if (order.indexOf(state.index) < 0 && order.length) { select(order[0]); }
      });
      filtersEl.appendChild(button);
    });
  }

  function select(index) {
    if (index < 0 || index >= state.clips.length) { return; }
    state.index = index;
    var clip = state.clips[index];
    textEl.textContent = clip.text;
    metaEl.innerHTML = "<span>" + esc(clip.provider) + " · " + esc(clip.voice_name) +
      "</span><span>音素 <code>" + esc(clip.phonemes) + "</code></span><span>" +
      Math.round(clip.bytes / 1024) + " KB</span><span>" + esc(clip.key) + "</span>";
    audio.src = clip.audio_url;
    audio.load();
    paintRows();
    if (state.autonext && state.played) { audio.play().catch(function () {}); }
    else if (state.autonext) { audio.play().then(function () { state.played = true; }).catch(function () {}); }
    preloadNext(index);
  }

  function preloadNext(index) {
    var order = visible();
    var position = order.indexOf(index);
    if (position < 0 || position + 1 >= order.length) { return; }
    var next = state.clips[order[position + 1]];
    if (!prefetch) { prefetch = new Audio(); prefetch.preload = "auto"; }
    if (prefetch.src.indexOf(next.audio_url) < 0) { prefetch.src = next.audio_url; }
  }

  function step(delta) {
    var order = visible();
    var position = order.indexOf(state.index);
    if (position < 0) { position = 0; }
    var target = position + delta;
    if (target < 0 || target >= order.length) { return; }
    select(order[target]);
  }

  function send(key, verdict) {
    return fetch("/api/verdict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: key, verdict: verdict })
    }).then(function (response) { return response.json(); }).then(function (payload) {
      state.verdicts = payload.verdicts;
      paintProgress();
      paintRows();
    });
  }

  function judge(verdict) {
    var clip = state.clips[state.index];
    if (!clip) { return; }
    send(clip.key, verdictOf(clip.key) === verdict ? "" : verdict).then(function () {
      if (verdict) { step(1); }
    });
  }

  document.getElementById("ok").addEventListener("click", function () { judge("ok"); });
  document.getElementById("bad").addEventListener("click", function () { judge("bad"); });
  document.getElementById("clear").addEventListener("click", function () {
    var clip = state.clips[state.index];
    if (clip) { send(clip.key, ""); }
  });
  document.getElementById("prev").addEventListener("click", function () { step(-1); });
  document.getElementById("next").addEventListener("click", function () { step(1); });
  document.getElementById("autonext").addEventListener("change", function (event) {
    state.autonext = event.target.checked;
  });
  audio.addEventListener("play", function () { state.played = true; });

  document.addEventListener("keydown", function (event) {
    if (event.target.tagName === "INPUT") { return; }
    if (event.code === "Space") { event.preventDefault(); if (audio.paused) { audio.play(); } else { audio.pause(); } }
    else if (event.key === "1") { judge("ok"); }
    else if (event.key === "2") { judge("bad"); }
    else if (event.key === "0") { var clip = state.clips[state.index]; if (clip) { send(clip.key, ""); } }
    else if (event.key === "ArrowLeft") { step(-1); }
    else if (event.key === "ArrowRight") { step(1); }
  });

  fetch("/api/clips").then(function (response) { return response.json(); }).then(function (payload) {
    state.clips = payload.clips;
    state.verdicts = payload.verdicts || {};
    renderFilters();
    renderList();
    paintProgress();
    var order = visible();
    if (order.length) { select(order[0]); } else { textEl.textContent = "这个筛选下没有片段"; }
  });
})();
</script>
</body>
</html>
"""


def read_index(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"index not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_clips(index_path: Path, root: Path) -> list[dict[str, Any]]:
    """Return the index rows whose audio still exists under root."""

    clips: list[dict[str, Any]] = []
    for row in read_index(index_path):
        relative = str(row.get("audio_path") or "")
        if not relative or not (root / relative).is_file():
            continue
        clips.append(
            {
                "key": str(row.get("key") or ""),
                "provider": str(row.get("provider") or ""),
                "voice_id": str(row.get("voice_id") or ""),
                "voice_name": str(row.get("voice_name") or ""),
                "text": str(row.get("text") or ""),
                "label": int(row.get("label") or 0),
                "phonemes": str(row.get("phonemes") or ""),
                "bytes": int(row.get("bytes") or 0),
                # The content hash goes in the URL, so a regenerated clip is a
                # different URL and the browser cannot serve the old audio.
                "audio_url": "/audio/"
                + urllib.parse.quote(relative)
                + "?v="
                + str(row.get("sha256") or "")[:12],
            }
        )
    return clips


@dataclass
class VerdictStore:
    """Append-only verdict log. The last entry for a key wins."""

    path: Path
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def load(self) -> None:
        if not self.path.is_file():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("key") or "")
            if key:
                self.entries[key] = row

    def record(self, key: str, verdict: str, note: str = "") -> None:
        row = {"key": key, "verdict": verdict, "note": note, "at": int(time.time())}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), file=handle)
        self.entries[key] = row


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, address: tuple[str, int], *, root: Path, clips: Sequence[Mapping[str, Any]], store: VerdictStore
    ) -> None:
        super().__init__(address, ReviewHandler)
        self.root = root
        self.clips = list(clips)
        self.store = store


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # keep the job log readable
        return

    def _send(self, body: bytes, content_type: str, *, status: int = 200, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, payload: Mapping[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status=status)

    def _clips_payload(self) -> dict[str, Any]:
        return {"clips": self.server.clips, "verdicts": self.server.store.entries}

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/clips":
            self._send_json(self._clips_payload())
            return
        if path.startswith("/audio/"):
            self._send_audio(urllib.parse.unquote(path[len("/audio/") :]))
            return
        self._send_json({"error": "not found"}, status=404)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path != "/api/verdict":
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json"}, status=400)
            return
        key = str(payload.get("key") or "")
        verdict = str(payload.get("verdict") or "")
        if not key or verdict not in VERDICTS:
            self._send_json({"error": "key and verdict are required"}, status=400)
            return
        self.server.store.record(key, verdict, str(payload.get("note") or ""))
        self._send_json(self._clips_payload())

    def _send_audio(self, relative: str) -> None:
        target = (self.server.root / relative).resolve()
        root = self.server.root.resolve()
        if not target.is_file() or root not in target.parents:
            self._send_json({"error": "not found"}, status=404)
            return
        # Regenerating a clip rewrites the same file, so caching is unsafe.
        self._send(target.read_bytes(), "audio/wav", cache="no-store")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "outputs/tts/hey_eva")
    parser.add_argument("--index", type=Path, help="default: <run-root>/index.jsonl")
    parser.add_argument("--verdicts", type=Path, help="default: <run-root>/reviews.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.run_root).expanduser().resolve()
    index_path = Path(args.index).expanduser() if args.index else root / "index.jsonl"
    verdict_path = Path(args.verdicts).expanduser() if args.verdicts else root / "reviews.jsonl"

    clips = load_clips(index_path, root)
    if not clips:
        raise SystemExit(f"no clips with audio under {root} (index: {index_path})")
    store = VerdictStore(verdict_path)
    store.load()
    server = ReviewServer((args.host, args.port), root=root, clips=clips, store=store)
    host, port = server.server_address[0], server.server_address[1]
    print(f"review page: http://{host}:{port}/", flush=True)
    print(f"clips: {len(clips)} | verdicts so far: {len(store.entries)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
