#!/usr/bin/env python3
"""Serve a local page for reviewing the real (human) Hey Eva recordings.

The page reads <review-root>/index.jsonl, plays the wav files under
--corpus-root, and appends every verdict to <review-root>/reviews.jsonl:
'ok' (正确), 'bad' (不可用) and 'wrong' (说的是别的词, with a free-text note
naming what was actually said).  The last entry for a key wins, so a page
reload never loses work.

    python3 scripts/review_real_clips.py \
        --corpus-root data/dma-kws/raw/hey_eva_real \
        --review-root outputs/review_real_hey_eva --port 8768

Keyboard: 1 ok, 2 bad, 3 wrong (submits the note field), 0 clears, space
plays/pauses, arrows move between clips.

--apply turns the verdict log into a reviewed manifest CSV.  An augmentation
inherits its original recording's verdict unless it was judged itself.  'bad'
clips are dropped, 'wrong' clips are relabelled with the heard phrase (a wrong
clip whose note is the keyword becomes a positive), and a 'wrong' clip with an
empty note is dropped because no usable transcript exists for it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERDICTS = ("ok", "bad", "wrong", "")
DEFAULT_KEYWORD = "Hey Eva"
DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "data/dma-kws/raw/hey_eva_real"
DEFAULT_REVIEW_ROOT = PROJECT_ROOT / "outputs/review_real_hey_eva"


PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>真人 Hey Eva 语料复核</title>
<style>
:root{--bg:#101319;--panel:#181d27;--line:#2a3140;--fg:#e7eaf1;--muted:#93a0b4;
--ok:#2fbf71;--bad:#e5484d;--wrong:#e5a13d;--accent:#4c8dff}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;display:flex;flex-direction:column;background:var(--bg);color:var(--fg);
font:14px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans SC",sans-serif}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 18px;
border-bottom:1px solid var(--line);background:var(--panel)}
#progress{font-variant-numeric:tabular-nums}
#progress b{color:var(--accent)}
button{background:#232a38;color:var(--fg);border:1px solid var(--line);border-radius:8px;
padding:6px 12px;cursor:pointer;font:inherit}
button:hover{border-color:var(--accent)}
button.active{border-color:var(--accent);color:var(--accent)}
button.ok{border-color:var(--ok)}
button.bad{border-color:var(--bad)}
button.wrong{border-color:var(--wrong)}
main{flex:1;display:grid;grid-template-columns:minmax(0,1fr) 360px;min-height:0}
section{padding:22px;display:flex;flex-direction:column;gap:14px;min-width:0;overflow-y:auto}
#clipText{font-size:34px;font-weight:600;letter-spacing:.5px}
#clipMeta{color:var(--muted);display:flex;gap:14px;flex-wrap:wrap}
#clipMeta code{color:var(--fg)}
#clipMeta .badge{border:1px solid var(--line);border-radius:6px;padding:0 8px;font-size:12px}
#origVerdict{color:var(--muted);font-size:13px}
audio{width:100%}
#actions{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
#noteRow{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
#note{flex:1;min-width:220px;background:#0d1016;color:var(--fg);border:1px solid var(--line);
border-radius:8px;padding:8px 10px;font:inherit}
#chips button{font-size:12px;padding:3px 8px}
#hint{color:var(--muted);font-size:12px}
aside{position:relative;border-left:1px solid var(--line);overflow-y:auto;background:var(--panel)}
.row{display:flex;gap:8px;align-items:center;padding:7px 12px;cursor:pointer;
border-bottom:1px solid rgba(255,255,255,.04)}
.row:hover{background:rgba(255,255,255,.04)}
.row.current{background:rgba(76,141,255,.16)}
.row .text{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row .who{color:var(--muted);font-size:12px;max-width:120px;overflow:hidden;
text-overflow:ellipsis;white-space:nowrap}
.dot{width:9px;height:9px;border-radius:50%;background:#3a4356;flex:none}
.dot.ok{background:#2fbf71}
.dot.bad{background:#e5484d}
.dot.wrong{background:#e5a13d}
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
    <div id="origVerdict"></div>
    <audio id="audio" controls preload="auto"></audio>
    <div id="actions">
      <button id="ok" class="ok">正确 (1)</button>
      <button id="bad" class="bad">不可用 (2)</button>
      <button id="wrong" class="wrong">说的是别的词 (3)</button>
      <button id="clear">清除 (0)</button>
      <button id="prev">上一条 (←)</button>
      <button id="next">下一条 (→)</button>
    </div>
    <div id="noteRow">
      <input id="note" list="phrases" placeholder="听到的实际内容（点了“别的词”时填，回车提交）">
      <datalist id="phrases"></datalist>
      <span id="chips"></span>
    </div>
    <div id="hint">空格播放/暂停。判定即时写入 reviews.jsonl；“别的词”不填内容也能记，但该条会因为没有可用文本而被丢弃。增广片段默认继承其原始录音的判定。</div>
  </section>
  <aside id="list"></aside>
</main>
<script>
(function () {
  "use strict";
  var state = { clips: [], verdicts: {}, index: 0, filter: "orig", autonext: true, played: false };
  var audio = document.getElementById("audio");
  var listEl = document.getElementById("list");
  var progressEl = document.getElementById("progress");
  var filtersEl = document.getElementById("filters");
  var textEl = document.getElementById("clipText");
  var metaEl = document.getElementById("clipMeta");
  var origEl = document.getElementById("origVerdict");
  var noteEl = document.getElementById("note");
  var phrasesEl = document.getElementById("phrases");
  var chipsEl = document.getElementById("chips");
  var prefetch = null;
  var rowEls = [];

  var FILTERS = [["orig", "原始录音"], ["pending", "只看未复核"], ["pending_orig", "未复核·原始"],
    ["aug", "增广"], ["ok", "正确"], ["bad", "不可用"], ["wrong", "别的词"],
    ["positive", "正样本"], ["near_negative", "近音负"], ["all", "全部"]];
  var QUICK_PHRASES = ["Hey Eva", "Hey Ava", "Hey Eve", "Hey Eric", "Hey Ivan", "Hi Eva"];

  function esc(value) {
    return String(value).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function verdictOf(key) {
    var entry = state.verdicts[key];
    return entry ? entry.verdict : "";
  }

  function effective(clip) {
    var own = verdictOf(clip.key);
    if (own) { return own; }
    if (clip.is_original === 0 && clip.orig_key) { return verdictOf(clip.orig_key); }
    return "";
  }

  function passes(clip) {
    var verdict = verdictOf(clip.key);
    var inherited = effective(clip);
    switch (state.filter) {
      case "all": return true;
      case "pending": return inherited === "";
      case "pending_orig": return clip.is_original === 1 && verdict === "";
      case "orig": return clip.is_original === 1;
      case "aug": return clip.is_original === 0;
      case "positive": return clip.category === "positive";
      case "near_negative": return clip.category === "near_negative";
      case "ok": case "bad": case "wrong": return inherited === state.filter;
      default: return true;
    }
  }

  function visible() {
    var out = [];
    for (var i = 0; i < state.clips.length; i += 1) {
      if (passes(state.clips[i])) { out.push(i); }
    }
    return out;
  }

  function counts() {
    var ok = 0, bad = 0, wrong = 0;
    for (var key in state.verdicts) {
      var verdict = state.verdicts[key].verdict;
      if (verdict === "ok") { ok += 1; }
      else if (verdict === "bad") { bad += 1; }
      else if (verdict === "wrong") { wrong += 1; }
    }
    return { ok: ok, bad: bad, wrong: wrong, total: state.clips.length };
  }

  function paintProgress() {
    var c = counts();
    progressEl.innerHTML = "已判定 <b>" + (c.ok + c.bad + c.wrong) + "</b> · 正确 " + c.ok +
      " · 不可用 <b>" + c.bad + "</b> · 别的词 <b>" + c.wrong + "</b> / 共 " + c.total;
  }

  function paintRow(el, clip) {
    var verdict = verdictOf(clip.key) || effective(clip);
    el.querySelector(".dot").className = "dot " + verdict;
    el.classList.toggle("current", state.clips[state.index] === clip);
  }

  function paintRows() {
    for (var i = 0; i < rowEls.length; i += 1) {
      var entry = rowEls[i];
      entry.el.style.display = passes(entry.clip) ? "" : "none";
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
      row.innerHTML = '<span class="dot"></span><span class="text">' +
        esc(clip.is_original === 1 ? clip.text : clip.text + " · " + clip.aug_type) +
        '</span><span class="tag">' + (clip.label === 1 ? "正" : "负") +
        '</span><span class="who">' + esc(clip.speaker) + "</span>";
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

  function renderPhrases() {
    var seen = {};
    QUICK_PHRASES.forEach(function (phrase) {
      if (seen[phrase]) { return; }
      seen[phrase] = true;
      var option = document.createElement("option");
      option.value = phrase;
      phrasesEl.appendChild(option);
      var chip = document.createElement("button");
      chip.textContent = phrase;
      chip.addEventListener("click", function () { noteEl.value = phrase; judge("wrong"); });
      chipsEl.appendChild(chip);
    });
  }

  function paintCurrent() {
    var clip = state.clips[state.index];
    if (!clip) { return; }
    var lines = [];
    var own = state.verdicts[clip.key];
    if (own) {
      lines.push("当前判定：" + zh(own.verdict) + (own.note ? "（" + own.note + "）" : ""));
    } else {
      lines.push("当前判定：未复核");
    }
    if (clip.is_original === 0 && clip.orig_key) {
      var originalVerdict = verdictOf(clip.orig_key);
      if (originalVerdict) {
        lines.push("原始录音判定：" + zh(originalVerdict) + "（未单独判定时继承）");
      }
    }
    origEl.textContent = lines.join(" · ");
  }

  function scrollRowIntoView(index) {
    var entry = rowEls[index];
    if (!entry) { return; }
    var top = entry.el.offsetTop;
    var bottom = top + entry.el.offsetHeight;
    var viewTop = listEl.scrollTop;
    var viewBottom = viewTop + listEl.clientHeight;
    if (top < viewTop) { listEl.scrollTop = top; }
    else if (bottom > viewBottom) { listEl.scrollTop = bottom - listEl.clientHeight; }
  }

  function select(index) {
    if (index < 0 || index >= state.clips.length) { return; }
    state.index = index;
    var clip = state.clips[index];
    textEl.textContent = clip.text + (clip.is_original === 1 ? "" : "（" + clip.aug_type + "）");
    metaEl.innerHTML = "<span class='badge'>" + (clip.category === "positive" ? "正样本" : "近音负") +
      "</span><span>说话人 <code>" + esc(clip.speaker) + "</code></span><span>时长 " +
      clip.duration.toFixed(2) + "s</span><span>" + (clip.is_original === 1 ? "原始录音" : "增广自 <code>" +
      esc(clip.orig_key.split("/").pop()) + "</code>") + "</span><span>" + esc(clip.key) + "</span>";
    paintCurrent();
    noteEl.value = (state.verdicts[clip.key] && state.verdicts[clip.key].note) || "";
    audio.src = clip.audio_url;
    audio.load();
    paintRows();
    scrollRowIntoView(index);
    if (state.autonext) {
      audio.play().then(function () { state.played = true; }).catch(function () {});
    }
    preloadNext(index);
  }

  function zh(verdict) {
    return { ok: "正确", bad: "不可用", wrong: "别的词" }[verdict] || "未复核";
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

  function send(key, verdict, note) {
    return fetch("/api/verdict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: key, verdict: verdict, note: note || "" })
    }).then(function (response) { return response.json(); }).then(function (payload) {
      state.verdicts = payload.verdicts;
      paintProgress();
      paintRows();
      paintCurrent();
    });
  }

  function judge(verdict) {
    var clip = state.clips[state.index];
    if (!clip) { return; }
    var note = verdict === "wrong" ? noteEl.value : "";
    var current = state.verdicts[clip.key];
    var toggle = current && current.verdict === verdict && verdict !== "wrong" &&
      (current.note || "") === note;
    send(clip.key, toggle ? "" : verdict, note).then(function () {
      if (!toggle) { step(1); }
    });
  }

  document.getElementById("ok").addEventListener("click", function () { judge("ok"); });
  document.getElementById("bad").addEventListener("click", function () { judge("bad"); });
  document.getElementById("wrong").addEventListener("click", function () { judge("wrong"); });
  document.getElementById("clear").addEventListener("click", function () {
    var clip = state.clips[state.index];
    if (clip) { noteEl.value = ""; send(clip.key, "", ""); }
  });
  document.getElementById("prev").addEventListener("click", function () { step(-1); });
  document.getElementById("next").addEventListener("click", function () { step(1); });
  document.getElementById("autonext").addEventListener("change", function (event) {
    state.autonext = event.target.checked;
  });
  noteEl.addEventListener("keydown", function (event) {
    if (event.key === "Enter") { event.preventDefault(); judge("wrong"); }
  });
  audio.addEventListener("play", function () { state.played = true; });

  document.addEventListener("keydown", function (event) {
    if (event.target.tagName === "INPUT") { return; }
    if (event.code === "Space") { event.preventDefault(); if (audio.paused) { audio.play(); } else { audio.pause(); } }
    else if (event.key === "1") { judge("ok"); }
    else if (event.key === "2") { judge("bad"); }
    else if (event.key === "3") { noteEl.focus(); }
    else if (event.key === "0") { var clip = state.clips[state.index]; if (clip) { send(clip.key, "", ""); } }
    else if (event.key === "ArrowLeft") { step(-1); }
    else if (event.key === "ArrowRight") { step(1); }
  });

  fetch("/api/clips").then(function (response) { return response.json(); }).then(function (payload) {
    state.clips = payload.clips;
    state.verdicts = payload.verdicts || {};
    renderPhrases();
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


def _corpus_relative(path: str) -> str:
    text = str(path or "").strip()
    if text.startswith("data/"):
        text = text[len("data/") :]
    return text


def build_index(corpus_root: Path, db_path: Path, out_path: Path) -> int:
    """Write one index row per recording that still exists under corpus_root."""

    if not db_path.is_file():
        raise SystemExit(f"recording database not found: {db_path}")
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT r.file_path, r.category, r.wake_word, r.is_original, r.aug_type,
                   r.group_id, r.orig_path, r.duration, s.name
            FROM recordings AS r
            JOIN speakers AS s ON s.id = r.speaker_id
            """
        ).fetchall()
    finally:
        connection.close()

    records: list[dict[str, Any]] = []
    missing = 0
    for (
        file_path,
        category,
        wake_word,
        is_original,
        aug_type,
        group_id,
        orig_path,
        duration,
        speaker,
    ) in rows:
        key = _corpus_relative(file_path)
        target = corpus_root / key
        if not key or not target.is_file():
            missing += 1
            continue
        records.append(
            {
                "key": key,
                "audio_path": key,
                "text": str(wake_word or ""),
                "label": 1 if str(category) == "positive" else 0,
                "category": str(category or ""),
                "wake_word": str(wake_word or ""),
                "speaker": str(speaker or ""),
                "is_original": int(is_original or 0),
                "aug_type": str(aug_type or ""),
                "group_id": str(group_id or ""),
                "orig_key": _corpus_relative(orig_path) or key,
                "duration": float(duration or 0.0),
                "bytes": target.stat().st_size,
            }
        )

    records.sort(
        key=lambda row: (
            -row["is_original"],
            0 if row["category"] == "positive" else 1,
            row["wake_word"],
            row["speaker"],
            row["key"],
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        f"index: {len(records)} clips ({missing} rows had no wav on disk) -> {out_path}",
        flush=True,
    )
    return len(records)


def load_clips(index_path: Path, root: Path) -> list[dict[str, Any]]:
    """Return the index rows whose audio still exists under root."""

    clips: list[dict[str, Any]] = []
    for row in read_index(index_path):
        relative = str(row.get("audio_path") or "")
        target = root / relative
        if not relative or not target.is_file():
            continue
        row = dict(row)
        row["key"] = str(row.get("key") or relative)
        row["bytes"] = target.stat().st_size
        # The size in the URL makes a replaced wav a different URL, so the
        # browser cannot replay a stale clip from its cache.
        row["audio_url"] = "/audio/" + urllib.parse.quote(relative) + "?v=" + str(row["bytes"])
        row.setdefault("orig_key", row["key"])
        row.setdefault("is_original", 1)
        clips.append(row)
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
        self,
        address: tuple[str, int],
        *,
        root: Path,
        clips: Sequence[Mapping[str, Any]],
        store: VerdictStore,
    ) -> None:
        super().__init__(address, ReviewHandler)
        self.root = root
        self.clips = [dict(clip) for clip in clips]
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
        note = str(payload.get("note") or "").strip()[:200]
        if not key or verdict not in VERDICTS:
            self._send_json({"error": "key and verdict are required"}, status=400)
            return
        self.server.store.record(key, verdict, note)
        # Only the verdict map: the full clip list is several megabytes and the
        # page never reads it from a POST response.
        self._send_json({"verdicts": self.server.store.entries})

    def _send_audio(self, relative: str) -> None:
        target = (self.server.root / relative).resolve()
        root = self.server.root.resolve()
        if not target.is_file() or root not in target.parents:
            self._send_json({"error": "not found"}, status=404)
            return
        content_type = "audio/wav" if target.suffix.casefold() == ".wav" else "application/octet-stream"
        self._send(target.read_bytes(), content_type, cache="no-store")


def _normalize_phrase(text: str, known: Mapping[str, str]) -> str:
    collapsed = " ".join(str(text or "").split())
    if not collapsed:
        return ""
    letters = re.sub(r"[^a-z ]", "", collapsed.casefold()).strip()
    letters = " ".join(letters.split())
    return known.get(letters, collapsed)


def _letters(value: str) -> str:
    return " ".join(re.sub(r"[^a-z ]", "", str(value or "").casefold()).split())


def _clean_aliases(speaker_aliases: Mapping[str, str] | None) -> dict[str, str]:
    return {
        str(old).strip(): str(new).strip()
        for old, new in (speaker_aliases or {}).items()
        if str(old).strip() and str(new).strip()
    }


def _parse_speaker_aliases(text: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for chunk in str(text or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"--speaker-alias expects OLD=NEW pairs, got {chunk!r}")
        old, new = (part.strip() for part in chunk.split("=", 1))
        if not old or not new:
            raise SystemExit(f"--speaker-alias expects OLD=NEW pairs, got {chunk!r}")
        aliases[old] = new
    return aliases


def resolve_reviewed_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    store: VerdictStore,
    keyword: str = DEFAULT_KEYWORD,
    eval_speakers: Sequence[str] = (),
    speaker_aliases: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Resolve every index row against the verdict log.

    A clip keeps its own verdict; an augmentation without one inherits its
    original recording's verdict.  A 'wrong' clip is relabelled with the heard
    phrase (a note that is the keyword turns it into a positive) and a 'wrong'
    clip without a note is dropped because no usable text exists.
    """

    known: dict[str, str] = {}
    for row in rows:
        phrase = str(row.get("text") or "").strip()
        if phrase:
            known.setdefault(_letters(phrase), phrase)
    keyword_letters = _letters(keyword)
    eval_set = {name.strip().casefold() for name in eval_speakers if name.strip()}
    aliases = _clean_aliases(speaker_aliases)

    keep: list[dict[str, Any]] = []
    dropped_bad: list[dict[str, Any]] = []
    dropped_no_text: list[dict[str, Any]] = []
    inherited = 0

    for row in rows:
        key = str(row.get("key") or "")
        entry = store.entries.get(key)
        source = "own"
        if entry is None and not int(row.get("is_original") or 0):
            entry = store.entries.get(str(row.get("orig_key") or ""))
            source = "inherited" if entry is not None else "none"
        verdict = str((entry or {}).get("verdict") or "")
        note = str((entry or {}).get("note") or "")
        if source == "inherited":
            inherited += 1

        dropped = {
            "audio_path": str(row.get("audio_path") or key),
            "verdict": verdict,
            "verdict_source": source,
            "note": note,
        }
        if verdict == "bad":
            dropped_bad.append({**dropped, "reason": "bad"})
            continue

        text = str(row.get("text") or "")
        if verdict == "wrong":
            text = _normalize_phrase(note, known)
            if not text:
                dropped_no_text.append({**dropped, "reason": "wrong_without_note"})
                continue
        label = 1 if _letters(text) == keyword_letters else 0
        source_speaker = str(row.get("speaker") or "")
        speaker = aliases.get(source_speaker, source_speaker)
        resolved = {
            "source_audio_path": str(row.get("audio_path") or key),
            "source_key": key,
            "text": text,
            "label": label,
            "phase": "real",
            "speaker_id": speaker,
            "source_speaker": source_speaker,
            "category": "positive" if label == 1 else "near_negative",
            "wake_word": str(row.get("wake_word") or ""),
            "is_original": int(row.get("is_original") or 0),
            "aug_type": str(row.get("aug_type") or ""),
            "verdict": verdict,
            "verdict_source": source,
            "note": note,
        }
        if eval_set:
            resolved["split"] = "eval" if speaker.casefold() in eval_set else "train"
        keep.append(resolved)

    by_verdict: dict[str, int] = {}
    for entry in store.entries.values():
        verdict = str(entry.get("verdict") or "")
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
    stats: dict[str, Any] = {
        "keyword": keyword,
        "index_rows": len(rows),
        "judged_clips": len(store.entries),
        "verdicts": by_verdict,
        "kept": len(keep),
        "kept_positive": sum(1 for row in keep if row["label"] == 1),
        "kept_negative": sum(1 for row in keep if row["label"] == 0),
        "kept_original": sum(1 for row in keep if row["is_original"] == 1),
        "kept_augmented": sum(1 for row in keep if row["is_original"] == 0),
        "dropped_bad": len(dropped_bad),
        "dropped_wrong_without_note": len(dropped_no_text),
        "inherited_verdicts": inherited,
        "eval_speakers": sorted(eval_set),
    }
    details = {"dropped_bad": dropped_bad, "dropped_wrong_without_note": dropped_no_text}
    return keep, stats, details


MANIFEST_FIELDS = (
    "speaker_id",
    "source_speaker",
    "category",
    "wake_word",
    "is_original",
    "aug_type",
    "verdict",
    "verdict_source",
    "note",
    "source_audio_path",
)


def _manifest_fields(*, split: bool) -> list[str]:
    fields = ["audio_path", "text", "label", "phase"]
    if split:
        fields.append("split")
    fields.extend(MANIFEST_FIELDS)
    return fields


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def apply_verdicts(
    *,
    index_path: Path,
    verdict_path: Path,
    output_path: Path,
    keyword: str = DEFAULT_KEYWORD,
    eval_speakers: Sequence[str] = (),
    speaker_aliases: Mapping[str, str] | None = None,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Write a reviewed manifest CSV from the index plus the verdict log."""

    rows = read_index(index_path)
    store = VerdictStore(verdict_path)
    store.load()
    keep, stats, _details = resolve_reviewed_rows(
        rows=rows,
        store=store,
        keyword=keyword,
        eval_speakers=eval_speakers,
        speaker_aliases=speaker_aliases,
    )
    fieldnames = _manifest_fields(split=bool(stats["eval_speakers"]))
    _write_csv(
        output_path,
        fieldnames,
        [{**row, "audio_path": row["source_audio_path"]} for row in keep],
    )
    stats["output"] = str(output_path)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return stats


def _aliased_speaker_dir(directory: str, *, source_speaker: str, aliased_speaker: str) -> str:
    """Rename a <speaker>_<gender>_<age> directory when the speaker is aliased."""

    if not source_speaker or not aliased_speaker or source_speaker == aliased_speaker:
        return directory
    if directory == source_speaker:
        return aliased_speaker
    if directory.startswith(source_speaker + "_"):
        return aliased_speaker + directory[len(source_speaker) :]
    return directory


def build_reviewed_view(
    *,
    index_path: Path,
    verdict_path: Path,
    source_root: Path,
    target_root: Path,
    keyword: str = DEFAULT_KEYWORD,
    eval_speakers: Sequence[str] = (),
    speaker_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Copy the reviewed clips into a corrected dataset tree plus manifests.

    Kept clips keep their audio bytes; bad clips (and the augmentations that
    inherit that verdict) are left out, and a wrong clip is placed under the
    phrase it actually says.  The source corpus is never modified.
    """

    if (target_root / "record_data").exists():
        raise SystemExit(f"target dataset already exists: {target_root}")
    rows = read_index(index_path)
    store = VerdictStore(verdict_path)
    store.load()
    keep, stats, details = resolve_reviewed_rows(
        rows=rows,
        store=store,
        keyword=keyword,
        eval_speakers=eval_speakers,
        speaker_aliases=speaker_aliases,
    )

    for row in keep:
        source_rel = Path(row["source_audio_path"])
        parts = source_rel.parts
        if len(parts) != 5 or parts[0] != "record_data":
            raise SystemExit(f"unexpected corpus path: {source_rel}")
        speaker_dir = _aliased_speaker_dir(
            parts[3],
            source_speaker=str(row.get("source_speaker") or ""),
            aliased_speaker=str(row.get("speaker_id") or ""),
        )
        target_rel = (
            Path("record_data")
            / row["category"]
            / row["text"].replace(" ", "_")
            / speaker_dir
            / parts[4]
        )
        source = source_root / source_rel
        if not source.is_file():
            raise SystemExit(f"missing source wav: {source}")
        target = target_root / target_rel
        if target.exists():
            raise SystemExit(f"duplicate target path while copying: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        row["audio_path"] = target_rel.as_posix()
        row["absolute_audio_path"] = str(target.resolve())

    fieldnames = _manifest_fields(split=bool(stats["eval_speakers"]))
    manifest_dir = target_root / "manifests"
    _write_csv(manifest_dir / "real_reviewed.csv", fieldnames, keep)
    _write_csv(
        manifest_dir / "real_reviewed_abs.csv",
        fieldnames,
        [{**row, "audio_path": row["absolute_audio_path"]} for row in keep],
    )
    dropped_rows = details["dropped_bad"] + details["dropped_wrong_without_note"]
    _write_csv(
        manifest_dir / "dropped_clips.csv",
        ["audio_path", "verdict", "verdict_source", "note", "reason"],
        dropped_rows,
    )

    aliases = _clean_aliases(speaker_aliases)
    stats["speaker_aliases"] = [f"{old} -> {new}" for old, new in sorted(aliases.items())]
    stats["source_root"] = str(source_root)
    stats["target_root"] = str(target_root)
    stats["manifest"] = str(manifest_dir / "real_reviewed.csv")
    stats["manifest_absolute"] = str(manifest_dir / "real_reviewed_abs.csv")
    summary = {"stats": stats, "dropped": details}
    (target_root / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target_root / "README.md").write_text(
        _view_readme(stats=stats, source_root=source_root),
        encoding="utf-8",
    )
    return stats


def _view_readme(*, stats: Mapping[str, Any], source_root: Path) -> str:
    return f"""# hey_eva_real_v2 — reviewed real "Hey Eva" dataset

Corrected view built from the pre-review backup `{source_root}` plus the
human verdict log (reviews.jsonl).  The backup is untouched.

## What changed

- {stats["kept"]} clips kept ({stats["kept_positive"]} positive / {stats["kept_negative"]} near
  negative; {stats["kept_original"]} originals + {stats["kept_augmented"]} augmentations).
- {stats["dropped_bad"]} clips dropped because their original recording was marked 不可用
  (augmentations inherit their original's verdict).
- {stats["dropped_wrong_without_note"]} clips dropped because they were marked 别的词 without a note.
- Clips marked 别的词 are placed under the phrase actually heard; a note equal to
  the keyword turns a clip into a positive.
- Speaker aliases applied: {", ".join(stats.get("speaker_aliases", [])) or "none"}.
- File names keep the original phrase for traceability; the manifest is the
  source of truth for text and label.  The source_speaker column keeps the
  pre-alias speaker name.

## Layout

    record_data/<category>/<phrase>/<speaker_gender_age>/*.wav
    manifests/real_reviewed.csv       # audio_path relative to this root
    manifests/real_reviewed_abs.csv   # absolute audio_path, ready for adapt.manifest_csv
    manifests/dropped_clips.csv       # every excluded clip with its reason
    review_summary.json

## Using it for Stage-II QbyT adaptation

Point the adapter at the absolute manifest:

    +experiment=adapt_hey_eva_icefall_adapter_v2 \\
      adapt.data_root=<a scratch adapt root> \\
      prep.manifest_csv={stats["manifest_absolute"]} \\
      adapt.train_phases=[real]

prepare_adapt mirrors fbank/ under adapt.data_root using the path segment after
raw/, so keep adapt.data_root outside this dataset directory.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--db", type=Path, help="default: <corpus-root>/app.db")
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--index", type=Path, help="default: <review-root>/index.jsonl")
    parser.add_argument("--verdicts", type=Path, help="default: <review-root>/reviews.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--keyword", default=DEFAULT_KEYWORD)
    parser.add_argument("--build-index", action="store_true", help="rebuild the index and continue")
    parser.add_argument(
        "--apply",
        type=Path,
        help="write the reviewed manifest CSV to this path and exit",
    )
    parser.add_argument(
        "--view",
        type=Path,
        help="copy the reviewed clips into this corrected dataset root (with manifests) and exit",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="default: <review-root>/review_summary.json; used with --apply",
    )
    parser.add_argument(
        "--eval-speakers",
        default="",
        help="comma-separated speaker names held out as the eval split",
    )
    parser.add_argument(
        "--speaker-alias",
        default="",
        help="comma-separated OLD=NEW speaker aliases, e.g. '解震2=解震'",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus_root = Path(args.corpus_root).expanduser().resolve()
    db_path = Path(args.db).expanduser() if args.db else corpus_root / "app.db"
    review_root = Path(args.review_root).expanduser().resolve()
    index_path = Path(args.index).expanduser() if args.index else review_root / "index.jsonl"
    verdict_path = Path(args.verdicts).expanduser() if args.verdicts else review_root / "reviews.jsonl"

    if args.build_index or not index_path.is_file():
        build_index(corpus_root, db_path, index_path)

    eval_speakers = [part for part in str(args.eval_speakers).split(",") if part.strip()]
    speaker_aliases = _parse_speaker_aliases(args.speaker_alias)
    if args.view is not None:
        stats = build_reviewed_view(
            index_path=index_path,
            verdict_path=verdict_path,
            source_root=corpus_root,
            target_root=Path(args.view).expanduser().resolve(),
            keyword=args.keyword,
            eval_speakers=eval_speakers,
            speaker_aliases=speaker_aliases,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.apply is not None:
        summary_path = Path(args.summary).expanduser() if args.summary else review_root / "review_summary.json"
        stats = apply_verdicts(
            index_path=index_path,
            verdict_path=verdict_path,
            output_path=Path(args.apply).expanduser(),
            keyword=args.keyword,
            eval_speakers=eval_speakers,
            speaker_aliases=speaker_aliases,
            summary_path=summary_path,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    clips = load_clips(index_path, corpus_root)
    if not clips:
        raise SystemExit(f"no clips with audio under {corpus_root} (index: {index_path})")
    store = VerdictStore(verdict_path)
    store.load()
    server = ReviewServer((args.host, args.port), root=corpus_root, clips=clips, store=store)
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
