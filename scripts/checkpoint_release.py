#!/usr/bin/env python3
"""Build, publish and fetch the checkpoint release.

The trained weights (453 files, about 9.2 GiB) are not committed to git. They are
published as GitHub release assets and indexed by a tracked manifest:

    checkpoints/manifest.tsv    asset <-> data/dma-kws/<relpath> <-> sha256

Every asset name is unique and reversible: it encodes the path relative to
data/dma-kws with slashes replaced by double underscores and any other unsafe
character replaced by a dash. The manifest is the source of truth, so the
encoding never has to be decoded by hand.

Subcommands:

    build     rescan data/dma-kws and rewrite the manifest (hashes every file)
    publish   upload the manifest's files to a release tag with the gh CLI
    fetch     download assets into data/dma-kws and verify their hashes

Examples:

    .venv/bin/python scripts/checkpoint_release.py build
    .venv/bin/python scripts/checkpoint_release.py publish --tag checkpoints-2026-10-08 --dry-run
    .venv/bin/python scripts/checkpoint_release.py publish --tag checkpoints-2026-10-08 --limit 3
    .venv/bin/python scripts/checkpoint_release.py fetch --tag checkpoints-2026-10-08 --only final/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

REPO_DEFAULT = "ty4045681/DMA-KWS"
TAG_DEFAULT = "checkpoints-2026-10-08"
MANIFEST_DEFAULT = "checkpoints/manifest.tsv"
DATA_ROOT_DEFAULT = "data/dma-kws"
STAGING_DEFAULT = "outputs/_checkpoint_release"
SNAPSHOT_ROOT_DEFAULT = "run_snapshots/lightning/exp"
SCAN_ROOTS = ("exp", "raw/kws-checkpoints")
MODEL_SUFFIXES = (".pt", ".ckpt", ".pth")
COLUMNS = ("asset", "relpath", "bytes", "sha256", "readout", "run", "snapshot")
MAX_ASSET_NAME = 200
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


# --------------------------------------------------------------------------- #
# naming and hashing
# --------------------------------------------------------------------------- #
def asset_name(relpath: str) -> str:
    """Encode a data/dma-kws relative path as a unique-safe asset name."""
    name = _SAFE.sub("-", relpath.replace("/", "__"))
    if len(name) > MAX_ASSET_NAME:
        digest = hashlib.sha256(relpath.encode("utf-8")).hexdigest()[:12]
        head, _, tail = name.rpartition(".")
        name = f"{head[: MAX_ASSET_NAME - len(digest) - 8]}-{digest}.{tail}"
    return name


def unique_asset_names(relpaths: Iterable[str]) -> dict[str, str]:
    """Map every relpath to a unique asset name, deterministically."""
    rows = sorted(set(relpaths))
    taken: dict[str, str] = {}
    names: dict[str, str] = {}
    for relpath in rows:
        name = asset_name(relpath)
        if taken.get(name) not in (None, relpath):
            digest = hashlib.sha256(relpath.encode("utf-8")).hexdigest()[:8]
            name = f"{name}-{digest}"
        taken[name] = relpath
        names[relpath] = name
    return names


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def readout_spec(path: Path) -> Any | None:
    """Return the QbyT score spec a checkpoint carries, or None.

    torch and dma_kws are imported lazily so the publish/fetch paths keep
    working on a machine that only has the gh CLI.
    """
    try:
        import torch  # noqa: PLC0415
        from dma_kws.training.checkpoint_io import checkpoint_qbyt_readout_spec
    except Exception:
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    try:
        return checkpoint_qbyt_readout_spec(payload)
    except Exception:
        return None


def format_readout(spec: Any | None) -> str:
    if spec is None:
        return ""
    return " ".join(str(spec).split())


def hydra_overrides(spec: Any) -> list[str]:
    """Ready-to-paste overrides describing the spec a checkpoint expects."""
    version = getattr(spec, "version", None)
    value = getattr(spec, "value", None)
    lines = [f"stage2.qbyt_readout_version={version}"]
    if version in (2, 3, 4) and hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            lines.append(f"stage2.qbyt_readout.{name}={_hydra_literal(getattr(value, name))}")
    return lines


def _hydra_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def iter_model_files(data_root: Path) -> list[str]:
    found: list[str] = []
    for scan_root in SCAN_ROOTS:
        base = data_root / scan_root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in MODEL_SUFFIXES:
                found.append(path.relative_to(data_root).as_posix())
    return sorted(found)


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def snapshot_keys(snapshot_root: Path) -> set[str]:
    """Run keys that have a mirrored Lightning hparams snapshot."""
    keys: set[str] = set()
    if snapshot_root.is_dir():
        for path in snapshot_root.rglob("hparams.yaml"):
            keys.add(path.parent.relative_to(snapshot_root).as_posix())
    return keys


def _version_index(parts: list[str]) -> int | None:
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].startswith("version_"):
            return index
    return None


def snapshot_candidates(relpath: str) -> tuple[str, list[str]]:
    """Return (run key, candidate snapshot keys) for one weights path.

    Data-disk runs repeat the run name around a checkpoints/ or logs/ directory,
    e.g. stage2_qbyt/checkpoints/<run>/<run>/version_0/last.ckpt, while the
    mirrored snapshot is stored once: stage2_qbyt/<run>/version_0/hparams.yaml.
    """
    parts = relpath.split("/")
    if parts and parts[0] == "exp":
        parts = parts[1:]
    index = _version_index(parts)
    if index is None:
        return "/".join(parts[:-1]), []

    run_parts = parts[:index]
    version = parts[index]
    candidates: list[str] = []

    def add(candidate: list[str]) -> None:
        key = "/".join([*candidate, version])
        if key not in candidates:
            candidates.append(key)

    add(list(run_parts))
    if len(run_parts) >= 2 and run_parts[-2] in ("checkpoints", "logs"):
        add(run_parts[:-2])
    run = candidates[-1].rsplit("/", 1)[0] if len(candidates) > 1 else "/".join(run_parts)
    return run, candidates


def write_manifest(path: Path, rows: list[dict]) -> None:
    lines = ["\t".join(COLUMNS)]
    for row in rows:
        lines.append("\t".join(str(row[column]) for column in COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if tuple(header) != COLUMNS:
            raise SystemExit(f"{path}: unexpected header {header!r}")
        rows = []
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            values = line.split("\t")
            if len(values) != len(COLUMNS):
                raise SystemExit(f"{path}: expected {len(COLUMNS)} columns, got {values!r}")
            rows.append(dict(zip(COLUMNS, values)))
    return rows


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def cmd_build(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root)
    files = iter_model_files(data_root)
    if not files:
        print(f"no model files under {data_root}", file=sys.stderr)
        return 1
    keys = snapshot_keys(Path(args.snapshot_root))
    names = unique_asset_names(files)

    rows: list[dict] = []
    total = 0
    for index, relpath in enumerate(files, start=1):
        path = data_root / relpath
        size = path.stat().st_size
        run, candidates = snapshot_candidates(relpath)
        snapshot = ""
        for key in candidates:
            if key in keys:
                snapshot = f"run_snapshots/lightning/exp/{key}/hparams.yaml"
                run = key.rsplit("/version_", 1)[0]
                break
        readout = "" if args.no_readout else format_readout(readout_spec(path))
        rows.append(
            {
                "asset": names[relpath],
                "relpath": relpath,
                "bytes": size,
                "sha256": sha256_file(path),
                "readout": readout,
                "run": run,
                "snapshot": snapshot,
            }
        )
        total += size
        if index % 25 == 0 or index == len(files):
            print(f"  hashed {index}/{len(files)} ({total / 1024 ** 3:.2f} GiB)", file=sys.stderr, flush=True)

    write_manifest(Path(args.manifest), rows)
    matched = sum(1 for row in rows if row["snapshot"])
    with_readout = sum(1 for row in rows if row["readout"])
    print(f"{len(rows)} files, {total / 1024 ** 3:.2f} GiB -> {args.manifest}")
    print(f"snapshots matched: {matched}/{len(rows)}; unique asset names: {len(set(names.values()))}")
    print(f"checkpoints carrying a QbyT readout spec: {with_readout}/{len(rows)}")
    return 0


# --------------------------------------------------------------------------- #
# publish
# --------------------------------------------------------------------------- #
def _gh(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True)


def release_exists(repo: str, tag: str) -> bool:
    return _gh(["release", "view", tag, "--repo", repo]).returncode == 0


def release_asset_sizes(repo: str, tag: str) -> dict[str, int]:
    proc = _gh(["release", "view", tag, "--repo", repo, "--json", "assets"])
    if proc.returncode != 0:
        return {}
    payload = json.loads(proc.stdout or "{}")
    return {asset["name"]: int(asset["size"]) for asset in payload.get("assets", [])}


def pending_rows(
    rows: list[dict],
    existing: dict[str, int],
    *,
    only: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    todo: list[dict] = []
    for row in rows:
        if only and only not in row["relpath"] and only not in row["asset"]:
            continue
        if existing.get(row["asset"]) == int(row["bytes"]):
            continue
        todo.append(row)
        if limit is not None and len(todo) >= limit:
            break
    return todo


def stage_asset(row: dict, data_root: Path, staging: Path) -> Path:
    """Symlink the weights under their unique asset name for gh to upload."""
    staging.mkdir(parents=True, exist_ok=True)
    link = staging / row["asset"]
    target = (data_root / row["relpath"]).resolve()
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)
    return link


def cmd_publish(args: argparse.Namespace) -> int:
    rows = read_manifest(Path(args.manifest))
    data_root = Path(args.data_root)
    staging = Path(args.staging)
    existing = release_asset_sizes(args.repo, args.tag)
    todo = pending_rows(rows, existing, only=args.only, limit=args.limit)
    todo_bytes = sum(int(row["bytes"]) for row in todo)
    print(
        f"{len(rows)} assets in manifest; {len(existing)} already on {args.tag}; "
        f"{len(todo)} to upload ({todo_bytes / 1024 ** 3:.2f} GiB)"
    )
    if args.dry_run:
        for row in todo[:10]:
            print(f"  would upload {row['asset']} ({int(row['bytes']) / 1024 ** 2:.1f} MiB)")
        if len(todo) > 10:
            print(f"  ... and {len(todo) - 10} more")
        return 0

    if not release_exists(args.repo, args.tag):
        cmd = [
            "release",
            "create",
            args.tag,
            "--repo",
            args.repo,
            "--title",
            args.title,
        ]
        if args.notes_file:
            cmd += ["--notes-file", args.notes_file]
        else:
            cmd += ["--notes", ""]
        proc = _gh(cmd)
        if proc.returncode != 0:
            print(proc.stderr.strip(), file=sys.stderr)
            return 1
        print(f"created release {args.tag}")

    uploaded = 0
    for start in range(0, len(todo), args.batch):
        chunk = todo[start : start + args.batch]
        links = [str(stage_asset(row, data_root, staging)) for row in chunk]
        proc = _gh(["release", "upload", args.tag, "--repo", args.repo, *links, "--clobber"])
        if proc.returncode != 0:
            print(proc.stderr.strip(), file=sys.stderr)
            print(f"failed at batch starting index {start}", file=sys.stderr)
            return 1
        uploaded += len(chunk)
        print(
            f"  uploaded {uploaded}/{len(todo)} "
            f"({sum(int(r['bytes']) for r in todo[:uploaded]) / 1024 ** 3:.2f} GiB sent)",
            file=sys.stderr,
            flush=True,
        )
    print(f"done: {uploaded} assets uploaded to {args.tag}")
    return 0


def cmd_show_readout(args: argparse.Namespace) -> int:
    path = Path(args.checkpoint)
    spec = readout_spec(path)
    if spec is None:
        print(f"{path}: no QbyT readout spec (raw encoder or unreadable checkpoint)")
        return 1
    print(f"{path}")
    print(f"  {format_readout(spec)}")
    print("  hydra overrides (add a leading '+' to keys your preset does not define):")
    for line in hydra_overrides(spec):
        print(f"    {line}")
    return 0


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #
def asset_url(repo: str, tag: str, asset: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{asset}"


def fetch_one(row: dict, repo: str, tag: str, data_root: Path) -> str:
    target = data_root / row["relpath"]
    expected = row["sha256"]
    if target.is_file() and sha256_file(target) == expected:
        return "have"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(
        asset_url(repo, tag, row["asset"]), headers={"User-Agent": "dma-kws-fetch"}
    )
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
    if sha256_file(partial) != expected:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"sha256 mismatch for {row['asset']}")
    os.replace(partial, target)
    return "fetched"


def cmd_fetch(args: argparse.Namespace) -> int:
    rows = read_manifest(Path(args.manifest))
    data_root = Path(args.data_root)
    selected = [row for row in rows if not args.only or args.only in row["relpath"] or args.only in row["asset"]]
    if args.list:
        for row in selected:
            print(f"{row['asset']}\t{row['relpath']}\t{int(row['bytes']) / 1024 ** 2:.1f} MiB")
        return 0
    fetched = 0
    for index, row in enumerate(selected, start=1):
        status = fetch_one(row, args.repo, args.tag, data_root)
        fetched += status == "fetched"
        if index % 10 == 0 or index == len(selected):
            print(f"  {index}/{len(selected)} ({fetched} downloaded)", file=sys.stderr, flush=True)
    print(f"{len(selected)} assets checked, {fetched} downloaded into {data_root}")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)


    build = subparsers.add_parser("build", help="rewrite the manifest from data/dma-kws")
    build.add_argument("--manifest", default=MANIFEST_DEFAULT)
    build.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    build.add_argument("--snapshot-root", default=SNAPSHOT_ROOT_DEFAULT)
    build.add_argument(
        "--no-readout",
        action="store_true",
        help="skip loading every checkpoint to record its QbyT readout spec",
    )
    build.set_defaults(func=cmd_build)

    show = subparsers.add_parser(
        "show-readout", help="print the QbyT readout spec a checkpoint expects"
    )
    show.add_argument("checkpoint", help="path to a .pt/.ckpt checkpoint")
    show.set_defaults(func=cmd_show_readout)

    publish = subparsers.add_parser("publish", help="upload manifest assets with the gh CLI")
    publish.add_argument("--manifest", default=MANIFEST_DEFAULT)
    publish.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    publish.add_argument("--repo", default=REPO_DEFAULT)
    publish.add_argument("--tag", default=TAG_DEFAULT)
    publish.add_argument("--title", default="DMA-KWS checkpoints")
    publish.add_argument("--notes-file", default="checkpoints/RELEASE_NOTES.md")
    publish.add_argument("--staging", default=STAGING_DEFAULT)
    publish.add_argument("--batch", type=int, default=20)
    publish.add_argument("--only", default=None)
    publish.add_argument("--limit", type=int, default=None)
    publish.add_argument("--dry-run", action="store_true")
    publish.set_defaults(func=cmd_publish)

    fetch = subparsers.add_parser("fetch", help="download manifest assets and verify hashes")
    fetch.add_argument("--manifest", default=MANIFEST_DEFAULT)
    fetch.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    fetch.add_argument("--repo", default=REPO_DEFAULT)
    fetch.add_argument("--tag", default=TAG_DEFAULT)
    fetch.add_argument("--only", default=None)
    fetch.add_argument("--list", action="store_true")
    fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
