#!/usr/bin/env python3
"""Mirror per-run config snapshots into the tracked run_snapshots/ tree.

Two kinds of snapshots exist on this machine:

* runs launched through the repo's Hydra entry points keep the composed config,
  the Hydra metadata and the exact CLI overrides under
  outputs/<date>/<time>/.hydra/ ;
* runs launched outside the repo (on the data disk) keep a Lightning
  hparams.yaml next to their logs under
  <exp_root>/<run>/logs/<run>/version_N/hparams.yaml .

Both outputs/ and the data-disk experiment tree are gitignored, so the snapshots
are copied (never moved) into run_snapshots/ and committed from there. The copy
is idempotent: a file is rewritten only when its bytes differ.

Usage:

    .venv/bin/python scripts/collect_run_snapshots.py
    .venv/bin/python scripts/collect_run_snapshots.py --check   # exit 1 if stale
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HYDRA_FILES = ("config.yaml", "hydra.yaml", "overrides.yaml")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hydra_snapshots(outputs_root: Path) -> list[tuple[Path, Path]]:
    found: list[tuple[Path, Path]] = []
    for hydra_dir in sorted(outputs_root.glob("*/*/.hydra")):
        rel = hydra_dir.parent.relative_to(outputs_root)
        for name in HYDRA_FILES:
            src = hydra_dir / name
            if src.is_file():
                found.append((src, Path("hydra") / rel / name))
    return found


def _lightning_snapshots(root: Path, label: str) -> list[tuple[Path, Path]]:
    found: list[tuple[Path, Path]] = []
    for src in sorted(root.rglob("hparams.yaml")):
        parts = list(src.relative_to(root).parts)
        # <run>/logs/<run>/version_N/hparams.yaml -> <run>/version_N/hparams.yaml
        if "logs" in parts:
            index = parts.index("logs")
            if 0 < index < len(parts) - 1 and parts[index - 1] == parts[index + 1]:
                del parts[index : index + 2]
        found.append((src, Path("lightning") / label / Path(*parts)))
    return found


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--exp-root", default="data/dma-kws/exp")
    parser.add_argument("--dest", default="run_snapshots")
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if run_snapshots/ is out of date",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    exp_root = Path(args.exp_root)
    dest = Path(args.dest)

    snapshots = _hydra_snapshots(outputs_root)
    snapshots += _lightning_snapshots(outputs_root, "outputs")
    if exp_root.is_dir():
        snapshots += _lightning_snapshots(exp_root, "exp")

    seen: dict[Path, Path] = {}
    for src, rel in snapshots:
        if rel in seen:
            raise SystemExit(f"snapshot path collision: {rel} from {src} and {seen[rel]}")
        seen[rel] = src

    rows: list[str] = []
    written = 0
    for src, rel in sorted(snapshots, key=lambda item: item[1].as_posix()):
        target = dest / rel
        payload = src.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if args.check:
            if not target.is_file() or _sha256(target) != digest:
                print(f"STALE {rel}", file=sys.stderr)
                return 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or _sha256(target) != digest:
                target.write_bytes(payload)
                written += 1
        rows.append(f"{rel.as_posix()}\t{_display(src)}\t{digest[:16]}\t{len(payload)}")

    if not args.check:
        index = dest / "index.tsv"
        index.write_text(
            "snapshot\tsource\tsha256\tbytes\n" + "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        print(f"{len(snapshots)} snapshots tracked in {dest} ({written} updated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
