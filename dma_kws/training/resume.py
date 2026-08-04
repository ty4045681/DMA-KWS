"""Resume-from-checkpoint path resolution for training entrypoints."""

from __future__ import annotations

from pathlib import Path


def resolve_resume_path(resume_from: str, checkpoint_dir: Path) -> Path | None:
    """Resolve the checkpoint path to resume training from.

    Returns ``None`` when ``resume_from`` is empty/falsy. When ``resume_from``
    is ``"last"`` the resolved path is ``checkpoint_dir / "last.ckpt"``;
    otherwise ``resume_from`` is treated as an explicit path. In both non-None
    cases a missing checkpoint raises ``SystemExit``.
    """
    if not resume_from:
        return None

    if resume_from == "last":
        path = Path(checkpoint_dir) / "last.ckpt"
    else:
        path = Path(resume_from)

    if not path.exists():
        raise SystemExit(f"Resume checkpoint not found: {path}")

    return path


def resolve_versioned_resume_path(
    resume_from: str,
    checkpoint_root: Path,
    run_name: str,
) -> Path | None:
    """Resolve resume input after checkpoints became run/version scoped.

    Explicit paths retain the exact behavior of :func:`resolve_resume_path`.
    ``last`` selects the newest numeric ``version_N/last.ckpt`` below this
    ``run_name``; a legacy root-level ``last.ckpt`` remains a fallback for runs
    created before version-scoped checkpoint directories were introduced.
    """
    checkpoint_root = Path(checkpoint_root)
    if resume_from != "last":
        return resolve_resume_path(resume_from, checkpoint_root)

    run_root = checkpoint_root / run_name
    candidates: list[tuple[int, Path]] = []
    if run_root.is_dir():
        for child in run_root.iterdir():
            if not child.is_dir() or not child.name.startswith("version_"):
                continue
            suffix = child.name.removeprefix("version_")
            candidate = child / "last.ckpt"
            if suffix.isdigit() and candidate.is_file():
                candidates.append((int(suffix), candidate))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    return resolve_resume_path("last", checkpoint_root)
