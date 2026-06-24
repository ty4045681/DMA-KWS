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
