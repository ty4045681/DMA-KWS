#!/usr/bin/env python3
"""Backward-compatible alias for ``scripts/prepare_stage2_paper.py``."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from scripts import prepare_stage2_paper

    prepare_stage2_paper.main()


if __name__ == "__main__":
    main()
