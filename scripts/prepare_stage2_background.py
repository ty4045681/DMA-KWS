#!/usr/bin/env python3
"""Pre-generate the Stage II background fbank crop cache."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.data_prep.stage2_background import main


if __name__ == "__main__":
    main()
