#!/usr/bin/env python3
"""AoI downstream: MP21 is the test set (fold 109), Ridge α up to 10."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dtl.downstream import main

if __name__ == "__main__":
    main("aoi")
