#!/usr/bin/env python3
"""LIC downstream: E_form (eV/atom). Default split is random/random (fold 104).

Other splits: --preset p2_grp_grp | p2_tar_rnd | p2_targrp_grp
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dtl.downstream import main

if __name__ == "__main__":
    main("lic")
