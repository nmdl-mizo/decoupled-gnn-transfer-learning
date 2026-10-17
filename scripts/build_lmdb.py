#!/usr/bin/env python3
"""Build train/valid/test LMDB graphs for the paper splits, from the bundled CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dtl.downstream import SPECS
from dtl.graphs import assign_paper_folds, load_structure_table, write_lmdb


def main() -> None:
    parser = argparse.ArgumentParser(description="Write train/valid/test LMDB files from the CSV")
    parser.add_argument("--dataset", choices=["lic", "aoi"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None, help="Cap each split, for a smoke test")
    parser.add_argument("--seed", type=int, default=3776)
    args = parser.parse_args()

    spec = SPECS[args.dataset]
    frame = assign_paper_folds(args.dataset, load_structure_table(args.dataset, csv_path=args.csv), seed=args.seed)
    masks = {
        "train": (frame["fold"] != 3) & (frame["fold"] != spec.test_fold),
        "valid": frame["fold"] == 3,
        "test": frame["fold"] == spec.test_fold,
    }
    for name, mask in masks.items():
        path = args.output_dir / f"{args.dataset}_{name}.lmdb"
        n = write_lmdb(frame.loc[mask], args.dataset, path, target_col=spec.target, max_rows=args.max_rows)
        print(f"{name}: {n} graphs -> {path}")


if __name__ == "__main__":
    main()
