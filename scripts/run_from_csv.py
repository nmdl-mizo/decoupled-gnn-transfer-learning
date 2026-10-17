#!/usr/bin/env python3
"""CSV to LMDB to GNN embeddings to Ridge/SVR, for one dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SCRIPTS = Path(__file__).resolve().parent
for _path in (_SCRIPTS, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dtl.downstream import main as downstream_main
from extract_gnn_features import PAPER_MODELS, extract


def main() -> None:
    parser = argparse.ArgumentParser(description="Run embedding extraction and the downstream models from the CSV")
    parser.add_argument("--dataset", choices=["lic", "aoi"], required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, default=None)
    parser.add_argument("--lmdb", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None, help="Smoke test only. Do not use this for the paper metrics.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--models", nargs="+", choices=PAPER_MODELS, default=list(PAPER_MODELS))
    parser.add_argument("--skip-svr", action="store_true")
    parser.add_argument("--cpu-svr", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--ridge-patience", type=int, default=0)
    args = parser.parse_args()

    embeddings_dir = args.embeddings_dir or (_ROOT / "work" / f"embeddings_{args.dataset}")
    extract(
        args.dataset,
        args.checkpoint_dir,
        embeddings_dir,
        lmdb_path=args.lmdb,
        models=tuple(args.models),
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        device=args.device,
    )
    if args.max_rows is not None:
        print("Skipping downstream: --max-rows writes a partial embedding file, which does not match the CSV.")
        return

    extra = ["--embeddings-dir", str(embeddings_dir), "--output-dir", str(args.output_dir)]
    if args.skip_svr:
        extra.append("--skip-svr")
    if args.cpu_svr:
        extra.append("--cpu-svr")
    if args.quiet:
        extra.append("--quiet")
    if args.no_report:
        extra.append("--no-report")
    if args.ridge_patience:
        extra.extend(["--ridge-patience", str(args.ridge_patience)])
    downstream_main(args.dataset, extra)


if __name__ == "__main__":
    main()
