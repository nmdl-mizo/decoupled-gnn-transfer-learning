#!/usr/bin/env python3
"""
Rebuild the paper GNN embeddings from the bundled CSV.

Writes one LMDB (all rows, in sort order), runs CGCNN, SchNet, and DimeNet++,
and saves the ``emb_x`` / ``sort`` / target arrays the downstream scripts read.
Open Catalyst checkpoints are not bundled. Pass the directory that contains:

  cgcnn_all.pt
  schnet_all_large.pt
  dimenetpp_all.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from ocpmodels.datasets import LmdbDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT / "third_party") not in sys.path:
    sys.path.insert(0, str(_ROOT / "third_party"))

from dtl.downstream import SPECS
from dtl.graphs import load_structure_table, write_lmdb

# Imported for their classes. Each module registers a name inside ocpmodels.
from get_emb_cgcnn import CGCNN
from get_emb_schnet_mean import SchNetWrap
from multi_get_emb_dimenet_plus_plus_mean import DimeNetPlusPlusWrap

PAPER_MODELS = ("CGCNN", "schnet_mean", "DimeNetPlusPlus_multi_mean")
CHECKPOINT_NAMES = {
    "CGCNN": "cgcnn_all.pt",
    "schnet_mean": "schnet_all_large.pt",
    "DimeNetPlusPlus_multi_mean": "dimenetpp_all.pt",
}


def build_model(name: str):
    if name == "CGCNN":
        return CGCNN(
            num_atoms=1,
            bond_feat_dim=100,
            num_targets=1,
            otf_graph=True,
            regress_forces=False,
            atom_embedding_size=512,
            fc_feat_size=128,
            num_fc_layers=3,
            num_graph_conv_layers=3,
            cutoff=6.0,
            num_gaussians=100,
            use_pbc=True,
        )
    if name == "schnet_mean":
        return SchNetWrap(
            num_atoms=1,
            bond_feat_dim=1,
            num_targets=1,
            otf_graph=True,
            regress_forces=False,
            hidden_channels=1024,
            num_filters=256,
            num_interactions=5,
            cutoff=6.0,
            num_gaussians=200,
            use_pbc=True,
        )
    if name == "DimeNetPlusPlus_multi_mean":
        return DimeNetPlusPlusWrap(
            num_atoms=1,
            bond_feat_dim=1,
            num_targets=1,
            otf_graph=True,
            regress_forces=False,
            hidden_channels=192,
            out_emb_channels=192,
            num_blocks=3,
            cutoff=6.0,
            num_radial=6,
            num_spherical=7,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
            use_pbc=True,
        )
    raise ValueError(name)


def _load_state_dict(checkpoint: Path, device: torch.device) -> dict:
    payload = torch.load(checkpoint, map_location=device)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    cleaned = {}
    for key, value in state.items():
        name = key
        for prefix in ("module.module.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        cleaned[name] = value
    return cleaned


def _embedding_from_output(name: str, pred: dict) -> torch.Tensor:
    raw = pred["x"]
    if name.startswith("DimeNetPlusPlus_multi"):
        return torch.cat(raw, dim=1)
    return raw


def _save_model_arrays(spec, model_name: str, output_dir: Path, emb, sort) -> None:
    if spec.naming == "lic":
        stem = f"{model_name}_{spec.data_name}_{spec.num_data}"
    else:
        stem = f"{model_name}_{spec.data_name}"
    np.save(output_dir / f"{stem}_emb_x.npy", emb)
    np.save(output_dir / f"{stem}_sort.npy", sort)
    print(f"saved {model_name}: emb {emb.shape} -> {output_dir}")


def extract(
    dataset: str,
    checkpoint_dir: Path,
    output_dir: Path,
    *,
    csv_path: Path | None = None,
    lmdb_path: Path | None = None,
    models: tuple[str, ...] = PAPER_MODELS,
    max_rows: int | None = None,
    batch_size: int = 16,
    device: str | None = None,
) -> Path:
    spec = SPECS[dataset]
    frame = load_structure_table(dataset, csv_path=csv_path)
    if lmdb_path is None:
        lmdb_path = _ROOT / "work" / "lmdb" / f"{dataset}_all.lmdb"
    n = write_lmdb(frame, dataset, lmdb_path, target_col=spec.target, max_rows=max_rows)
    print(f"LMDB: {n} graphs -> {lmdb_path}")

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset_lmdb = LmdbDataset({"src": str(lmdb_path)})
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        checkpoint = checkpoint_dir / CHECKPOINT_NAMES[model_name]
        if not checkpoint.is_file():
            raise SystemExit(f"Missing checkpoint: {checkpoint}")
        print(f"=== {model_name}  {checkpoint}  device={torch_device} ===")
        model = build_model(model_name)
        model.load_state_dict(_load_state_dict(checkpoint, torch_device))
        model = model.to(torch_device)
        model.eval()

        features, sorts = [], []
        loader = DataLoader(dataset_lmdb, batch_size=batch_size)
        for batch in tqdm(loader, desc=model_name):
            batch = batch.to(torch_device)
            with torch.no_grad():
                pred = model(batch)
            features.append(_embedding_from_output(model_name, pred).cpu())
            sorts.append(batch.sort.detach().cpu().reshape(-1))

        emb = torch.cat(features).numpy()
        sort = torch.cat(sorts).numpy()
        order = np.argsort(sort, kind="mergesort")
        _save_model_arrays(spec, model_name, output_dir, emb[order], sort[order])
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract paper GNN embeddings from the CSV")
    parser.add_argument("--dataset", choices=["lic", "aoi"], required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--lmdb", type=Path, default=None, help="Intermediate LMDB. Default: work/lmdb/<dataset>_all.lmdb")
    parser.add_argument("--models", nargs="+", choices=PAPER_MODELS, default=list(PAPER_MODELS))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda or cpu. Default: cuda when available")
    args = parser.parse_args()
    extract(
        args.dataset,
        args.checkpoint_dir,
        args.output_dir,
        csv_path=args.csv,
        lmdb_path=args.lmdb,
        models=tuple(args.models),
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
