"""CSV rows to OCP crystal graphs. Used by the LMDB builder and the embedding script."""

from __future__ import annotations

import ast
import json
import pickle
import tempfile
from pathlib import Path

import lmdb
import pandas as pd
from ase.io import read
from ocpmodels.preprocessing import AtomsToGraphs
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from .downstream import ROOT, SPECS
from .presets import get_preset
from .splitting import assign_folds
from .targets import add_ef_per_atom_column


def load_structure_table(dataset: str, *, csv_path: Path | None = None, root: Path = ROOT) -> pd.DataFrame:
    """Rows needed to rebuild graphs. Does not read the bundled embeddings."""
    spec = SPECS[dataset]
    path = Path(csv_path) if csv_path is not None else root / spec.csv_name
    if dataset == "lic":
        frame = pd.read_csv(path, usecols=["sort", "cif", "Host_composition", "Ef (eV/f.u.)"])
        frame = add_ef_per_atom_column(frame)
        return frame
    frame = pd.read_csv(path, usecols=["structure_as_dict", spec.target, "dataset"])
    frame["sort"] = frame.index + 1
    return frame


def assign_paper_folds(dataset: str, frame: pd.DataFrame, *, seed: int = 3776) -> pd.DataFrame:
    spec = SPECS[dataset]
    return assign_folds(
        frame,
        target=spec.target,
        preset=get_preset(spec.preset),
        seed=seed,
        test_n_fold=5 if dataset == "lic" else 10,
        test_fold=spec.test_fold,
        val_n_fold=4,
        val_fold=3,
        time_series_split=spec.time_series_split,
        identifier="dataset",
    )


def _atoms_from_cif(cif_text: str):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as handle:
        handle.write(str(cif_text))
        tmp = handle.name
    try:
        return read(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _atoms_from_structure_dict(text: str):
    if isinstance(text, dict):
        payload = text
    else:
        raw = str(text)
        try:
            payload = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            payload = json.loads(raw)
    return AseAtomsAdaptor.get_atoms(Structure.from_dict(payload))


def atoms_for_row(dataset: str, row: pd.Series):
    if dataset == "lic":
        return _atoms_from_cif(row["cif"])
    return _atoms_from_structure_dict(row["structure_as_dict"])


def write_lmdb(
    frame: pd.DataFrame,
    dataset: str,
    path: Path,
    *,
    target_col: str,
    max_rows: int | None = None,
) -> int:
    """Write one LMDB. Rows are stored in ascending ``sort`` order."""
    subset = frame.sort_values("sort")
    if max_rows is not None:
        subset = subset.head(max_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    lock = Path(str(path) + "-lock")
    if lock.exists():
        lock.unlink()
    converter = AtomsToGraphs(max_neigh=50, radius=6, r_energy=False, r_forces=False, r_distances=False)
    db = lmdb.open(str(path), map_size=1099511627776 * 2, subdir=False, meminit=False, map_async=True)
    for counter, (_, row) in enumerate(tqdm(subset.iterrows(), total=len(subset), desc=path.name)):
        graph = converter.convert(atoms_for_row(dataset, row))
        if isinstance(graph, list):
            graph = graph[0]
        graph.y = float(row[target_col])
        graph.sort = int(row["sort"])
        with db.begin(write=True) as txn:
            txn.put(f"{counter}".encode("ascii"), pickle.dumps(graph, protocol=-1))
    with db.begin(write=True) as txn:
        txn.put(b"length", pickle.dumps(len(subset), protocol=-1))
    db.sync()
    db.close()
    return len(subset)
