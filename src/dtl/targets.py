"""Post-processed targets (p2-1_per_atom_rnd_rnd style)."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd


def parse_formula(formula: str) -> dict[str, int]:
    """Element counts from a composition string (same regex as p2-1 notebook)."""
    if formula is None or (isinstance(formula, float) and np.isnan(formula)):
        return {}
    formula = str(formula)
    pattern = r"([A-Z][a-z]*)(\d*)"
    element_counts: defaultdict[str, int] = defaultdict(int)
    for element, count in re.findall(pattern, formula):
        n = int(count) if count else 1
        element_counts[element] += n
    return dict(element_counts)


def count_atoms(formula: str) -> int:
    return int(sum(parse_formula(formula).values()))


def add_ef_per_atom_column(
    df: pd.DataFrame,
    *,
    host_col: str = "Host_composition",
    ef_fu_col: str = "Ef (eV/f.u.)",
    out_col: str = "Ef (eV/atom)",
) -> pd.DataFrame:
    """
    p2-1: ``atom_counts = count_atoms(Host_composition) + 1``,
    ``Ef (eV/atom) = Ef (eV/f.u.) / atom_counts``.
    """
    for c in (host_col, ef_fu_col):
        if c not in df.columns:
            raise ValueError(f"Missing column {c!r} for ef_per_atom target")
    ac = df[host_col].map(count_atoms).astype(np.float64) + 1.0
    df = df.copy()
    df["atom_counts"] = ac.astype(np.int64)
    df[out_col] = df[ef_fu_col].astype(np.float64) / df["atom_counts"].replace(0, np.nan)
    return df


def stability_keep_mask(
    df: pd.DataFrame,
    *,
    column: str,
    threshold: float,
    lte: bool,
) -> pd.Series:
    """Notebook: if stability_th_max then keep column <= threshold, else keep column >= threshold."""
    if column not in df.columns:
        raise ValueError(f"Missing stability column {column!r}")
    s = df[column]
    if lte:
        return s <= threshold
    return s >= threshold
