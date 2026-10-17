from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_stacked_embeddings(
    export_dir: Path | str,
    *,
    model_name: str,
    second_model_name: str | None,
    third_model_name: str | None,
    data_name: str,
    num_data: int | None = None,
    use_second: bool = True,
    use_third: bool = True,
    naming: str = "lic",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load stacked GNN embeddings.

    naming="lic": ``{model}_{data}_{num}_{emb_x,sort}.npy``
    naming="aoi": ``{model}_{data}_{emb_x,sort}.npy``

    The regression target is read from the CSV, not from these files.
    """
    export_dir = Path(export_dir)

    def stem(model: str) -> str:
        if naming == "lic":
            if num_data is None:
                raise ValueError("naming='lic' requires num_data")
            return f"{model}_{data_name}_{num_data}"
        if naming == "aoi":
            return f"{model}_{data_name}"
        raise ValueError(f"Unknown naming {naming!r}")

    primary = stem(model_name)
    feat = np.load(export_dir / f"{primary}_emb_x.npy")
    sort_idx = np.load(export_dir / f"{primary}_sort.npy")

    if use_second and second_model_name:
        feat = np.concatenate([feat, np.load(export_dir / f"{stem(second_model_name)}_emb_x.npy")], axis=1)
    if use_third and third_model_name:
        feat = np.concatenate([feat, np.load(export_dir / f"{stem(third_model_name)}_emb_x.npy")], axis=1)
    return feat.astype(np.float64), sort_idx


def slice_arrays_by_df_sort(
    df: pd.DataFrame,
    all_feat: np.ndarray,
    all_sort: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep embedding rows whose ``sort`` appears in ``df['sort']`` (same convention as p2-1 row filter).
    ``all_sort`` from npy must be sorted ascending.
    """
    sort_full = np.asarray(all_sort).ravel()
    if sort_full.ndim != 1:
        raise ValueError("sort must be 1d")
    if len(sort_full) >= 2 and not bool(np.all(sort_full[:-1] <= sort_full[1:])):
        raise ValueError("embedding sort.npy must be sorted ascending for alignment")
    want = np.asarray(df["sort"].to_numpy())
    idx = np.searchsorted(sort_full, want)
    if not np.all(sort_full[idx] == want):
        bad = want[sort_full[idx] != want][:5]
        raise RuntimeError(f"sort values in CSV not found in embedding sort.npy (examples: {bad})")
    return all_feat[idx].copy(), sort_full[idx].copy()
