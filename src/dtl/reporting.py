"""Figures and tables for splits, parity plots, and regression metrics (p1 notebooks equivalent)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .metrics import calc_loss, predict_ensemble


def save_train_test_histogram_and_describe(
    y_raw: np.ndarray,
    fold: np.ndarray,
    *,
    test_fold: int,
    target_col: str,
    report_dir: Path,
    basename: str = "split_train_valid_vs_test_target",
    axis_label: str | None = None,
) -> None:
    """
    Histogram and describe table at the first split:
    ``1_train-valid`` (fold != test_fold) vs ``2_test`` (fold == test_fold).
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    y_raw = np.asarray(y_raw, dtype=np.float64)
    fold = np.asarray(fold)
    if len(y_raw) != len(fold):
        raise ValueError("y_raw and fold length mismatch")

    split_tt = np.where(fold != test_fold, "1_train-valid", "2_test")
    pdf = pd.DataFrame({"split_train_test": split_tt, target_col: y_raw})

    order = ["1_train-valid", "2_test"]
    colors = {"1_train-valid": "C0", "2_test": "C1"}

    plt.figure(figsize=(8, 5))
    for name in order:
        sub = pdf.loc[pdf["split_train_test"] == name, target_col]
        if len(sub) == 0:
            continue
        nb = min(80, max(10, len(sub) // 20))
        plt.hist(
            sub.values,
            bins=nb,
            alpha=0.45,
            color=colors.get(name, "gray"),
            label=f"{name} (n={len(sub)})",
        )
    plt.xlabel(axis_label or target_col)
    plt.ylabel("count")
    plt.title("Target distribution: train-valid vs test (before target standardization)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / f"{basename}_histogram.png", dpi=150)
    plt.close()

    desc = pdf.groupby("split_train_test")[target_col].describe()
    desc = desc.reindex([s for s in order if s in desc.index])
    desc.to_csv(report_dir / f"{basename}_describe.csv")
    with open(report_dir / f"{basename}_describe.txt", "w", encoding="utf-8") as f:
        f.write(desc.to_string())


def make_parity_arrays(
    models: list,
    feat: np.ndarray,
    true_z: np.ndarray,
    fold: np.ndarray,
    *,
    val_folds: list[int],
    test_fold: int,
    loss: str,
    clip: bool,
    clip_min: float,
    clip_max: float,
    train_y_mean: float,
    train_y_std: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """
    Two series, ``oof`` and ``test``.

    - **oof**: every non-test row, predicted by the model that held that fold out.
    - **test**: the mean of the CV models (RMSE).
    """
    feat = np.asarray(feat)
    true_z = np.asarray(true_z, dtype=np.float64)
    f = np.asarray(fold)

    idx_map = {vf: i for i, vf in enumerate(val_folds)}
    pred_z = np.empty(len(feat), dtype=np.float64)
    pred_z[:] = np.nan

    for vf, mi in idx_map.items():
        m = f == vf
        if np.any(m):
            pred_z[m] = models[mi].predict(feat[m])

    te = f == test_fold
    if np.any(te):
        pred_z[te] = predict_ensemble(
            models,
            feat[te],
            loss=loss,
            clip=clip,
            clip_min=clip_min,
            clip_max=clip_max,
            train_y_mean=train_y_mean,
            train_y_std=train_y_std,
        )

    oof_m = f != test_fold
    true_list: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []
    set_names_out: list[str] = []

    if np.any(oof_m):
        true_list.append(true_z[oof_m])
        pred_list.append(pred_z[oof_m])
        set_names_out.append("oof")
    if np.any(te):
        true_list.append(true_z[te])
        pred_list.append(pred_z[te])
        set_names_out.append("test")

    return true_list, pred_list, set_names_out


def save_parity_plot(
    true_list: list[np.ndarray],
    pred_list: list[np.ndarray],
    set_names: list[str],
    *,
    title: str,
    x_label: str,
    y_label: str,
    standardized: bool,
    train_y_mean: float,
    train_y_std: float,
    training_loss: str,
    out_path: Path,
    s: float = 2.0,
    alpha: float = 0.35,
    figsize: float = 6.0,
) -> None:
    """Scatter vs diagonal. Legend uses ``--loss`` (mae/rmse) via ``calc_loss``, matching the notebook `loss_func` line."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not true_list:
        return

    stacks = []
    for j in range(len(true_list)):
        if standardized:
            stacks.append(true_list[j] * train_y_std + train_y_mean)
            stacks.append(pred_list[j] * train_y_std + train_y_mean)
        else:
            stacks.append(true_list[j])
            stacks.append(pred_list[j])
    concat = np.concatenate(stacks)

    lo = float(concat.min()) - 0.2
    hi = float(concat.max()) + 0.2

    plt.figure(figsize=(figsize, figsize))
    plt.title(title)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().set_aspect("equal", adjustable="box")

    zstd = train_y_std if standardized else 1.0
    for j, sn in enumerate(set_names):
        if standardized:
            a = true_list[j] * train_y_std + train_y_mean
            b = pred_list[j] * train_y_std + train_y_mean
            loss_val = calc_loss(true_list[j], pred_list[j], loss=training_loss, target_train_std=zstd)
        else:
            a = true_list[j]
            b = pred_list[j]
            loss_val = calc_loss(true_list[j], pred_list[j], loss=training_loss, target_train_std=zstd)
        plt.scatter(a, b, label=f"{sn}: {training_loss}={loss_val:.4f}", s=s, alpha=alpha)
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def append_regression_metrics(
    model_name: str,
    subset: str,
    y_true_orig: np.ndarray,
    y_pred_orig: np.ndarray,
    *,
    rows: list[dict],
) -> None:
    y_true_orig = np.asarray(y_true_orig, dtype=np.float64).ravel()
    y_pred_orig = np.asarray(y_pred_orig, dtype=np.float64).ravel()
    if len(y_true_orig) == 0:
        return
    rows.append(
        {
            "model": model_name,
            "subset": subset,
            "n": len(y_true_orig),
            "mae": float(mean_absolute_error(y_true_orig, y_pred_orig)),
            "rmse": float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))),
            "r2": float(r2_score(y_true_orig, y_pred_orig)),
        }
    )


def save_metrics_table(rows: list[dict], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def to_original_space(arr_z: np.ndarray, *, standardized: bool, train_y_mean: float, train_y_std: float) -> np.ndarray:
    arr_z = np.asarray(arr_z, dtype=np.float64)
    if standardized:
        return arr_z * train_y_std + train_y_mean
    return arr_z
