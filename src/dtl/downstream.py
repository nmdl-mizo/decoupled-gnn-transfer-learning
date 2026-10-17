"""Shared LIC / AoI downstream run (stacked GNN embeddings + Ridge / SVR)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import load_stacked_embeddings, slice_arrays_by_df_sort
from .metrics import calc_loss, predict_ensemble
from .presets import get_preset
from .reporting import (
    append_regression_metrics,
    make_parity_arrays,
    save_metrics_table,
    save_parity_plot,
    save_train_test_histogram_and_describe,
    to_original_space,
)
from .splitting import assign_folds, scale_features_trainval_excluding_test, standardize_target_excluding_test
from .targets import add_ef_per_atom_column
from .training import load_models, save_models, train_ridge_cv, train_svr_cv

ROOT = Path(__file__).resolve().parents[2]

_RIDGE_ALPHA_MIN = 0.0
_RIDGE_ALPHA_MAX = 10.0
_RIDGE_ALPHA_STEP = 0.05
# Paper: C is 10–1000 in 5 geometric steps; ε is 0–0.06 in steps of 0.02.
_SVR_C_NUM = 5
_SVR_C_MIN = 10.0
_SVR_C_MAX = 1000.0
_SVR_EPS_MIN = 0.0
_SVR_EPS_MAX = 0.06
_SVR_EPS_STEP = 0.02
_VAL_FOLDS = [0, 1, 2, 3]
_VAL_FOLD = 3
# Notebook axis text. The CSV column stays the stored name; figures use this label
# so formation energy is not written as Ef next to the Fermi energy.
_TARGET_TEXT = "E_form (eV/atom)"
_TARGET_MATH = r"$E\mathrm{_{form}}$ (eV/atom)"
_PARITY_X = r"$E\mathrm{_{form}}$ from DFT calculation (eV/atom)"
_PARITY_Y = r"$E\mathrm{_{form}}$ from prediction (eV/atom)"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_name: str
    embeddings_dir: str
    data_name: str
    naming: str
    num_data: int | None
    target: str
    derived_target: str
    preset: str
    test_fold: int
    time_series_split: bool
    alpha_max: float
    output_base: str
    add_sort_from_index: bool


SPECS: dict[str, DatasetSpec] = {
    "lic": DatasetSpec(
        name="lic",
        csv_name="data/lic/LC_database_rev_with_cif.csv",
        embeddings_dir="data/lic/embeddings",
        data_name="lc_data",
        naming="lic",
        num_data=10000,  # legacy stem in shipped LIC npy names; not a row limit  # filename token only; the LIC table has 9024 rows
        target="Ef (eV/atom)",
        derived_target="ef_per_atom",
        preset="p2_rnd_rnd",
        test_fold=104,
        time_series_split=False,
        alpha_max=_RIDGE_ALPHA_MAX,
        output_base="p2-1_per_atom_rnd_rnd",
        add_sort_from_index=False,
    ),
    "aoi": DatasetSpec(
        name="aoi",
        csv_name="data/aoi/AoI_with_timestamp.csv",
        embeddings_dir="data/aoi/embeddings",
        data_name="mp_aoi",
        naming="aoi",
        num_data=None,
        target="formation_energy_per_atom",
        derived_target="none",
        preset="p2_rnd_rnd",
        test_fold=109,
        time_series_split=True,
        alpha_max=_RIDGE_ALPHA_MAX,
        output_base="3-train_test",
        add_sort_from_index=True,
    ),
}


@dataclass
class Prepared:
    df: pd.DataFrame
    features: np.ndarray
    true_z: np.ndarray
    y_raw: np.ndarray
    train_y_mean: float
    train_y_std: float
    test_fold: int


def prepare(
    spec: DatasetSpec,
    *,
    root: Path = ROOT,
    seed: int = 3776,
    csv_path: Path | None = None,
    embeddings_dir: Path | None = None,
    preset_name: str | None = None,
) -> Prepared:
    csv_path = Path(csv_path) if csv_path is not None else root / spec.csv_name
    emb_dir = Path(embeddings_dir) if embeddings_dir is not None else root / spec.embeddings_dir
    preset = get_preset(preset_name or spec.preset)

    if spec.name == "lic":
        df = pd.read_csv(csv_path, usecols=lambda c: c != "cif")
        df = add_ef_per_atom_column(df)
        keep = ~df[spec.target].isna()
        if not bool(np.all(keep)):
            df = df.loc[keep].sort_values("sort").reset_index(drop=True)
    else:
        df = pd.read_csv(csv_path, usecols=[spec.target, "dataset"])
        if spec.add_sort_from_index:
            df["sort"] = df.index + 1

    df = assign_folds(
        df,
        target=spec.target,
        preset=preset,
        seed=seed,
        test_n_fold=5 if spec.name == "lic" else 10,
        test_fold=spec.test_fold,
        val_n_fold=4,
        val_fold=_VAL_FOLD,
        time_series_split=spec.time_series_split,
        identifier="dataset",
    )

    feat, sort_idx = load_stacked_embeddings(
        emb_dir,
        model_name="CGCNN",
        second_model_name="schnet_mean",
        third_model_name="DimeNetPlusPlus_multi_mean",
        data_name=spec.data_name,
        num_data=spec.num_data,
        naming=spec.naming,
    )
    if len(df) != len(sort_idx):
        feat, sort_idx = slice_arrays_by_df_sort(df, feat, sort_idx)
    if int((df["sort"].to_numpy() == np.asarray(sort_idx).ravel()).sum()) != len(df):
        raise RuntimeError("CSV sort does not match embedding sort.npy")

    y_raw = df[spec.target].to_numpy(dtype=np.float64)
    fold = df["fold"].to_numpy()
    true_z, mean, std = standardize_target_excluding_test(y_raw, fold, spec.test_fold)
    feat, _scaler = scale_features_trainval_excluding_test(feat, fold, spec.test_fold)
    return Prepared(
        df=df,
        features=feat,
        true_z=true_z,
        y_raw=y_raw,
        train_y_mean=mean,
        train_y_std=std,
        test_fold=spec.test_fold,
    )


def regression_rows(model_name: str, models: list, prepared: Prepared, *, loss: str = "rmse") -> list[dict]:
    fold = prepared.df["fold"].to_numpy()
    true_list, pred_list, names = make_parity_arrays(
        models,
        prepared.features,
        prepared.true_z,
        fold,
        val_folds=_VAL_FOLDS,
        test_fold=prepared.test_fold,
        loss=loss,
        clip=False,
        clip_min=-99999.0,
        clip_max=99999.0,
        train_y_mean=prepared.train_y_mean,
        train_y_std=prepared.train_y_std,
    )
    rows: list[dict] = []
    for true_z, pred_z, subset in zip(true_list, pred_list, names):
        y = to_original_space(
            true_z,
            standardized=True,
            train_y_mean=prepared.train_y_mean,
            train_y_std=prepared.train_y_std,
        )
        yhat = to_original_space(
            pred_z,
            standardized=True,
            train_y_mean=prepared.train_y_mean,
            train_y_std=prepared.train_y_std,
        )
        append_regression_metrics(model_name, subset, y, yhat, rows=rows)
        rows[-1]["loss_rmse"] = calc_loss(
            true_z, pred_z, loss="rmse", target_train_std=prepared.train_y_std
        )
    return rows


def _print_config(spec: DatasetSpec, args: argparse.Namespace, preset_name: str) -> None:
    n_alpha = len(np.arange(_RIDGE_ALPHA_MIN, spec.alpha_max + 0.0001, _RIDGE_ALPHA_STEP))
    cand_c = np.geomspace(_SVR_C_MIN, _SVR_C_MAX, num=_SVR_C_NUM)
    cand_eps = np.arange(_SVR_EPS_MIN, _SVR_EPS_MAX + 0.0001, _SVR_EPS_STEP)
    print("=" * 72)
    print(f"downstream — {spec.name}")
    print("=" * 72)
    print(f"csv: {args.database_csv}")
    print(f"embeddings: {args.embeddings_dir}")
    print(f"output: {args.output_dir}")
    print(f"target: {_TARGET_TEXT}  column={spec.target}  derived={spec.derived_target}  seed={args.seed}")
    print(
        f"split: time_series={spec.time_series_split}  test_fold={spec.test_fold}  "
        f"val_folds={_VAL_FOLDS}  preset={preset_name}"
    )
    print(f"stack: CGCNN + schnet_mean + DimeNetPlusPlus_multi_mean  naming={spec.naming}")
    print(f"standardize target and MinMax features using non-test rows only")
    print(
        f"Ridge α ∈ [{_RIDGE_ALPHA_MIN}, {spec.alpha_max}] step {_RIDGE_ALPHA_STEP} "
        f"({n_alpha} values × 4 fits)"
    )
    if args.ridge_patience > 0:
        print(f"Ridge early stop: halt after {args.ridge_patience} alphas in a row with no OOF improvement")
    else:
        print("Ridge early stop: off (search the full grid, as in the paper)")
    print(f"SVR C={np.array2string(cand_c, precision=4)}  ε={np.array2string(cand_eps, precision=2)}")
    print(f"SVR backend: {'sklearn CPU' if args.cpu_svr else 'cuml GPU (fallback to sklearn if unavailable)'}")
    print("=" * 72)


def build_parser(default_dataset: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper downstream: stacked GNN embeddings + Ridge/SVR")
    p.add_argument("--dataset", choices=sorted(SPECS), default=default_dataset)
    p.add_argument("--database-csv", type=Path, default=None)
    p.add_argument("--embeddings-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=3776)
    p.add_argument("--skip-svr", action="store_true")
    p.add_argument("--cpu-svr", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-report", action="store_true")
    p.add_argument(
        "--ridge-patience",
        type=int,
        default=0,
        help="Stop the Ridge alpha search after this many consecutive alphas with no OOF improvement. 0 searches the full grid.",
    )
    p.add_argument(
        "--preset",
        choices=["p2_rnd_rnd", "p2_grp_grp", "p2_tar_rnd", "p2_targrp_grp"],
        default=None,
        help="LIC split. Default p2_rnd_rnd. The four names are random/random, group/group, target/random, and target+group/group.",
    )
    p.add_argument("--load-ridge", type=Path, default=None, help="Score this pickle instead of retraining Ridge")
    p.add_argument("--load-svr", type=Path, default=None, help="Score this pickle instead of retraining SVR")
    return p


def main(default_dataset: str, argv: list[str] | None = None) -> None:
    args = build_parser(default_dataset).parse_args(argv)
    spec = SPECS[args.dataset]
    root = ROOT
    if args.database_csv is None:
        args.database_csv = root / spec.csv_name
    if args.embeddings_dir is None:
        args.embeddings_dir = root / spec.embeddings_dir
    if args.output_dir is None:
        args.output_dir = root / "outputs" / spec.name

    if args.preset and spec.name != "lic":
        raise SystemExit("--preset is for the LIC dataset")
    preset_name = args.preset or spec.preset
    output_base = get_preset(preset_name).default_base_name if spec.name == "lic" else spec.output_base

    _print_config(spec, args, preset_name)
    prepared = prepare(
        spec,
        root=root,
        seed=args.seed,
        csv_path=args.database_csv,
        embeddings_dir=args.embeddings_dir,
        preset_name=preset_name,
    )

    n_train = int((prepared.df["fold"] != spec.test_fold).sum())
    n_test = int((prepared.df["fold"] == spec.test_fold).sum())
    print(
        f"aligned: n={len(prepared.df)}  train-valid={n_train}  test={n_test}  "
        f"n_features={prepared.features.shape[1]}"
    )
    print(f"target mean={prepared.train_y_mean:.6f}  std={prepared.train_y_std:.6f} (non-test)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = out_dir / "report"
    metric_rows: list[dict] = []

    if not args.no_report:
        save_train_test_histogram_and_describe(
            prepared.y_raw,
            prepared.df["fold"].to_numpy(),
            test_fold=spec.test_fold,
            target_col=_TARGET_TEXT,
            report_dir=report_dir,
            basename="split_train_valid_vs_test_target",
            axis_label=_TARGET_MATH,
        )

    def _reports(label: str, models: list) -> None:
        rows = regression_rows(label.lower(), models, prepared)
        metric_rows.extend(rows)
        for row in rows:
            print(
                f"[{label}] {row['subset']}: n={row['n']}  "
                f"MAE={row['mae']:.4f}  RMSE={row['rmse']:.4f}  R2={row['r2']:.4f}"
            )
        if args.no_report:
            return
        true_list, pred_list, names = make_parity_arrays(
            models,
            prepared.features,
            prepared.true_z,
            prepared.df["fold"].to_numpy(),
            val_folds=_VAL_FOLDS,
            test_fold=spec.test_fold,
            loss="rmse",
            clip=False,
            clip_min=-99999.0,
            clip_max=99999.0,
            train_y_mean=prepared.train_y_mean,
            train_y_std=prepared.train_y_std,
        )
        save_parity_plot(
            true_list,
            pred_list,
            names,
            title=f"{output_base} — {label}",
            x_label=_PARITY_X,
            y_label=_PARITY_Y,
            standardized=True,
            train_y_mean=prepared.train_y_mean,
            train_y_std=prepared.train_y_std,
            training_loss="rmse",
            out_path=report_dir / f"parity_{label.lower()}.png",
        )

    fold = prepared.df["fold"].to_numpy()
    if args.load_ridge:
        ridge_models = load_models(args.load_ridge)
        print(f"[ridge] loaded {args.load_ridge}")
    else:
        ridge_models, oof_ridge = train_ridge_cv(
            prepared.features,
            prepared.true_z,
            fold,
            val_folds=_VAL_FOLDS,
            test_fold=spec.test_fold,
            loss="rmse",
            target_train_std=prepared.train_y_std,
            tune_alpha=True,
            ridge_alpha=1.0,
            alpha_min=_RIDGE_ALPHA_MIN,
            alpha_max=spec.alpha_max,
            alpha_step=_RIDGE_ALPHA_STEP,
            verbose=not args.quiet,
            patience=args.ridge_patience,
        )
        save_models(ridge_models, out_dir / f"{output_base}_ridge.pkl")
        chosen = getattr(ridge_models[0], "alpha", 0.0)
        print(f"[ridge] selected α={float(chosen):.2f}  oof rmse={oof_ridge:.6f}")
    _reports("Ridge", ridge_models)

    if args.load_svr:
        svr_models = load_models(args.load_svr)
        print(f"[svr] loaded {args.load_svr}")
        _reports("SVR", svr_models)
    elif not args.skip_svr:
        svr_models, oof_svr = train_svr_cv(
            prepared.features,
            prepared.true_z,
            fold,
            val_folds=_VAL_FOLDS,
            test_fold=spec.test_fold,
            loss="rmse",
            target_train_std=prepared.train_y_std,
            tune=True,
            svr_c=1.0,
            svr_epsilon=0.1,
            c_num=_SVR_C_NUM,
            c_min=_SVR_C_MIN,
            c_max=_SVR_C_MAX,
            eps_min=_SVR_EPS_MIN,
            eps_max=_SVR_EPS_MAX,
            eps_step=_SVR_EPS_STEP,
            use_rapids=not args.cpu_svr,
            verbose=not args.quiet,
        )
        save_models(svr_models, out_dir / f"{output_base}_svr.pkl")
        print(f"[svr] oof rmse (tuning space): {oof_svr:.6f}")
        _reports("SVR", svr_models)

    if metric_rows and not args.no_report:
        save_metrics_table(metric_rows, report_dir / "regression_metrics.csv")
        print(pd.DataFrame(metric_rows).to_string(index=False))
