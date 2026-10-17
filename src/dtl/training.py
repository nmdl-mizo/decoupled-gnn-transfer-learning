from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

from .metrics import calc_loss


def _oof_masks(df_fold, val_folds: list[int], test_fold: int):
    for fold in val_folds:
        tr = (df_fold != fold) & (df_fold != test_fold)
        va = df_fold == fold
        yield tr, va


def train_ridge_cv(
    all_feature: np.ndarray,
    all_true_z: np.ndarray,
    df_fold,
    *,
    val_folds: list[int],
    test_fold: int,
    loss: str,
    target_train_std: float,
    tune_alpha: bool,
    ridge_alpha: float,
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    verbose: bool = True,
    patience: int = 0,
) -> tuple[list, float]:
    """OOF tuning then refit per fold (same structure as notebook)."""
    df_fold = np.asarray(df_fold)
    if tune_alpha:
        best_alpha = ridge_alpha
        best_score = 1e18
        # Notebook adds 0.0001 so alpha_max itself is included.
        candidates = np.arange(alpha_min, alpha_max + 0.0001, alpha_step)
        if verbose:
            print(
                f"[ridge] α tuning: {loss} OOF, α ∈ [{alpha_min}, {alpha_max}] "
                f"step {alpha_step} ({len(candidates)} values × {len(val_folds)} fits)"
            )
            if patience > 0:
                print(f"[ridge] early stop: halt after {patience} alphas in a row with no OOF improvement")
            else:
                print("[ridge] early stop off: the alpha grid is searched through the last value")
        # Slice once. The notebook indexes again at every α; the fits themselves are unchanged.
        fold_splits = []
        for tr_m, va_m in _oof_masks(df_fold, val_folds, test_fold):
            fold_splits.append(
                (
                    np.ascontiguousarray(all_feature[tr_m]),
                    np.ascontiguousarray(all_true_z[tr_m]),
                    np.ascontiguousarray(all_feature[va_m]),
                    np.ascontiguousarray(all_true_z[va_m]),
                    int(va_m.sum()),
                )
            )
        stale = 0
        for alpha in candidates:
            oof_loss = 0.0
            n_va = 0
            for xf, yf, xv, yv, n_fold in fold_splits:
                model = Ridge(alpha=alpha) if alpha != 0 else LinearRegression()
                model.fit(xf, yf)
                pred = model.predict(xv)
                part = calc_loss(yv, pred, loss=loss, target_train_std=target_train_std)
                oof_loss += part * n_fold
                n_va += n_fold
            score = oof_loss / max(n_va, 1)
            if score < best_score:
                best_score = score
                best_alpha = float(alpha)
                stale = 0
            else:
                stale += 1
            if verbose:
                print(f"[ridge] α={float(alpha):.2f}: score={score:.4f}")
            if patience > 0 and stale >= patience:
                print(
                    f"[ridge] OOF did not improve for {patience} alphas; "
                    f"stopping the search at α={float(alpha):.2f}"
                )
                break
        ridge_alpha = best_alpha
        if verbose:
            print(f"[ridge] best: α={best_alpha:.2f}, score={best_score:.4f}")
            print(f"[ridge] α set to {ridge_alpha:.2f}")
    elif verbose:
        print(f"[ridge] no search; using α={ridge_alpha:.2f}")

    models: list = []
    oof_pred_list = []
    oof_true_list = []
    for tr_m, va_m in _oof_masks(df_fold, val_folds, test_fold):
        xf, yf = all_feature[tr_m], all_true_z[tr_m]
        xv, yv = all_feature[va_m], all_true_z[va_m]
        model = Ridge(alpha=ridge_alpha) if ridge_alpha != 0 else LinearRegression()
        model.fit(xf, yf)
        pred = model.predict(xv)
        models.append(model)
        oof_pred_list.append(pred)
        oof_true_list.append(yv)
    oof_pred = np.concatenate(oof_pred_list)
    oof_true = np.concatenate(oof_true_list)
    oof_score = calc_loss(oof_true, oof_pred, loss=loss, target_train_std=target_train_std)
    return models, oof_score


def save_models(models: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(models, f)


def load_models(path: Path) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def train_svr_cv(
    all_feature: np.ndarray,
    all_true_z: np.ndarray,
    df_fold,
    *,
    val_folds: list[int],
    test_fold: int,
    loss: str,
    target_train_std: float,
    tune: bool,
    svr_c: float,
    svr_epsilon: float,
    c_num: int,
    c_min: float,
    c_max: float,
    eps_min: float,
    eps_max: float,
    eps_step: float,
    use_rapids: bool,
    verbose: bool = True,
) -> tuple[list, float]:
    used_cuml = False
    if use_rapids:
        try:
            from cuml.svm import SVR as CuSVR  # type: ignore

            SvrCls = CuSVR
            used_cuml = True
        except Exception:
            SvrCls = SVR
    else:
        SvrCls = SVR

    df_fold = np.asarray(df_fold)
    if verbose and use_rapids and not used_cuml:
        print(
            "[svr] warning: cuml.svm.SVR (GPU) could not be imported. "
            "Continuing with sklearn.svm.SVR (CPU). Install RAPIDS, or pass --cpu-svr if that fallback is intentional."
        )
    if verbose and not use_rapids and len(all_feature) > 1500:
        print(
            "[svr] --cpu-svr selects sklearn (CPU). "
            "This is slow at this sample size and feature dimension. Leave the default when a GPU and cuML are available."
        )

    def eval_params(c: float, eps: float) -> float:
        oof_loss = 0.0
        n_va = 0
        for tr_m, va_m in _oof_masks(df_fold, val_folds, test_fold):
            xf, yf = all_feature[tr_m], all_true_z[tr_m]
            xv, yv = all_feature[va_m], all_true_z[va_m]
            model = SvrCls(C=c, epsilon=eps)
            model.fit(xf, yf)
            pred = model.predict(xv)
            part = calc_loss(yv, pred, loss=loss, target_train_std=target_train_std)
            oof_loss += part * int(va_m.sum())
            n_va += int(va_m.sum())
        return oof_loss / max(n_va, 1)

    if tune:
        candidates_c = np.geomspace(c_min, c_max, num=c_num)
        candidates_eps = np.arange(eps_min, eps_max + 0.0001, eps_step)
        if verbose:
            print(f"[svr] C・ε tuning: {loss} OOF")
            print(f"[svr] C candidates: {candidates_c}")
            print(f"[svr] epsilon candidates: {candidates_eps}")
        best = (svr_c, svr_epsilon, 1e18)
        for c in candidates_c:
            if verbose:
                print(f"[svr] ----- C={c:g} -----")
            for eps in candidates_eps:
                sc = eval_params(float(c), float(eps))
                if sc < best[2]:
                    best = (float(c), float(eps), sc)
                if verbose:
                    print(f"[svr] ε={float(eps):.2f}: score={sc:.4f}")
        svr_c, svr_epsilon = best[0], best[1]
        if verbose:
            print("[svr] " + "-" * 20)
            print(f"[svr] best: C={svr_c:.2f}, ε={svr_epsilon:.2f}, score={best[2]:.4f}")
            print(f"[svr] C set to {svr_c:.2f}, epsilon set to {svr_epsilon:.2f}")
    elif verbose:
        print(f"[svr] no search; using C={svr_c:.2f}, epsilon={svr_epsilon:.2f}")

    models: list = []
    oof_pred_list = []
    oof_true_list = []
    for tr_m, va_m in _oof_masks(df_fold, val_folds, test_fold):
        xf, yf = all_feature[tr_m], all_true_z[tr_m]
        xv, yv = all_feature[va_m], all_true_z[va_m]
        model = SvrCls(C=svr_c, epsilon=svr_epsilon)
        model.fit(xf, yf)
        pred = model.predict(xv)
        models.append(model)
        oof_pred_list.append(pred)
        oof_true_list.append(yv)
    oof_pred = np.concatenate(oof_pred_list)
    oof_true = np.concatenate(oof_true_list)
    score = calc_loss(oof_true, oof_pred, loss=loss, target_train_std=target_train_std)
    return models, score
