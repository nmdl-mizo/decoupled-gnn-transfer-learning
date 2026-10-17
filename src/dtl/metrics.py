from __future__ import annotations

import numpy as np


def calc_loss(
    true: np.ndarray,
    pred: np.ndarray,
    *,
    loss: str,
    target_train_std: float,
) -> float:
    """
    true / pred are in the same space (typically standardized target z-scores).
    Multiply by target_train_std to report error in original target units.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if loss == "mae":
        return float(np.mean(np.abs(pred - true)) * target_train_std)
    if loss == "rmse":
        return float(np.sqrt(np.mean(np.square(pred - true))) * target_train_std)
    raise ValueError(loss)


def predict_ensemble(
    models: list,
    feat: np.ndarray,
    *,
    loss: str,
    clip: bool,
    clip_min: float,
    clip_max: float,
    train_y_mean: float,
    train_y_std: float,
) -> np.ndarray:
    preds = [m.predict(feat) for m in models]
    if loss == "mae":
        out = np.median(preds, axis=0)
    else:
        out = np.mean(preds, axis=0)
    if clip:
        lo = clip_min - train_y_mean / train_y_std
        hi = clip_max - train_y_mean / train_y_std
        out = np.clip(out, lo, hi)
    return out
