from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from .presets import SplitPreset


def assign_folds(
    df: pd.DataFrame,
    *,
    target: str,
    preset: SplitPreset,
    seed: int,
    test_n_fold: int = 5,
    test_fold: int = 104,
    val_n_fold: int = 4,
    val_fold: int = 3,
    group_for_train_test: str = "Host_composition",
    group_for_train_valid: str = "Host_composition",
    load_splitted_df: bool = False,
    splitted_group_csv: str | None = None,
    splitted_random_csv: str | None = None,
    time_series_split: bool = False,
    identifier: str = "dataset",
    use_loaded_split_as_is: bool = False,
) -> pd.DataFrame:
    """
    Train/test then train/valid, matching p1-* notebooks.
    test_fold default 104 = (test_n_fold - 1) + 100.
    """
    df = df.copy()
    if load_splitted_df:
        if splitted_group_csv is None or splitted_random_csv is None:
            raise ValueError("load_splitted_df requires splitted_group_csv and splitted_random_csv")
        path = splitted_group_csv if preset.use_group_for_train_test else splitted_random_csv
        return pd.read_csv(path, usecols=lambda c: c != "cif")

    df["fold"] = 777

    if time_series_split:
        mask = df[identifier] == "MP21"
        df.loc[mask, "fold"] = test_fold
    elif preset.split_test_by_target:
        if preset.use_group_for_train_test:
            hi = list(df.loc[df[target] > preset.target_upper_limit_train, group_for_train_test].unique())
            lo = list(df.loc[df[target] < preset.target_bottom_limit_train, group_for_train_test].unique())
            groups_out = set(hi + lo)
            mask = df[group_for_train_test].isin(groups_out)
        else:
            mask_upper = df[target] > preset.target_upper_limit_train
            mask_bottom = df[target] < preset.target_bottom_limit_train
            mask = mask_upper | mask_bottom
        df.loc[mask, "fold"] = test_fold
    else:
        if preset.use_group_for_train_test:
            kfold = GroupKFold(n_splits=test_n_fold)
            for i, (_tr, va_idx) in enumerate(kfold.split(df, None, groups=df[group_for_train_test])):
                if i + 100 == test_fold:
                    df.loc[va_idx, "fold"] = test_fold
        else:
            kfold = KFold(n_splits=test_n_fold, shuffle=True, random_state=seed)
            for i, (_tr, va_idx) in enumerate(kfold.split(df, None)):
                if i + 100 == test_fold:
                    df.loc[va_idx, "fold"] = test_fold

    df_test = df.loc[df["fold"] == test_fold].reset_index(drop=True)
    df_train = df.loc[df["fold"] != test_fold].reset_index(drop=True)

    if not use_loaded_split_as_is:
        if preset.use_group_for_train_valid:
            kfold = GroupKFold(n_splits=val_n_fold)
            for i, (_tr, va_idx) in enumerate(kfold.split(df_train, None, groups=df_train[group_for_train_valid])):
                df_train.loc[va_idx, "fold"] = i
        else:
            kfold = KFold(n_splits=val_n_fold, shuffle=True, random_state=seed)
            for i, (_tr, va_idx) in enumerate(kfold.split(df_train, None)):
                df_train.loc[va_idx, "fold"] = i

    out = pd.concat([df_train, df_test]).sort_values("sort").reset_index(drop=True)
    out.loc[(out["fold"] != val_fold) & (out["fold"] != test_fold), "split"] = "1_train"
    out.loc[out["fold"] == val_fold, "split"] = "2_valid"
    out.loc[out["fold"] == test_fold, "split"] = "3_test"
    return out


def scale_features_trainval_excluding_test(all_feature, fold, test_fold: int, *, use_minmax: bool = True):
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    fold = np.asarray(fold)
    mask = fold != test_fold
    scaler = MinMaxScaler() if use_minmax else StandardScaler()
    scaler.fit(all_feature[mask])
    return scaler.transform(all_feature), scaler


def standardize_target_excluding_test(all_true, fold, test_fold: int):
    import numpy as np

    fold = np.asarray(fold)
    mask = fold != test_fold
    mean = float(all_true[mask].mean())
    std = float(all_true[mask].std())
    if std < 1e-12:
        std = 1.0
    return (all_true - mean) / std, mean, std
