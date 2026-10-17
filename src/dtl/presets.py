from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SplitPresetName = Literal[
    "rnd_rnd",
    "grp_grp",
    "tar_rnd",
    "targrp_grp",
    "p2_rnd_rnd",
    "p2_grp_grp",
    "p2_tar_rnd",
    "p2_targrp_grp",
]


@dataclass(frozen=True)
class SplitPreset:
    """Matches the four p1-* notebooks (only flags that differ)."""

    name: str
    use_group_for_train_test: bool
    use_group_for_train_valid: bool
    split_test_by_target: bool
    target_upper_limit_train: float
    target_bottom_limit_train: float
    default_base_name: str


PRESETS: dict[str, SplitPreset] = {
    "rnd_rnd": SplitPreset(
        name="rnd_rnd",
        use_group_for_train_test=False,
        use_group_for_train_valid=False,
        split_test_by_target=False,
        target_upper_limit_train=3.0,
        target_bottom_limit_train=-11.0,
        default_base_name="p1-1_save_models_rnd_rnd",
    ),
    "grp_grp": SplitPreset(
        name="grp_grp",
        use_group_for_train_test=True,
        use_group_for_train_valid=True,
        split_test_by_target=False,
        target_upper_limit_train=3.0,
        target_bottom_limit_train=-11.0,
        default_base_name="p1-2_save_models_grp_grp",
    ),
    "tar_rnd": SplitPreset(
        name="tar_rnd",
        use_group_for_train_test=False,
        use_group_for_train_valid=False,
        split_test_by_target=True,
        target_upper_limit_train=2.0,
        target_bottom_limit_train=-9.0,
        default_base_name="p1-3_save_models_tar_rnd",
    ),
    "targrp_grp": SplitPreset(
        name="targrp_grp",
        use_group_for_train_test=True,
        use_group_for_train_valid=True,
        split_test_by_target=True,
        target_upper_limit_train=3.0,
        target_bottom_limit_train=-11.0,
        default_base_name="p1-4_save_models_targrp_grp",
    ),
    # p2-1 … p2-4. Active thresholds are those left uncommented in the p2 notebooks.
    # E_form on this table spans about -3.3 to 0.9 eV/atom, so the commented
    # cuts at ±2–11 eV/atom leave the test set empty.
    "p2_rnd_rnd": SplitPreset(
        name="p2_rnd_rnd",
        use_group_for_train_test=False,
        use_group_for_train_valid=False,
        split_test_by_target=False,
        target_upper_limit_train=3.0,
        target_bottom_limit_train=-11.0,
        default_base_name="p2-1_per_atom_rnd_rnd",
    ),
    "p2_grp_grp": SplitPreset(
        name="p2_grp_grp",
        use_group_for_train_test=True,
        use_group_for_train_valid=True,
        split_test_by_target=False,
        target_upper_limit_train=3.0,
        target_bottom_limit_train=-11.0,
        default_base_name="p2-2_per_atom_grp_grp",
    ),
    "p2_tar_rnd": SplitPreset(
        name="p2_tar_rnd",
        use_group_for_train_test=False,
        use_group_for_train_valid=False,
        split_test_by_target=True,
        target_upper_limit_train=0.1,
        target_bottom_limit_train=-1.7,
        default_base_name="p2-3_per_atom_tar_rnd",
    ),
    "p2_targrp_grp": SplitPreset(
        name="p2_targrp_grp",
        use_group_for_train_test=True,
        use_group_for_train_valid=True,
        split_test_by_target=True,
        target_upper_limit_train=0.4,
        target_bottom_limit_train=-2.6,
        default_base_name="p2-4_per_atom_targrp_grp",
    ),
}


def get_preset(name: str) -> SplitPreset:
    key = name.lower().strip()
    if key not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}. Choose from: {list(PRESETS)}")
    return PRESETS[key]
