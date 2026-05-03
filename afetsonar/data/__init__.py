"""AFETSONAR data utilities.

Public API::

    from afetsonar.data import XBDDatasetV2
    from afetsonar.data import get_train_augmentation_v2, get_val_augmentation_v2
    from afetsonar.data import mask_from_json, compute_sample_weights, build_split_csv
    from afetsonar.data import CopyPasteAugmentation, CopyPasteDataset
"""

try:
    from afetsonar.data.augmentations import get_train_augmentation_v2, get_val_augmentation_v2
    from afetsonar.data.dataset import XBDDatasetV2
    from afetsonar.data.preprocessing import (
        XBD_DAMAGE_CLASSES,
        build_split_csv,
        compute_sample_weights,
        mask_from_json,
    )
except ModuleNotFoundError:
    # albumentations / torch absent (local dev without training deps)
    pass

from afetsonar.data.copy_paste import CopyPasteAugmentation, CopyPasteDataset

__all__ = [
    "XBDDatasetV2",
    "get_train_augmentation_v2",
    "get_val_augmentation_v2",
    "XBD_DAMAGE_CLASSES",
    "mask_from_json",
    "compute_sample_weights",
    "build_split_csv",
    "CopyPasteAugmentation",
    "CopyPasteDataset",
]
