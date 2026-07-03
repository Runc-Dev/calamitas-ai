"""Training and validation augmentation pipelines (v2).

Design principles:

- **Damage-signal preservation** — aggressive colour jitter that shifts pixel
  intensities uniformly between pre and post images would destroy the
  change signal.  Therefore saturation/hue shifts are kept minimal and
  CoarseDropout/GaussNoise are excluded.
- **Geometric symmetry** — all geometric transforms are applied identically
  to the pre image and the post image (via ``additional_targets``).
- **ImageNet normalisation** — required because SegFormer uses ImageNet
  pretrained weights.

Augmentation strategy (v2 vs v1):
  - Removed: GaussNoise, CoarseDropout (destroy damage detection signal).
  - Kept: random rotate 90°, horizontal/vertical flip, small brightness/contrast.
  - Added: mild hue saturation jitter for inter-disaster variety.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_augmentation_v2(
    image_size: int = 512,
    mode: str = "teacher",
) -> A.Compose:
    """Build training augmentation pipeline.

    Args:
        image_size: Spatial resolution of the output crops.
        mode: ``"teacher"`` adds ``pre`` as an additional sync target;
            ``"student"`` augments the post image only.

    Returns:
        An ``albumentations.Compose`` instance.
    """
    transforms = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
        A.RandomCrop(height=image_size, width=image_size),
        # Geometry (class-preserving)
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        # Mild colour jitter (careful — damage signal is spectral)
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.3),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    if mode == "teacher":
        return A.Compose(transforms, additional_targets={"pre": "image"})
    return A.Compose(transforms)


def get_val_augmentation_v2(
    image_size: int = 512,
    mode: str = "teacher",
) -> A.Compose:
    """Build deterministic validation augmentation pipeline.

    No random operations — only resize, pad, and normalise.

    Args:
        image_size: Spatial resolution of the output images.
        mode: ``"teacher"`` or ``"student"``.

    Returns:
        An ``albumentations.Compose`` instance.
    """
    transforms = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    if mode == "teacher":
        return A.Compose(transforms, additional_targets={"pre": "image"})
    return A.Compose(transforms)
