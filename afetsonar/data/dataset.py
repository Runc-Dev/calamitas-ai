"""xBD damage segmentation dataset (v2).

Supports 6 damage classes:

=====  ================
Index  Class
=====  ================
0      background
1      no_damage
2      minor_damage
3      major_damage
4      destroyed
5      unclassified
=====  ================

Key features:

- **Building-aware crop** — 80 % of training crops are centred on a region
  that contains at least one building pixel.  This dramatically reduces the
  fraction of background-only patches.
- **WeightedRandomSampler support** — ``get_sample_weights()`` returns a
  per-sample weight vector suitable for PyTorch's
  ``WeightedRandomSampler``.
- **Two modes** — ``"teacher"`` returns a 6-channel (pre+post) tensor;
  ``"student"`` returns a 3-channel (post only) tensor.

References
----------
- Gupta et al. 2019 — xBD: A Dataset for Assessing Building Damage from
  Satellite Imagery.  arXiv:1911.09296.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class XBDDatasetV2(Dataset):
    """xBD multi-class damage segmentation dataset.

    The CSV file must have columns:

    - ``post_path`` — path to post-disaster RGB image.
    - ``pre_path`` — path to pre-disaster RGB image (teacher mode only).
    - ``mask_path`` — path to 8-bit grayscale damage mask (values 0–5).
    - ``disaster_idx`` — integer disaster-type label (0–4).
    - ``filename`` — basename used for logging.
    - ``sample_weight`` (optional) — pre-computed sampling weight.

    Args:
        csv_path: Path to split CSV file.
        mode: ``"teacher"`` (6-ch input) or ``"student"`` (3-ch input).
        augmentation: An ``albumentations`` ``Compose`` transform applied
            to ``(image, mask)`` — or ``(image, pre, mask)`` for teacher
            mode (use ``additional_targets={"pre": "image"}``).
        image_size: Target spatial resolution after cropping.
        building_aware_crop: Enable priority crop towards building regions.
        building_crop_prob: Probability of centering crop on a building
            pixel (vs. random crop).

    Example:
        >>> from afetsonar.data.augmentations import get_train_augmentation_v2
        >>> ds = XBDDatasetV2("splits/train.csv", mode="teacher",
        ...                   augmentation=get_train_augmentation_v2())
        >>> sample = ds[0]
        >>> sample["image"].shape   # torch.Size([6, 768, 768])
    """

    def __init__(
        self,
        csv_path: str,
        mode: str = "teacher",
        augmentation: Optional[Any] = None,
        image_size: int = 768,
        building_aware_crop: bool = True,
        building_crop_prob: float = 0.8,
    ) -> None:
        assert mode in ("teacher", "student"), f"mode must be 'teacher' or 'student', got {mode}"
        self.df = pd.read_csv(csv_path)
        self.mode = mode
        self.augmentation = augmentation
        self.image_size = image_size
        self.building_aware_crop = building_aware_crop
        self.building_crop_prob = building_crop_prob

    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_mask(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        return mask

    # ------------------------------------------------------------------
    # Cropping
    # ------------------------------------------------------------------

    def _building_aware_crop(
        self,
        post: np.ndarray,
        mask: np.ndarray,
        pre: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Crop toward building-containing regions when possible."""
        h, w = mask.shape
        th = tw = self.image_size

        if h <= th and w <= tw:
            return post, mask, pre

        nonzero = np.where(mask > 0)
        if len(nonzero[0]) == 0 or random.random() > self.building_crop_prob:
            y0 = random.randint(0, max(0, h - th))
            x0 = random.randint(0, max(0, w - tw))
        else:
            idx = random.randint(0, len(nonzero[0]) - 1)
            cy, cx = nonzero[0][idx], nonzero[1][idx]
            jy, jx = th // 4, tw // 4
            y0 = max(0, min(h - th, cy - th // 2 + random.randint(-jy, jy)))
            x0 = max(0, min(w - tw, cx - tw // 2 + random.randint(-jx, jx)))

        post_c = post[y0:y0 + th, x0:x0 + tw]
        mask_c = mask[y0:y0 + th, x0:x0 + tw]
        pre_c = pre[y0:y0 + th, x0:x0 + tw] if pre is not None else None
        return post_c, mask_c, pre_c

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single sample dict.

        Returns:
            Dict with keys:

            - ``"image"`` — float tensor ``(C, H, W)`` in [0, 1] (ImageNet
              normalised if augmentation normalises).
            - ``"mask"`` — long tensor ``(H, W)`` with values 0–5.
            - ``"disaster_idx"`` — long scalar.
            - ``"filename"`` — str.
        """
        row = self.df.iloc[idx]

        post = self._load_image(row["post_path"])
        mask = self._load_mask(row["mask_path"])

        # Ensure consistent spatial size
        if post.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(
                mask, (post.shape[1], post.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        if self.mode == "teacher":
            pre = self._load_image(row["pre_path"])
            if pre.shape[:2] != post.shape[:2]:
                pre = cv2.resize(pre, (post.shape[1], post.shape[0]))

            if self.building_aware_crop and post.shape[0] > self.image_size:
                post, mask, pre = self._building_aware_crop(post, mask, pre)

            if self.augmentation is not None:
                aug = self.augmentation(image=post, pre=pre, mask=mask)
                post, pre, mask = aug["image"], aug["pre"], aug["mask"]

            if isinstance(post, torch.Tensor):
                image = torch.cat([pre, post], dim=0)
            else:
                image_np = np.concatenate([pre, post], axis=2)
                image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        else:  # student mode
            if self.building_aware_crop and post.shape[0] > self.image_size:
                post, mask, _ = self._building_aware_crop(post, mask, None)

            if self.augmentation is not None:
                aug = self.augmentation(image=post, mask=mask)
                post, mask = aug["image"], aug["mask"]

            if isinstance(post, torch.Tensor):
                image = post
            else:
                image = torch.from_numpy(post).permute(2, 0, 1).float() / 255.0

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask": mask.long(),
            "disaster_idx": torch.tensor(row["disaster_idx"], dtype=torch.long),
            "filename": row["filename"],
        }

    # ------------------------------------------------------------------
    # Sampler support
    # ------------------------------------------------------------------

    def get_sample_weights(self) -> np.ndarray:
        """Return per-sample weights for ``WeightedRandomSampler``.

        Uses ``sample_weight`` column if present, otherwise heuristically
        up-weights damaged images by a factor of 5.

        Returns:
            Float32 array of length ``len(self)``.
        """
        if "sample_weight" in self.df.columns:
            return self.df["sample_weight"].values.astype(np.float32)
        weights = np.ones(len(self.df), dtype=np.float32)
        for col in ("damage_present", "has_any_damage", "has_damage"):
            if col in self.df.columns:
                weights[self.df[col].values.astype(bool)] = 5.0
                break
        return weights
