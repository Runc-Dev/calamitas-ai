"""Copy-Paste augmentation for xBD damage segmentation training.

Copies contiguous building regions from a donor (pre, post, mask) onto a base
sample, creating synthetic training examples that increase the frequency of
rare damage classes:
    minor_damage  : 0.53% → effectively boosted by pasting
    major_damage  : 0.92% → effectively boosted by pasting
    destroyed     : 1.10% → effectively boosted by pasting

Reference: Ghiasi et al. (2021) "Simple Copy-Paste is a Strong Data
Augmentation Method for Instance Segmentation" — adapted for semantic
segmentation on Siamese (pre+post) pairs.

Usage inside a dataset's __getitem__::

    aug = CopyPasteAugmentation(paste_probability=0.5)
    base  = {"post": post_img, "pre": pre_img, "mask": mask}
    donor = dataset[random.randint(0, len(dataset) - 1)]
    result = aug(base, donor)

Or use the drop-in dataset wrapper::

    from afetsonar.data.copy_paste import CopyPasteDataset, CopyPasteAugmentation

    aug     = CopyPasteAugmentation(paste_probability=0.5)
    dataset = CopyPasteDataset(train_dataset, aug)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class CopyPasteAugmentation:
    """Copy building regions from a donor sample onto a base sample.

    Extracts connected components belonging to ``damage_classes_to_paste``
    from the donor mask and blends them onto the base images, updating the
    base mask accordingly.  Both the pre- and post-disaster channels are
    updated simultaneously to keep the Siamese pair consistent.

    Args:
        paste_probability: Probability of applying the augmentation per call.
        damage_classes_to_paste: Class IDs to copy.  Defaults to the three
            rare/high-value classes (minor 2, major 3, destroyed 4).
        min_area_px: Minimum contour area in pixels to be eligible.
        max_regions: Maximum number of regions pasted per sample.
        blend_alpha: Blend weight at paste boundaries.  ``1.0`` = hard copy
            (default).  Values < 1 give a soft linear blend.
        scale_jitter: ``(min, max)`` range for random region scaling before
            paste.  ``(1.0, 1.0)`` disables jitter.
    """

    def __init__(
        self,
        paste_probability: float = 0.5,
        damage_classes_to_paste: Sequence[int] = (2, 3, 4),
        min_area_px: int = 100,
        max_regions: int = 4,
        blend_alpha: float = 1.0,
        scale_jitter: Tuple[float, float] = (0.75, 1.25),
    ) -> None:
        self.paste_probability = paste_probability
        self.damage_classes_to_paste = list(damage_classes_to_paste)
        self.min_area_px = min_area_px
        self.max_regions = max_regions
        self.blend_alpha = blend_alpha
        self.scale_jitter = scale_jitter

    # ------------------------------------------------------------------

    def __call__(
        self,
        base: Dict[str, np.ndarray],
        donor: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Apply Copy-Paste augmentation.

        Args:
            base: Dict with ``"post"`` (H,W,3) uint8, ``"pre"`` (H,W,3)
                uint8, ``"mask"`` (H,W) uint8.
            donor: Same structure as ``base``.

        Returns:
            New dict with same keys.  Input dicts are not modified.
        """
        if random.random() > self.paste_probability:
            return base

        base_post = base["post"].copy()
        base_pre  = base["pre"].copy()
        base_mask = base["mask"].copy()
        H, W = base_mask.shape

        regions = self._extract_regions(
            donor["post"], donor["pre"], donor["mask"], H, W
        )
        if not regions:
            return base

        random.shuffle(regions)
        for region in regions[: self.max_regions]:
            base_post, base_pre, base_mask = self._paste_region(
                base_post, base_pre, base_mask, region
            )

        return {"post": base_post, "pre": base_pre, "mask": base_mask}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_regions(
        self,
        post: np.ndarray,
        pre: np.ndarray,
        mask: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> List[Dict]:
        """Return list of region dicts extracted from the donor arrays."""
        import cv2

        dH, dW = mask.shape
        if dH != target_h or dW != target_w:
            post = cv2.resize(post, (target_w, target_h))
            pre  = cv2.resize(pre,  (target_w, target_h))
            mask = cv2.resize(
                mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )

        regions: List[Dict] = []
        for cls in self.damage_classes_to_paste:
            binary = (mask == cls).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                if cv2.contourArea(cnt) < self.min_area_px:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                seg_mask = np.zeros((target_h, target_w), dtype=np.uint8)
                cv2.drawContours(seg_mask, [cnt], -1, 1, cv2.FILLED)

                regions.append({
                    "post":     post[y:y+h, x:x+w].copy(),
                    "pre":      pre[y:y+h, x:x+w].copy(),
                    "seg_mask": seg_mask[y:y+h, x:x+w],   # binary (h, w)
                    "cls_id":   cls,
                    "h":        h,
                    "w":        w,
                })

        return regions

    def _paste_region(
        self,
        base_post: np.ndarray,
        base_pre: np.ndarray,
        base_mask: np.ndarray,
        region: Dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Paste one region at a random location within the base sample."""
        import cv2

        H, W = base_mask.shape
        rh, rw = region["h"], region["w"]

        scale = random.uniform(*self.scale_jitter)
        if scale != 1.0:
            new_h = max(4, round(rh * scale))
            new_w = max(4, round(rw * scale))
            region = {
                "post":     cv2.resize(region["post"],     (new_w, new_h)),
                "pre":      cv2.resize(region["pre"],      (new_w, new_h)),
                "seg_mask": cv2.resize(
                    region["seg_mask"], (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST,
                ),
                "cls_id":   region["cls_id"],
                "h":        new_h,
                "w":        new_w,
            }
            rh, rw = new_h, new_w

        if rh > H or rw > W:
            return base_post, base_pre, base_mask

        ty = random.randint(0, H - rh)
        tx = random.randint(0, W - rw)

        seg = region["seg_mask"].astype(bool)

        if self.blend_alpha >= 1.0:
            for ch in range(3):
                base_post[ty:ty+rh, tx:tx+rw, ch] = np.where(
                    seg,
                    region["post"][:, :, ch],
                    base_post[ty:ty+rh, tx:tx+rw, ch],
                )
                base_pre[ty:ty+rh, tx:tx+rw, ch] = np.where(
                    seg,
                    region["pre"][:, :, ch],
                    base_pre[ty:ty+rh, tx:tx+rw, ch],
                )
        else:
            a = self.blend_alpha
            for ch in range(3):
                bp = base_post[ty:ty+rh, tx:tx+rw, ch].astype(np.float32)
                bq = base_pre[ty:ty+rh, tx:tx+rw, ch].astype(np.float32)
                np_ = region["post"][:, :, ch].astype(np.float32)
                nq  = region["pre"][:, :, ch].astype(np.float32)
                base_post[ty:ty+rh, tx:tx+rw, ch] = np.clip(
                    np.where(seg, a * np_ + (1 - a) * bp, bp), 0, 255
                ).astype(np.uint8)
                base_pre[ty:ty+rh, tx:tx+rw, ch] = np.clip(
                    np.where(seg, a * nq + (1 - a) * bq, bq), 0, 255
                ).astype(np.uint8)

        base_mask[ty:ty+rh, tx:tx+rw] = np.where(
            seg,
            np.full((rh, rw), region["cls_id"], dtype=np.uint8),
            base_mask[ty:ty+rh, tx:tx+rw],
        )

        return base_post, base_pre, base_mask


class CopyPasteDataset:
    """Dataset wrapper that applies :class:`CopyPasteAugmentation` lazily.

    Wraps any dataset whose ``__getitem__`` returns dicts with ``"post"``,
    ``"pre"``, and ``"mask"`` keys.  On each access it draws a random donor
    index and applies the augmentation.

    Args:
        dataset: The base dataset to wrap.
        augmentation: A :class:`CopyPasteAugmentation` instance.
            Defaults to ``CopyPasteAugmentation()`` with default settings.

    Example::

        from afetsonar.data.copy_paste import CopyPasteDataset, CopyPasteAugmentation

        aug     = CopyPasteAugmentation(paste_probability=0.5)
        dataset = CopyPasteDataset(train_dataset, aug)
        loader  = DataLoader(dataset, batch_size=4, shuffle=True)
    """

    def __init__(
        self,
        dataset: Any,
        augmentation: Optional[CopyPasteAugmentation] = None,
    ) -> None:
        self.dataset = dataset
        self.augmentation = augmentation or CopyPasteAugmentation()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        base = self.dataset[idx]
        donor_idx = random.randint(0, len(self.dataset) - 1)
        donor = self.dataset[donor_idx]
        return self.augmentation(base, donor)
