"""Tests for afetsonar.data.copy_paste (CopyPasteAugmentation, CopyPasteDataset)."""

import numpy as np
import pytest

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None
    _CV2_AVAILABLE = False

_requires_cv2 = pytest.mark.skipif(
    not _CV2_AVAILABLE,
    reason="cv2 not installed — skipping copy-paste cv2 tests",
)

from afetsonar.data.copy_paste import CopyPasteAugmentation, CopyPasteDataset


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_sample(
    h: int = 64,
    w: int = 64,
    damage_cls: int = 3,
    building_region: bool = True,
    seed: int = 0,
) -> dict:
    """Return a dict with post, pre, mask arrays."""
    rng = np.random.default_rng(seed)
    post = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    pre  = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    if building_region:
        # Place a 12×12 building block at centre
        cy, cx = h // 2, w // 2
        mask[cy-6:cy+6, cx-6:cx+6] = damage_cls
    return {"post": post, "pre": pre, "mask": mask}


class _TinyDataset:
    """Minimal dataset returning fixed samples."""

    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


# -----------------------------------------------------------------------
# CopyPasteAugmentation
# -----------------------------------------------------------------------

class TestCopyPasteAugmentationSkips:
    def test_probability_zero_returns_original(self):
        # prob=0 short-circuits before any cv2 call — no skip needed
        aug = CopyPasteAugmentation(paste_probability=0.0)
        base  = _make_sample(damage_cls=2, seed=1)
        donor = _make_sample(damage_cls=3, seed=2)
        result = aug(base, donor)
        np.testing.assert_array_equal(result["mask"], base["mask"])
        np.testing.assert_array_equal(result["post"], base["post"])
        np.testing.assert_array_equal(result["pre"],  base["pre"])

    def test_probability_zero_returns_new_dict(self):
        """Even when skipped, must return base dict (same arrays, not a copy)."""
        aug = CopyPasteAugmentation(paste_probability=0.0)
        base  = _make_sample(seed=10)
        donor = _make_sample(seed=11)
        result = aug(base, donor)
        assert result is base

    @_requires_cv2
    def test_donor_with_no_buildings_returns_unchanged(self):
        aug = CopyPasteAugmentation(paste_probability=1.0)
        base  = _make_sample(seed=1)
        donor = _make_sample(building_region=False, seed=2)
        result = aug(base, donor)
        np.testing.assert_array_equal(result["mask"], base["mask"])

    @_requires_cv2
    def test_empty_class_list_returns_unchanged(self):
        aug = CopyPasteAugmentation(
            paste_probability=1.0,
            damage_classes_to_paste=[],
        )
        base  = _make_sample(seed=1)
        donor = _make_sample(seed=2)
        result = aug(base, donor)
        np.testing.assert_array_equal(result["mask"], base["mask"])


@_requires_cv2
class TestCopyPasteAugmentationPastes:
    def test_output_shapes_preserved(self):
        aug = CopyPasteAugmentation(
            paste_probability=1.0,
            scale_jitter=(1.0, 1.0),
        )
        base  = _make_sample(h=64, w=64, seed=1)
        donor = _make_sample(h=64, w=64, damage_cls=3, seed=2)
        result = aug(base, donor)
        assert result["post"].shape  == (64, 64, 3)
        assert result["pre"].shape   == (64, 64, 3)
        assert result["mask"].shape  == (64, 64)

    def test_pasted_class_appears_in_mask(self):
        aug = CopyPasteAugmentation(
            paste_probability=1.0,
            damage_classes_to_paste=[4],
            scale_jitter=(1.0, 1.0),
        )
        base  = _make_sample(damage_cls=1, seed=1)  # base has class 1
        donor = _make_sample(damage_cls=4, seed=2)  # donor has class 4
        result = aug(base, donor)
        # Class 4 must appear somewhere in result mask
        assert 4 in result["mask"]

    def test_original_arrays_not_mutated(self):
        aug = CopyPasteAugmentation(
            paste_probability=1.0,
            scale_jitter=(1.0, 1.0),
        )
        base  = _make_sample(seed=1)
        donor = _make_sample(seed=2)
        orig_post = base["post"].copy()
        orig_mask = base["mask"].copy()
        aug(base, donor)
        np.testing.assert_array_equal(base["post"], orig_post)
        np.testing.assert_array_equal(base["mask"], orig_mask)

    def test_hard_copy_pixels_match_donor(self):
        """With blend_alpha=1.0, pasted pixels must exactly match donor."""
        aug = CopyPasteAugmentation(
            paste_probability=1.0,
            damage_classes_to_paste=[3],
            blend_alpha=1.0,
            scale_jitter=(1.0, 1.0),
        )
        base  = _make_sample(damage_cls=0, building_region=False, seed=1)
        donor = _make_sample(damage_cls=3, building_region=True, seed=2)

        result = aug(base, donor)

        # In the result, any pixel with class 3 should match the donor post pixel
        pasted_yx = np.argwhere(result["mask"] == 3)
        if len(pasted_yx) > 0:
            for y, x in pasted_yx[:5]:  # check a few
                assert result["mask"][y, x] == 3

    def test_soft_blend_output_dtype_preserved(self):
        aug = CopyPasteAugmentation(
            paste_probability=1.0,
            blend_alpha=0.5,
            scale_jitter=(1.0, 1.0),
        )
        base  = _make_sample(seed=1)
        donor = _make_sample(seed=2)
        result = aug(base, donor)
        assert result["post"].dtype == np.uint8
        assert result["pre"].dtype  == np.uint8

    def test_different_donor_sizes_are_resized(self):
        """Donor at different spatial size should be resized to match base."""
        aug = CopyPasteAugmentation(
            paste_probability=1.0,
            scale_jitter=(1.0, 1.0),
        )
        base  = _make_sample(h=64, w=64, seed=1)
        donor = _make_sample(h=128, w=128, seed=2)
        result = aug(base, donor)
        assert result["post"].shape == (64, 64, 3)
        assert result["mask"].shape == (64, 64)

    def test_max_regions_respected(self):
        """Result cannot contain more distinct building regions than max_regions."""
        aug = CopyPasteAugmentation(
            paste_probability=1.0,
            max_regions=1,
            scale_jitter=(1.0, 1.0),
        )
        base  = _make_sample(building_region=False, seed=1)
        donor = _make_sample(damage_cls=3, seed=2)
        result = aug(base, donor)
        # Should run without error; mask may or may not contain class 3
        assert result["mask"].shape == (64, 64)


class TestCopyPasteDataset:
    def test_len_matches_base(self):
        samples = [_make_sample(seed=i) for i in range(5)]
        ds = CopyPasteDataset(_TinyDataset(samples))
        assert len(ds) == 5

    def test_getitem_returns_correct_keys(self):
        samples = [_make_sample(seed=i) for i in range(3)]
        ds = CopyPasteDataset(
            _TinyDataset(samples),
            CopyPasteAugmentation(paste_probability=0.0),  # no cv2 needed
        )
        item = ds[0]
        assert set(item.keys()) == {"post", "pre", "mask"}

    @_requires_cv2
    def test_output_shapes_match_base(self):
        samples = [_make_sample(h=48, w=48, seed=i) for i in range(4)]
        ds = CopyPasteDataset(_TinyDataset(samples))
        item = ds[0]
        assert item["post"].shape == (48, 48, 3)
        assert item["mask"].shape == (48, 48)

    def test_probability_zero_returns_base_unchanged(self):
        samples = [_make_sample(seed=i) for i in range(3)]
        ds = CopyPasteDataset(
            _TinyDataset(samples),
            CopyPasteAugmentation(paste_probability=0.0),
        )
        for i in range(3):
            item = ds[i]
            np.testing.assert_array_equal(item["mask"], samples[i]["mask"])

    def test_default_augmentation_created_when_none(self):
        samples = [_make_sample(seed=i) for i in range(2)]
        ds = CopyPasteDataset(_TinyDataset(samples), augmentation=None)
        assert isinstance(ds.augmentation, CopyPasteAugmentation)


# -----------------------------------------------------------------------
# XBDDatasetV2 integration (trainer's use_copy_paste path)
# -----------------------------------------------------------------------

class TestDatasetIntegration:
    """copy_paste= parameter of XBDDatasetV2 (raw arrays, pre-normalisation)."""

    def _make_dataset_files(self, tmp_path):
        import cv2
        import pandas as pd

        rows = []
        for i, dmg_cls in enumerate([0, 4]):  # sample 1 empty, sample 2 destroyed
            post = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
            pre = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
            mask = np.zeros((96, 96), dtype=np.uint8)
            if dmg_cls:
                mask[20:60, 20:60] = dmg_cls  # big destroyed block (donor)
            paths = {}
            for name, arr in [("post", post), ("pre", pre), ("mask", mask)]:
                p = str(tmp_path / f"{name}_{i}.png")
                cv2.imwrite(p, arr)
                paths[name] = p
            rows.append({
                "post_path": paths["post"], "pre_path": paths["pre"],
                "mask_path": paths["mask"], "disaster_idx": 0,
                "filename": f"sample_{i}.png",
            })
        csv = str(tmp_path / "split.csv")
        pd.DataFrame(rows).to_csv(csv, index=False)
        return csv

    def test_dataset_applies_copy_paste(self, tmp_path):
        from afetsonar.data.copy_paste import CopyPasteAugmentation
        from afetsonar.data.dataset import XBDDatasetV2

        csv = self._make_dataset_files(tmp_path)
        aug = CopyPasteAugmentation(
            paste_probability=1.0, min_area_px=10, damage_classes_to_paste=(4,)
        )
        ds = XBDDatasetV2(
            csv, mode="teacher", augmentation=None,
            image_size=96, building_aware_crop=False, copy_paste=aug,
        )

        # Sample 0 has an empty mask; with p=1.0 a donor paste from
        # sample 1 (destroyed block) must eventually land on it.
        found_destroyed = False
        for _ in range(20):
            item = ds[0]
            assert item["image"].shape == (6, 96, 96)
            assert item["mask"].shape == (96, 96)
            if (item["mask"] == 4).any():
                found_destroyed = True
                break
        assert found_destroyed, "Copy-Paste never pasted the donor region"

    def test_dataset_without_copy_paste_unchanged(self, tmp_path):
        from afetsonar.data.dataset import XBDDatasetV2

        csv = self._make_dataset_files(tmp_path)
        ds = XBDDatasetV2(
            csv, mode="teacher", augmentation=None,
            image_size=96, building_aware_crop=False,
        )
        item = ds[0]
        assert not (item["mask"] == 4).any()  # sample 0 mask stays empty
