"""Tests for afetsonar.evaluation.tta (TTAWrapper)."""

import io
import sys

import numpy as np
import pytest

# -----------------------------------------------------------------------
# torch guard — same pattern as test_trainer.py
# -----------------------------------------------------------------------
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None
    _TORCH_AVAILABLE = False

_requires_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="torch not installed — skipping TTA torch tests",
)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

from afetsonar.evaluation.tta import _TTATransform, _TTA_TRANSFORMS, TTAWrapper


def _random_image(h=64, w=64) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _random_logits(c=6, h=64, w=64) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.standard_normal((c, h, w)).astype(np.float32)


# -----------------------------------------------------------------------
# _TTATransform: round-trip invariance
# -----------------------------------------------------------------------

class TestTTATransformRoundTrip:
    """apply_image followed by apply_logits (inverse) must be an identity."""

    @pytest.mark.parametrize("t", _TTA_TRANSFORMS)
    def test_image_is_invertible(self, t):
        img = _random_image(32, 32)
        transformed = t.apply_image(img)
        # Reconstruct original via inverse on logits representation
        logits = np.transpose(img, (2, 0, 1)).astype(np.float32)
        t_logits = np.transpose(transformed, (2, 0, 1)).astype(np.float32)
        recovered = t.apply_logits(t_logits)
        np.testing.assert_array_equal(
            logits, recovered,
            err_msg=f"Round-trip failed for transform {t}",
        )

    @pytest.mark.parametrize("t", _TTA_TRANSFORMS)
    def test_logits_shape_preserved(self, t):
        lgt = _random_logits(6, 32, 32)
        t_lgt = t.apply_logits(lgt)
        assert t_lgt.shape == lgt.shape

    @pytest.mark.parametrize("t", _TTA_TRANSFORMS)
    def test_image_shape_preserved_square(self, t):
        img = _random_image(32, 32)
        out = t.apply_image(img)
        assert out.shape == img.shape, f"Shape changed for {t}: {img.shape} → {out.shape}"

    def test_identity_is_no_op(self):
        t = _TTATransform()
        img = _random_image(16, 16)
        np.testing.assert_array_equal(img, t.apply_image(img))
        lgt = _random_logits(6, 16, 16)
        np.testing.assert_array_equal(lgt, t.apply_logits(lgt))

    def test_double_hflip_is_identity(self):
        t = _TTATransform(flip_h=True)
        img = _random_image(16, 16)
        np.testing.assert_array_equal(img, t.apply_image(t.apply_image(img)))

    def test_four_rot90_is_identity(self):
        img = _random_image(16, 16)
        out = img.copy()
        t = _TTATransform(rot90=1)
        for _ in range(4):
            out = t.apply_image(out)
        np.testing.assert_array_equal(img, out)


# -----------------------------------------------------------------------
# TTAWrapper construction
# -----------------------------------------------------------------------

class TestTTAWrapperConstruction:
    def test_valid_n_augmentations(self):
        wrapper = TTAWrapper.__new__(TTAWrapper)
        wrapper.pipeline = None
        wrapper.transforms = _TTA_TRANSFORMS[:4]
        wrapper.scales = (1.0,)
        assert len(wrapper.transforms) == 4

    def test_invalid_n_raises(self):
        class _FakePipeline:
            pass
        with pytest.raises(ValueError, match="n_augmentations"):
            TTAWrapper(_FakePipeline(), n_augmentations=0)
        with pytest.raises(ValueError, match="n_augmentations"):
            TTAWrapper(_FakePipeline(), n_augmentations=9)

    def test_n1_selects_only_identity(self):
        class _FakePipeline:
            pass
        tta = TTAWrapper(_FakePipeline(), n_augmentations=1)
        assert len(tta.transforms) == 1
        assert tta.transforms[0] == _TTATransform()  # identity

    def test_n8_uses_all_transforms(self):
        class _FakePipeline:
            pass
        tta = TTAWrapper(_FakePipeline(), n_augmentations=8)
        assert len(tta.transforms) == 8


# -----------------------------------------------------------------------
# TTAWrapper with mock pipeline — requires torch
# -----------------------------------------------------------------------

@_requires_torch
class TestTTAWrapperInference:
    """End-to-end TTA inference using a trivial mock pipeline."""

    def _make_pipeline(self, num_classes=6, image_size=32):
        """Minimal mock pipeline that mirrors AfetsonarPipeline's interface."""
        import torch
        import torch.nn as nn

        class _MockConfig:
            image_size = 32
            num_classes = 6

        class _MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(6, 6, 1)

            def forward(self, x):
                return {"damage_logits": self.conv(x)}

        class _MockPipeline:
            config = _MockConfig()
            device = _torch.device("cpu")
            model = _MockModel().eval()

            @staticmethod
            def _load_file(path):
                return np.zeros((32, 32, 3), dtype=np.uint8)

            @staticmethod
            def _resolve_pre(post_path, pre_path, lat, lon):
                return None

            @staticmethod
            def mask_to_buildings(mask, bbox_latlon=None, pixel_size_m=None):
                return []

        return _MockPipeline()

    def test_predict_from_arrays_returns_correct_shape(self):
        pipeline = self._make_pipeline()
        tta = TTAWrapper(pipeline, n_augmentations=4)
        post = _random_image(32, 32)
        mask = tta.predict_from_arrays(post)
        assert mask.shape == (32, 32)
        assert mask.dtype == np.uint8

    def test_predict_from_arrays_all_valid_classes(self):
        pipeline = self._make_pipeline()
        tta = TTAWrapper(pipeline, n_augmentations=8)
        post = _random_image(32, 32)
        mask = tta.predict_from_arrays(post)
        assert mask.min() >= 0
        assert mask.max() < 6

    def test_n1_matches_plain_inference(self):
        """With n=1 (identity only), TTA output must match direct pipeline output."""
        import torch
        pipeline = self._make_pipeline()
        tta_single = TTAWrapper(pipeline, n_augmentations=1)

        post = _random_image(32, 32)

        # TTA (identity only)
        tta_mask = tta_single.predict_from_arrays(post)

        # Direct pipeline output
        tensor = tta_single._preprocess_at_size(post, None, 32)
        with torch.no_grad():
            outputs = pipeline.model(tensor)
            logits = outputs["damage_logits"]
        direct_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        np.testing.assert_array_equal(tta_mask, direct_mask)

    def test_multiscale_runs_without_error(self):
        pipeline = self._make_pipeline()
        tta = TTAWrapper(pipeline, n_augmentations=2, scales=(0.75, 1.0, 1.25))
        post = _random_image(32, 32)
        mask = tta.predict_from_arrays(post)
        assert mask.shape == (32, 32)

    def test_pre_none_uses_post_as_fallback(self):
        """pre=None should not crash (uses post as pre inside _preprocess_at_size)."""
        pipeline = self._make_pipeline()
        tta = TTAWrapper(pipeline, n_augmentations=2)
        post = _random_image(32, 32)
        mask = tta.predict_from_arrays(post, pre=None)
        assert mask.shape == (32, 32)

    def test_analyze_returns_mask_and_buildings(self):
        from unittest.mock import patch
        pipeline = self._make_pipeline()
        tta = TTAWrapper(pipeline, n_augmentations=2)
        post = _random_image(32, 32)

        with (
            patch.object(pipeline, "_load_file", return_value=post),
            patch.object(pipeline, "_resolve_pre", return_value=None),
        ):
            result = tta.analyze("dummy_post.png")

        assert "mask" in result
        assert "buildings" in result
        assert result["mask"].shape == (32, 32)


# -----------------------------------------------------------------------
# tta_forward: batched tensor-level TTA
# -----------------------------------------------------------------------

@_requires_torch
class TestTTAForward:
    """Tensor-level TTA used by scripts/evaluate.py --tta."""

    def _make_model(self):
        class _DummyModel(_torch.nn.Module):
            def __init__(self):
                super().__init__()
                _torch.manual_seed(0)
                self.conv = _torch.nn.Conv2d(6, 6, 3, padding=1)

            def forward(self, x):
                return {"damage_logits": self.conv(x)}

        return _DummyModel().eval()

    @pytest.mark.parametrize("t", _TTA_TRANSFORMS)
    def test_tensor_round_trip_is_identity(self, t):
        x = _torch.randn(2, 6, 32, 32)
        recovered = t.invert_tensor(t.apply_tensor(x))
        assert _torch.equal(x, recovered), f"Round-trip failed for {t}"

    def test_output_shape_and_probabilities(self):
        from afetsonar.evaluation.tta import tta_forward

        model = self._make_model()
        x = _torch.randn(2, 6, 32, 32)
        probs = tta_forward(model, x, n_augmentations=8)
        assert probs.shape == (2, 6, 32, 32)
        sums = probs.sum(dim=1)
        assert _torch.allclose(sums, _torch.ones_like(sums), atol=1e-5)

    def test_single_augmentation_equals_plain_softmax(self):
        from afetsonar.evaluation.tta import tta_forward

        model = self._make_model()
        x = _torch.randn(1, 6, 32, 32)
        probs = tta_forward(model, x, n_augmentations=1)
        with _torch.no_grad():
            expected = _torch.softmax(model(x)["damage_logits"], dim=1)
        assert _torch.allclose(probs, expected, atol=1e-6)

    def test_multi_scale_preserves_shape(self):
        from afetsonar.evaluation.tta import tta_forward

        model = self._make_model()
        x = _torch.randn(1, 6, 64, 64)
        probs = tta_forward(model, x, n_augmentations=4, scales=(0.5, 1.0, 1.5))
        assert probs.shape == (1, 6, 64, 64)

    def test_invalid_n_augmentations_raises(self):
        from afetsonar.evaluation.tta import tta_forward

        model = self._make_model()
        x = _torch.randn(1, 6, 32, 32)
        with pytest.raises(ValueError):
            tta_forward(model, x, n_augmentations=9)
