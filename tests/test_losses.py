"""Unit tests for afetsonar.losses.*"""

from __future__ import annotations

import pytest
torch = pytest.importorskip("torch", reason="torch not installed — skipping loss tests")


class TestLovaszSoftmaxLoss:
    def test_forward_scalar(self, dummy_mask, dummy_siamese_batch):
        from afetsonar.losses import LovaszSoftmaxLoss
        B, _, H, W = dummy_siamese_batch.shape
        logits = torch.randn(B, 6, H, W)
        loss_fn = LovaszSoftmaxLoss()
        loss = loss_fn(logits, dummy_mask)
        assert loss.ndim == 0 and loss.item() >= 0, "Loss should be a non-negative scalar"

    def test_perfect_prediction_low_loss(self):
        from afetsonar.losses import LovaszSoftmaxLoss
        # All pixels are class 1 — perfect logits should give near-zero loss
        targets = torch.ones(2, 8, 8, dtype=torch.long)
        logits = torch.zeros(2, 6, 8, 8)
        logits[:, 1, :, :] = 10.0  # class 1 dominates
        loss_fn = LovaszSoftmaxLoss()
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.1, f"Expected low loss for perfect prediction, got {loss.item()}"


class TestComboDamageLossV3:
    def test_forward_dict_keys(self, dummy_mask, dummy_siamese_batch):
        from afetsonar.losses import ComboDamageLossV3
        B, _, H, W = dummy_siamese_batch.shape
        logits = torch.randn(B, 6, H, W)
        loss_fn = ComboDamageLossV3(num_classes=6)
        out = loss_fn(logits, dummy_mask)
        for key in ("total", "lovasz", "dice", "focal"):
            assert key in out, f"Missing key: {key}"

    def test_total_non_negative(self, dummy_mask, dummy_siamese_batch):
        from afetsonar.losses import ComboDamageLossV3
        B, _, H, W = dummy_siamese_batch.shape
        logits = torch.randn(B, 6, H, W)
        out = ComboDamageLossV3(num_classes=6)(logits, dummy_mask)
        assert out["total"].item() >= 0


class TestLocalizationLoss:
    def test_forward(self, dummy_binary_mask):
        from afetsonar.losses import LocalizationLoss
        B, H, W = dummy_binary_mask.shape
        logits = torch.randn(B, 2, H, W)
        out = LocalizationLoss()(logits, dummy_binary_mask)
        assert "total" in out and "ce" in out and "dice" in out
        assert out["total"].item() >= 0

    def test_derive_change_mask(self):
        from afetsonar.losses import derive_change_mask
        mask = torch.tensor([[[0, 1, 2, 3, 4, 5]]])
        change = derive_change_mask(mask)
        expected = torch.tensor([[[0, 0, 1, 1, 1, 0]]])
        assert torch.equal(change, expected)

    def test_derive_building_mask(self):
        from afetsonar.losses import derive_building_mask
        mask = torch.tensor([[[0, 1, 2, 3, 4, 5]]])
        building = derive_building_mask(mask)
        expected = torch.tensor([[[0, 1, 1, 1, 1, 1]]])
        assert torch.equal(building, expected)


class TestKnowledgeDistillationLoss:
    @pytest.fixture
    def models(self):
        from afetsonar.models import StudentSiameseSegformer, SiameseTeacherSegformerV3
        student = StudentSiameseSegformer(pretrained=False)
        teacher = SiameseTeacherSegformerV3(pretrained=False)
        return student, teacher

    def test_forward_keys(self, models, dummy_siamese_batch, dummy_mask):
        from afetsonar.losses import KnowledgeDistillationLoss
        student, teacher = models
        s_out = student(dummy_siamese_batch)
        with torch.no_grad():
            t_out = teacher(dummy_siamese_batch)

        kd = KnowledgeDistillationLoss(num_classes=6)
        result = kd(s_out, t_out, {"damage_mask": dummy_mask})
        for key in ("total", "kd", "ce", "feat", "att", "combo"):
            assert key in result, f"Missing key: {key}"

    def test_total_non_negative(self, models, dummy_siamese_batch, dummy_mask):
        from afetsonar.losses import KnowledgeDistillationLoss
        student, teacher = models
        s_out = student(dummy_siamese_batch)
        with torch.no_grad():
            t_out = teacher(dummy_siamese_batch)
        result = KnowledgeDistillationLoss(num_classes=6)(s_out, t_out, {"damage_mask": dummy_mask})
        assert result["total"].item() >= 0


def test_dice_class_weights_length_validated():
    """Review finding #7: wrong-length class_weights must fail at
    construction, not at the first training batch."""
    import pytest as _pytest
    from afetsonar.losses.combo import ComboDamageLossV3, DiceLoss

    with _pytest.raises(ValueError, match="class_weights length"):
        DiceLoss(num_classes=6, class_weights=[1.0, 2.0])

    with _pytest.raises(ValueError, match="class_weights length"):
        ComboDamageLossV3(num_classes=6, class_weights=[1.0, 2.0, 3.0])
