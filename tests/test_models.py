"""Unit tests for afetsonar.models.*"""

from __future__ import annotations

import pytest
import torch


# ============================================================
# LocalizerSegformer
# ============================================================

class TestLocalizerSegformer:
    @pytest.fixture
    def model(self):
        from afetsonar.models import LocalizerSegformer
        return LocalizerSegformer(pretrained=False)

    def test_forward_shape(self, model, dummy_rgb_batch):
        """Output logits must match input spatial resolution."""
        B, C, H, W = dummy_rgb_batch.shape
        logits = model(dummy_rgb_batch)
        assert logits.shape == (B, 2, H, W), f"Expected (B,2,H,W) got {logits.shape}"

    def test_num_parameters(self, model):
        params = model.num_parameters()
        assert params > 0

    def test_encoder_state_dict(self, model):
        sd = model.get_encoder_state_dict()
        assert isinstance(sd, dict) and len(sd) > 0


# ============================================================
# SiameseTeacherSegformerV3
# ============================================================

class TestTeacher:
    @pytest.fixture
    def model(self):
        from afetsonar.models import SiameseTeacherSegformerV3
        return SiameseTeacherSegformerV3(pretrained=False, use_deep_supervision=True)

    def test_forward_keys(self, model, dummy_siamese_batch):
        outputs = model(dummy_siamese_batch)
        assert "damage_logits" in outputs
        assert "change_logits" in outputs
        assert "disaster_logits" in outputs

    def test_damage_logits_shape(self, model, dummy_siamese_batch):
        B, C, H, W = dummy_siamese_batch.shape
        out = model(dummy_siamese_batch)
        # Deep supervision: list of logits
        dmg = out["damage_logits"]
        main = dmg[0] if isinstance(dmg, list) else dmg
        assert main.shape[0] == B and main.shape[-2:] == (H, W)

    def test_wrong_channels_raises(self, model):
        x = torch.randn(1, 3, 64, 64)
        with pytest.raises(AssertionError):
            model(x)

    def test_num_parameters_50m(self, model):
        # B3 Siamese teacher should have ~50M params
        params = model.num_parameters()
        assert params > 40_000_000, f"Expected ~50M, got {params/1e6:.1f}M"


# ============================================================
# StudentSiameseSegformer
# ============================================================

class TestStudent:
    @pytest.fixture
    def model(self):
        from afetsonar.models import StudentSiameseSegformer
        return StudentSiameseSegformer(pretrained=False)

    def test_forward_keys(self, model, dummy_siamese_batch):
        outputs = model(dummy_siamese_batch)
        for key in ("damage_logits", "change_logits", "disaster_logits", "feat_for_kd"):
            assert key in outputs, f"Missing key: {key}"

    def test_damage_logits_shape(self, model, dummy_siamese_batch):
        B, C, H, W = dummy_siamese_batch.shape
        out = model(dummy_siamese_batch)
        logits = out["damage_logits"]
        assert logits.shape == (B, 6, H, W)

    def test_num_parameters_smaller_than_teacher(self, model):
        from afetsonar.models import SiameseTeacherSegformerV3
        teacher = SiameseTeacherSegformerV3(pretrained=False)
        assert model.num_parameters() < teacher.num_parameters()


# ============================================================
# ModelEMA
# ============================================================

class TestModelEMA:
    def test_shadow_updates(self, dummy_siamese_batch):
        from afetsonar.models import ModelEMA, StudentSiameseSegformer
        model = StudentSiameseSegformer(pretrained=False)
        ema = ModelEMA(model, decay=0.9)

        # Mutate model weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1)

        # EMA shadow should change after update
        before = {k: v.clone() for k, v in ema.shadow.items()}
        ema.update(model)
        changed = sum(
            1 for k in before if not torch.equal(before[k], ema.shadow[k])
        )
        assert changed > 0, "EMA shadow did not update"

    def test_apply_and_restore(self):
        from afetsonar.models import ModelEMA, StudentSiameseSegformer
        model = StudentSiameseSegformer(pretrained=False)
        ema = ModelEMA(model, decay=0.9)

        original = {n: p.data.clone() for n, p in model.named_parameters()}
        backup = ema.apply_to(model)
        ema.restore(model, backup)

        for name, param in model.named_parameters():
            assert torch.allclose(param.data, original[name]), f"Parameter {name} not restored"
