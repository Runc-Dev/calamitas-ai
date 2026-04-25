"""Knowledge Distillation loss — 5-component objective (Phase 3).

The student is trained to simultaneously match the teacher across five
complementary signals:

1. **Soft-label KD** — KL-divergence between teacher and student softmax
   distributions (Hinton et al. 2015 temperature scaling).
2. **Hard-label CE** — Standard cross-entropy against ground-truth labels.
3. **Feature matching** — MSE between projected student and teacher last-stage
   feature maps (spatial knowledge transfer).
4. **Attention transfer** — L2 distance between normalised attention maps
   summed over channels (Zagoruyko & Komodakis 2017).
5. **Combo damage loss** — Lovász+Dice+Focal on student hard predictions
   (ensures the student optimises mIoU directly, not just matching the teacher).

References
----------
- Hinton et al. 2015 — Distilling the Knowledge in a Neural Network.
  arXiv:1503.02531.
- Zagoruyko & Komodakis 2017 — Paying More Attention to Attention (ICLR 2017).
- Furlanello et al. 2018 — Born Again Networks (ICML 2018).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from afetsonar.losses.combo import ComboDamageLossV3


class KnowledgeDistillationLoss(nn.Module):
    """5-component knowledge distillation loss for student training.

    Args:
        num_classes: Number of damage classes.
        temperature: Softening temperature for the KD term (default 4.0).
        alpha: Weight for the KD (soft-label) term.
        beta: Weight for the CE (hard-label) term.
        gamma: Weight for the feature-matching term.
        delta: Weight for the attention-transfer term.
        epsilon: Weight for the combo damage term.
        class_weights: Per-class weights passed to the combo loss.
        teacher_feat_channels: Channel count of the teacher's last-stage
            feature map.  Used to build a projection layer that adapts the
            student's feature dimension.
        student_feat_channels: Channel count of the student's last-stage
            feature map.
    """

    def __init__(
        self,
        num_classes: int = 6,
        temperature: float = 4.0,
        alpha: float = 0.30,   # KD soft labels
        beta: float = 0.25,    # CE hard labels
        gamma: float = 0.20,   # feature matching
        delta: float = 0.10,   # attention transfer
        epsilon: float = 0.15, # combo damage
        class_weights: Optional[List[float]] = None,
        teacher_feat_channels: int = 512,
        student_feat_channels: int = 256,
    ) -> None:
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

        self.combo = ComboDamageLossV3(
            num_classes=num_classes, class_weights=class_weights
        )

        # Linear projection to match teacher feature channels
        if teacher_feat_channels != student_feat_channels:
            self.feat_proj: Optional[nn.Module] = nn.Conv2d(
                student_feat_channels, teacher_feat_channels, kernel_size=1, bias=False
            )
        else:
            self.feat_proj = None

    # ------------------------------------------------------------------
    # Sub-loss helpers
    # ------------------------------------------------------------------

    def _kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL-divergence KD term with temperature scaling."""
        B, C, H, W = student_logits.shape
        s_log = F.log_softmax(student_logits / self.T, dim=1)
        t_soft = F.softmax(teacher_logits / self.T, dim=1)
        return F.kl_div(s_log, t_soft, reduction="batchmean") * (self.T ** 2)

    @staticmethod
    def _attention_map(feat: torch.Tensor) -> torch.Tensor:
        """Compute normalised attention map: sum(|F|^2, dim=C)."""
        att = feat.pow(2).mean(dim=1, keepdim=True)
        B, _, H, W = att.shape
        att = att.view(B, -1)
        att = F.normalize(att, p=2, dim=1)
        return att

    def _attention_loss(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> torch.Tensor:
        """L2 distance between normalised attention maps."""
        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            student_feat = F.interpolate(
                student_feat,
                size=teacher_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        s_att = self._attention_map(student_feat)
        t_att = self._attention_map(teacher_feat)
        return F.mse_loss(s_att, t_att)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        student_outputs: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        teacher_outputs: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute the 5-component KD loss.

        Args:
            student_outputs: Output dict from the student model.
            teacher_outputs: Output dict from the teacher model
                (gradient should be detached before calling this).
            targets: Ground-truth dict with key ``"damage_mask"``.

        Returns:
            Dict with keys ``total``, ``kd``, ``ce``, ``feat``, ``att``,
            ``combo``.
        """
        # Unwrap damage logits (teacher may return a list for deep supervision)
        s_logits: torch.Tensor = student_outputs["damage_logits"]
        t_logits_raw = teacher_outputs["damage_logits"]
        t_logits: torch.Tensor = (
            t_logits_raw[0] if isinstance(t_logits_raw, (list, tuple)) else t_logits_raw
        )

        # Align spatial sizes if teacher used a different resolution
        if s_logits.shape[-2:] != t_logits.shape[-2:]:
            t_logits = F.interpolate(
                t_logits, size=s_logits.shape[-2:], mode="bilinear", align_corners=False
            )

        damage_mask = targets["damage_mask"]

        # 1. KD soft-label loss
        l_kd = self._kd_loss(s_logits, t_logits)

        # 2. CE hard-label loss
        l_ce = F.cross_entropy(s_logits, damage_mask)

        # 3. Feature matching
        s_feat: torch.Tensor = student_outputs["feat_for_kd"]
        t_feat_raw = teacher_outputs.get("feat_for_kd")
        if t_feat_raw is None:
            l_feat = torch.tensor(0.0, device=s_logits.device)
        else:
            t_feat: torch.Tensor = t_feat_raw  # type: ignore[assignment]
            if self.feat_proj is not None:
                s_feat_proj = self.feat_proj(s_feat)
            else:
                s_feat_proj = s_feat
            if s_feat_proj.shape[-2:] != t_feat.shape[-2:]:
                s_feat_proj = F.interpolate(
                    s_feat_proj,
                    size=t_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            l_feat = F.mse_loss(s_feat_proj, t_feat.detach())

        # 4. Attention transfer
        if t_feat_raw is not None:
            l_att = self._attention_loss(s_feat, t_feat_raw.detach())  # type: ignore[arg-type]
        else:
            l_att = torch.tensor(0.0, device=s_logits.device)

        # 5. Combo damage loss (direct mIoU optimisation)
        combo_dict = self.combo(s_logits, damage_mask)
        l_combo = combo_dict["total"]

        total = (
            self.alpha * l_kd
            + self.beta * l_ce
            + self.gamma * l_feat
            + self.delta * l_att
            + self.epsilon * l_combo
        )

        return {
            "total": total,
            "kd": l_kd.detach(),
            "ce": l_ce.detach(),
            "feat": l_feat.detach(),
            "att": l_att.detach(),
            "combo": l_combo.detach(),
        }
