"""5-component knowledge distillation loss for the AFETSONAR student.

The loss combines five terms:

1. ``L_hard``     — student damage vs ground truth (ComboDamageLossV3).
2. ``L_soft``     — KL divergence of softened damage logits, student vs teacher.
3. ``L_feat``     — MSE between the projected student feature and the teacher
                    feature at the deepest encoder stage.
4. ``L_change``   — KL on softened change logits.
5. ``L_disaster`` — KL on softened disaster-type logits.

All KD terms share a single temperature ``T`` and the losses are combined as a
convex combination whose weights must sum to ~1.0.

References
----------
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a
  Neural Network. *arXiv:1503.02531*.
- Romero, A. et al. (2015). FitNets: Hints for Thin Deep Nets —
  the feature-hint MSE term.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from afetsonar.losses.combo import ComboDamageLossV3


class KnowledgeDistillationLoss(nn.Module):
    """5-component knowledge distillation loss.

    Parameters
    ----------
    student_feat_channels:
        Channel count of the student's ``feat_for_kd`` tensor.
    teacher_feat_channels:
        Channel count of the teacher feature used as the MSE target.
    temperature:
        Softmax temperature ``T`` for the KD soft-label terms. ``T`` between
        2 and 6 works well; 4.0 is the classic Hinton default.
    weights:
        Dict with keys ``w_hard, w_soft, w_feat, w_change, w_disaster``. They
        must sum to ~1.0.
    ignore_index:
        Label value to exclude from ``L_hard`` and ``L_change`` computations.
    """

    def __init__(
        self,
        student_feat_channels: int,
        teacher_feat_channels: int,
        temperature: float = 4.0,
        weights: Optional[Dict[str, float]] = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.T = temperature
        self.ignore_index = ignore_index
        self.weights = weights or {
            "w_hard": 0.30,
            "w_soft": 0.40,
            "w_feat": 0.15,
            "w_change": 0.10,
            "w_disaster": 0.05,
        }
        total_weight = sum(self.weights.values())
        assert abs(total_weight - 1.0) < 1e-3, (
            f"KD weights must sum to 1.0, got {total_weight:.4f}"
        )

        self.hard_loss = ComboDamageLossV3(num_classes=6, ignore_index=ignore_index)
        self.feat_proj = nn.Conv2d(
            student_feat_channels, teacher_feat_channels, kernel_size=1, bias=False
        )
        self.change_ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    @staticmethod
    def _kd_kl(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        T: float,
    ) -> torch.Tensor:
        """Temperature-scaled KL divergence between softmax distributions.

        Applied pixel-wise for segmentation heads and batch-wise for the
        classification head.
        """
        s_log = F.log_softmax(student_logits / T, dim=1)
        t_prob = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(s_log, t_prob, reduction="batchmean") * (T * T)

    def forward(
        self,
        student_out: Dict[str, torch.Tensor],
        teacher_out: Dict[str, torch.Tensor],
        gt_damage: torch.Tensor,
        gt_change: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the total KD loss and its components.

        Parameters
        ----------
        student_out, teacher_out:
            Dicts with keys ``damage_logits``, ``change_logits``,
            ``disaster_logits`` and ``feat_for_kd``.
        gt_damage:
            Long tensor ``[B, H, W]`` of damage labels.
        gt_change:
            Currently unused — hard change labels are learned purely through
            the teacher via KD. Parameter kept for API compatibility.
        """
        del gt_change  # reserved for future hard-change supervision.

        losses: Dict[str, Any] = {}

        raw_hard = self.hard_loss(student_out["damage_logits"], gt_damage)
        l_hard = raw_hard["total"] if isinstance(raw_hard, dict) else raw_hard
        losses["L_hard"] = l_hard

        losses["L_soft"] = self._kd_kl(
            student_out["damage_logits"], teacher_out["damage_logits"], self.T
        )

        s_feat = student_out["feat_for_kd"]
        t_feat = teacher_out["feat_for_kd"]
        if s_feat.shape[2:] != t_feat.shape[2:]:
            s_feat = F.interpolate(
                s_feat, size=t_feat.shape[2:], mode="bilinear", align_corners=False
            )
        s_feat_proj = self.feat_proj(s_feat)
        losses["L_feat"] = F.mse_loss(s_feat_proj, t_feat)

        losses["L_change"] = self._kd_kl(
            student_out["change_logits"], teacher_out["change_logits"], self.T
        )
        losses["L_disaster"] = self._kd_kl(
            student_out["disaster_logits"], teacher_out["disaster_logits"], self.T
        )

        w = self.weights
        total = (
            w["w_hard"] * losses["L_hard"]
            + w["w_soft"] * losses["L_soft"]
            + w["w_feat"] * losses["L_feat"]
            + w["w_change"] * losses["L_change"]
            + w["w_disaster"] * losses["L_disaster"]
        )
        losses["L_total"] = total
        return total, losses
