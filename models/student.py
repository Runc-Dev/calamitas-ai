"""Lightweight Siamese student network trained via knowledge distillation.

The student is a Siamese SegFormer-B0 (~3.7M params) trained against the
~50M-param teacher using the 5-component KD loss in
:mod:`afetsonar.losses.distillation`. It is designed to be edge-deployable
(Jetson Nano class, ~36 ms/image at 768x768 on an RTX 3060).

The exact student architecture is the one that shipped in
``03_student_distillation.ipynb`` and is mirrored here so that the trained
checkpoint ``checkpoints/student/student_v1_best_ema.pth`` loads cleanly.

References
----------
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a
  Neural Network. *arXiv:1503.02531*.
- Xie et al. 2021 — SegFormer (NeurIPS).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel


# Default class counts (must match the teacher / training pipeline).
NUM_DAMAGE_CLASSES = 6       # bg, no, minor, major, destroyed, unclassified
NUM_CHANGE_CLASSES = 2       # no-change, change
NUM_DISASTER_CLASSES = 5     # wind, fire, flood, earthquake, volcano


class LightDecoder(nn.Module):
    """SegFormer-style MLP decoder, trimmed for the student model.

    The decoder projects each encoder stage to ``embed_dim`` channels with a
    1x1 conv, upsamples all of them to the largest feature-map resolution,
    concatenates, fuses, and classifies. A reference to the pre-classifier
    feature (``fused``) is returned for use with feature distillation.
    """

    def __init__(
        self,
        in_channels_list: Sequence[int],
        embed_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.linear_c = nn.ModuleList(
            [nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels_list]
        )
        self.linear_fuse = nn.Conv2d(
            embed_dim * len(in_channels_list), embed_dim, kernel_size=1
        )
        self.bn = nn.BatchNorm2d(embed_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode a list of hierarchical features to class logits.

        Parameters
        ----------
        features:
            Feature maps ``[B, C_i, H_i, W_i]`` listed smallest-to-largest
            (i.e. the standard SegFormer encoder output order).
        target_size:
            Output spatial resolution ``(H, W)``.
        """
        projected: List[torch.Tensor] = []
        for feat, proj in zip(features, self.linear_c):
            x = proj(feat)
            x = F.interpolate(
                x, size=features[0].shape[2:], mode="bilinear", align_corners=False
            )
            projected.append(x)
        fused = torch.cat(projected, dim=1)
        fused = self.linear_fuse(fused)
        fused = self.act(self.bn(fused))
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        logits = F.interpolate(
            logits, size=target_size, mode="bilinear", align_corners=False
        )
        return logits, fused


class StudentSiameseSegformer(nn.Module):
    """Small Siamese student built on a shared SegFormer-B0 backbone.

    Parameters
    ----------
    pretrained_name:
        HuggingFace id of the backbone (defaults to ``nvidia/mit-b0``).
    embed_dim:
        Channel width used by both light decoders (damage + change).
    num_damage:
        Damage class count (6).
    num_change:
        Change class count (2 — changed / unchanged).
    num_disaster:
        Disaster type class count (5).

    Output
    ------
    dict with keys:
        - ``damage_logits``    : ``[B, num_damage,   H, W]``
        - ``change_logits``    : ``[B, num_change,   H, W]``
        - ``disaster_logits``  : ``[B, num_disaster]``
        - ``feat_for_kd``      : ``[B, 2*hidden[-1], h, w]`` for KD loss
    """

    def __init__(
        self,
        pretrained_name: str = "nvidia/mit-b0",
        embed_dim: int = 128,
        num_damage: int = NUM_DAMAGE_CLASSES,
        num_change: int = NUM_CHANGE_CLASSES,
        num_disaster: int = NUM_DISASTER_CLASSES,
    ) -> None:
        super().__init__()
        self.backbone = SegformerModel.from_pretrained(pretrained_name)
        hidden_sizes: Sequence[int] = self.backbone.config.hidden_sizes  # B0: [32,64,160,256]
        self.hidden_sizes = list(hidden_sizes)

        # Damage decoder consumes concat(pre, post) per stage.
        damage_in = [c * 2 for c in hidden_sizes]
        self.damage_decoder = LightDecoder(damage_in, embed_dim, num_damage)

        # Change decoder consumes |pre - post| per stage.
        change_in = list(hidden_sizes)
        self.change_decoder = LightDecoder(change_in, embed_dim, num_change)

        self.disaster_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_sizes[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_disaster),
        )

    def _extract(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Return the 4 hierarchical SegFormer hidden states."""
        out = self.backbone(
            pixel_values=img, output_hidden_states=True, return_dict=True
        )
        return list(out.hidden_states)

    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run a Siamese forward pass.

        Parameters
        ----------
        pre:
            Pre-disaster image ``[B, 3, H, W]``.
        post:
            Post-disaster image ``[B, 3, H, W]``.
        """
        _, _, h, w = pre.shape
        pre_feats = self._extract(pre)
        post_feats = self._extract(post)

        damage_feats = [torch.cat([p, q], dim=1) for p, q in zip(pre_feats, post_feats)]
        damage_logits, _ = self.damage_decoder(damage_feats, target_size=(h, w))

        change_feats = [torch.abs(p - q) for p, q in zip(pre_feats, post_feats)]
        change_logits, _ = self.change_decoder(change_feats, target_size=(h, w))

        disaster_logits = self.disaster_head(post_feats[-1])

        feat_for_kd = damage_feats[-1]  # [B, 2*hidden[-1], h, w]

        return {
            "damage_logits": damage_logits,
            "change_logits": change_logits,
            "disaster_logits": disaster_logits,
            "feat_for_kd": feat_for_kd,
        }

    def num_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
