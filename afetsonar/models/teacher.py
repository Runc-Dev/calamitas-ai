"""Siamese teacher network for Phase 2 damage classification.

The teacher uses a single shared SegFormer-B3 encoder that sees ``pre`` and
``post`` disaster images independently.  The resulting feature pyramids are
fused per-stage via ``[pre | post | |pre - post|]`` concatenation and passed
through the MiT decode head for 6-class damage segmentation.  Two auxiliary
heads (change + disaster type) anchor multi-task learning, and optional
auxiliary damage heads at intermediate encoder stages enable deep supervision.

References
----------
- Xie et al. 2021 — SegFormer (NeurIPS 2021).  arXiv:2105.15203.
- Zhao et al. 2017 — PSPNet auxiliary loss / deep supervision (CVPR 2017).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SiameseTeacherSegformerV3(nn.Module):
    """Siamese SegFormer-B3 teacher for building damage assessment.

    Args:
        backbone_name: HuggingFace model id or local checkpoint path.
        num_damage_classes: Number of damage severity classes (default 6).
        num_disaster_classes: Number of disaster event types (default 5).
        pretrained: Load pretrained backbone weights if ``True``.
        use_deep_supervision: Add auxiliary damage heads at intermediate
            encoder stages when ``True``.

    Input format:
        6-channel tensor ``(B, 6, H, W)`` where the first three channels are
        the pre-disaster RGB image and the last three are the post-disaster
        RGB image.

    Returns:
        A dict with keys:

        - ``"damage_logits"`` — list of logit tensors when deep supervision
          is enabled, otherwise a single tensor ``(B, C_dmg, H, W)``.
        - ``"change_logits"`` — binary change map ``(B, 2, H, W)``.
        - ``"disaster_logits"`` — image-level disaster type ``(B, C_dis)``.
    """

    def __init__(
        self,
        backbone_name: str = "nvidia/segformer-b3-finetuned-ade-512-512",
        num_damage_classes: int = 6,
        num_disaster_classes: int = 5,
        pretrained: bool = True,
        use_deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.num_damage_classes = num_damage_classes
        self.num_disaster_classes = num_disaster_classes
        self.use_deep_supervision = use_deep_supervision

        # ---- Backbone ----
        if pretrained:
            base = SegformerForSemanticSegmentation.from_pretrained(
                backbone_name,
                num_labels=num_damage_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            cfg = SegformerConfig.from_pretrained(backbone_name)
            cfg.num_labels = num_damage_classes
            base = SegformerForSemanticSegmentation(cfg)

        self.encoder = base.segformer.encoder
        self.decode_head = base.decode_head
        enc_channels: List[int] = list(base.config.hidden_sizes)  # e.g. [64,128,320,512]
        self.enc_channels = enc_channels

        # ---- Stage fusion convolutions  [pre | post | diff] → stage_ch ----
        self.fusion_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch * 3, ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                )
                for ch in enc_channels
            ]
        )

        # ---- Deep supervision heads (intermediate stages, excluding last) ----
        if use_deep_supervision:
            self.aux_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1),
                        nn.BatchNorm2d(ch // 2),
                        nn.ReLU(inplace=True),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(ch // 2, num_damage_classes, kernel_size=1),
                    )
                    for ch in enc_channels[:-1]
                ]
            )
        else:
            self.aux_heads = None

        last_ch = enc_channels[-1]

        # ---- Binary change detection head ----
        self.change_head = nn.Sequential(
            nn.Conv2d(last_ch, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 2, kernel_size=1),
        )

        # ---- Image-level disaster classification head ----
        self.disaster_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_disaster_classes),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run the shared encoder and return all intermediate feature maps."""
        out = self.encoder(x, output_hidden_states=True, return_dict=True)
        return list(out.hidden_states)

    def _fuse(
        self,
        feats_pre: List[torch.Tensor],
        feats_post: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Fuse pre and post feature pyramids stage by stage."""
        fused = []
        for i, (pre, post) in enumerate(zip(feats_pre, feats_post)):
            diff = torch.abs(post - pre)
            combined = torch.cat([pre, post, diff], dim=1)
            fused.append(self.fusion_convs[i](combined))
        return fused

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass.

        Args:
            x: Tensor of shape ``(B, 6, H, W)`` — [pre_rgb | post_rgb].

        Returns:
            Dict with ``damage_logits``, ``change_logits``,
            ``disaster_logits``.
        """
        B, C, H, W = x.shape
        assert C == 6, f"Expected 6 input channels, got {C}"

        pre, post = x[:, :3], x[:, 3:]
        feats_pre = self._encode(pre)
        feats_post = self._encode(post)
        fused = self._fuse(feats_pre, feats_post)

        # Main decoder
        damage_main = F.interpolate(
            self.decode_head(fused),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        # Deep supervision
        if self.use_deep_supervision and self.aux_heads is not None:
            damage_logits: Union[torch.Tensor, List[torch.Tensor]] = [damage_main]
            for i, aux_head in enumerate(self.aux_heads):
                aux = F.interpolate(
                    aux_head(fused[i]),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                damage_logits.append(aux)  # type: ignore[union-attr]
        else:
            damage_logits = damage_main

        # Auxiliary heads
        last_fused = fused[-1]
        change_logits = F.interpolate(
            self.change_head(last_fused),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        disaster_logits = self.disaster_head(last_fused)

        return {
            "damage_logits": damage_logits,
            "change_logits": change_logits,
            "disaster_logits": disaster_logits,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def load_encoder_from_localizer(self, localizer_state_dict: dict) -> Tuple[int, List[str]]:
        """Transfer encoder weights from a trained LocalizerSegformer.

        Args:
            localizer_state_dict: State dict returned by
                ``LocalizerSegformer.get_encoder_state_dict()``.

        Returns:
            ``(loaded_count, skipped_keys)`` — number of parameters loaded
            and list of keys that could not be transferred.
        """
        own_state = self.encoder.state_dict()
        loaded, skipped = 0, []
        for name, param in localizer_state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            else:
                skipped.append(name)
        return loaded, skipped

    def enable_gradient_checkpointing(self) -> bool:
        """Enable gradient checkpointing on the shared encoder."""
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
            return True
        return False

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
