"""Siamese teacher network for Phase 2 damage classification.

The teacher uses a single shared SegFormer-B3 encoder that sees ``pre`` and
``post`` disaster images independently. The resulting feature pyramids are
fused per-stage via ``[pre | post | |pre - post|]`` concatenation and passed
through the MiT decode head for 6-class damage segmentation. Two auxiliary
heads (change + disaster type) anchor multi-task learning, and optional
auxiliary damage heads at intermediate encoder stages enable deep
supervision.

References
----------
- Xie et al. 2021 — SegFormer (NeurIPS).
- Zhao et al. 2017 — PSPNet auxiliary loss / deep supervision (CVPR).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SiameseTeacherSegformerV3(nn.Module):
    """Siamese SegFormer-B3 damage classifier.

    Parameters
    ----------
    backbone_name:
        HuggingFace id of the SegFormer backbone to use.
    num_damage_classes:
        Number of damage classes (default 6: background + no/minor/major/
        destroyed + unclassified).
    num_disaster_classes:
        Number of disaster type classes used by the auxiliary disaster head.
    pretrained:
        Whether to load pretrained backbone weights.
    use_deep_supervision:
        When ``True``, auxiliary heads are attached to the first 3 encoder
        stages; their losses are added to the main loss.
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

        if pretrained:
            base_model = SegformerForSemanticSegmentation.from_pretrained(
                backbone_name,
                num_labels=num_damage_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(backbone_name)
            config.num_labels = num_damage_classes
            base_model = SegformerForSemanticSegmentation(config)

        self.encoder = base_model.segformer.encoder
        self.decode_head = base_model.decode_head

        encoder_channels: List[int] = base_model.config.hidden_sizes  # e.g. [64, 128, 320, 512]
        self.encoder_channels = encoder_channels

        # Per-stage Siamese fusion: [pre | post | |pre - post|] -> c channels.
        self.fusion_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch * 3, ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                )
                for ch in encoder_channels
            ]
        )

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
                    for ch in encoder_channels[:-1]
                ]
            )
        else:
            self.aux_heads = None

        last_dim = encoder_channels[-1]
        self.change_head = nn.Sequential(
            nn.Conv2d(last_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 2, kernel_size=1),
        )

        self.disaster_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_disaster_classes),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return the list of 4 hierarchical encoder features."""
        outputs = self.encoder(x, output_hidden_states=True, return_dict=True)
        return list(outputs.hidden_states)

    def _fuse_features(
        self,
        features_pre: List[torch.Tensor],
        features_post: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Per-stage Siamese fusion."""
        fused: List[torch.Tensor] = []
        for i, (pre, post) in enumerate(zip(features_pre, features_post)):
            diff = torch.abs(post - pre)
            combined = torch.cat([pre, post, diff], dim=1)
            fused_feat = self.fusion_convs[i](combined)
            fused.append(fused_feat)
        return fused

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """Run Siamese forward pass.

        Parameters
        ----------
        x:
            Tensor of shape ``[B, 6, H, W]`` where channels ``0:3`` are the
            ``pre_disaster`` image and ``3:6`` are the ``post_disaster`` image.

        Returns
        -------
        dict
            ``damage_logits`` (main + aux heads when deep supervision),
            ``change_logits`` and ``disaster_logits``.
        """
        _, c, h, w = x.shape
        assert c == 6, f"Expected 6 channels (pre|post), got {c}"

        pre = x[:, :3]
        post = x[:, 3:]

        features_pre = self._encode(pre)
        features_post = self._encode(post)
        fused_features = self._fuse_features(features_pre, features_post)

        decoder_output = self.decode_head(fused_features)
        damage_main = F.interpolate(
            decoder_output, size=(h, w), mode="bilinear", align_corners=False
        )

        damage_output: Union[torch.Tensor, List[torch.Tensor]]
        if self.use_deep_supervision and self.aux_heads is not None:
            damage_logits_list: List[torch.Tensor] = [damage_main]
            for i, aux_head in enumerate(self.aux_heads):
                aux_logits = aux_head(fused_features[i])
                aux_logits = F.interpolate(
                    aux_logits, size=(h, w), mode="bilinear", align_corners=False
                )
                damage_logits_list.append(aux_logits)
            damage_output = damage_logits_list
        else:
            damage_output = damage_main

        last_fused = fused_features[-1]
        change_low = self.change_head(last_fused)
        change_logits = F.interpolate(
            change_low, size=(h, w), mode="bilinear", align_corners=False
        )

        disaster_logits = self.disaster_head(last_fused)

        return {
            "damage_logits": damage_output,
            "change_logits": change_logits,
            "disaster_logits": disaster_logits,
        }

    def num_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_gradient_checkpointing(self) -> bool:
        """Enable encoder gradient checkpointing if supported."""
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
            return True
        return False

    def load_encoder_from_localizer(
        self, localizer_state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[int, List[str]]:
        """Transfer encoder weights from a trained Phase 1 localizer.

        Parameters
        ----------
        localizer_state_dict:
            The dict returned by :meth:`LocalizerSegformer.get_encoder_state_dict`.

        Returns
        -------
        tuple(int, list[str])
            Number of weights copied and the list of skipped keys.
        """
        own_state = self.encoder.state_dict()
        loaded_count = 0
        skipped: List[str] = []
        for name, param in localizer_state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded_count += 1
            else:
                skipped.append(name)
        return loaded_count, skipped
