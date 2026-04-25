"""Binary building localization model (Phase 1).

A single SegFormer encoder-decoder that segments pre-disaster imagery into
background vs. building pixels.  Weights are later transferred to the Siamese
teacher encoder.

References
----------
- Xie et al. 2021 — SegFormer: Simple and Efficient Design for Semantic
  Segmentation with Transformers (NeurIPS 2021).  arXiv:2105.15203.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class LocalizerSegformer(nn.Module):
    """Phase-1 building localization model.

    A thin wrapper around HuggingFace SegformerForSemanticSegmentation that
    resizes the low-resolution decoder output back to the input resolution.

    Args:
        backbone_name: HuggingFace model id (or local path) used both to load
            weights and to derive the encoder architecture.
        pretrained: If ``True``, loads the public pretrained checkpoint;
            otherwise builds the model from config only (useful for unit
            tests without internet access).

    Example:
        >>> model = LocalizerSegformer()
        >>> x = torch.randn(2, 3, 512, 512)
        >>> logits = model(x)          # shape (2, 2, 512, 512)
    """

    NUM_CLASSES: int = 2  # background / building

    def __init__(
        self,
        backbone_name: str = "nvidia/segformer-b3-finetuned-ade-512-512",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if pretrained:
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                backbone_name,
                num_labels=self.NUM_CLASSES,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(backbone_name)
            config.num_labels = self.NUM_CLASSES
            self.segformer = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: RGB image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Logit tensor of shape ``(B, 2, H, W)`` — same spatial size as
            the input.
        """
        B, C, H, W = x.shape
        logits = self.segformer(x).logits
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def get_encoder_state_dict(self) -> dict:
        """Return a copy of the encoder weights for transfer to the teacher."""
        return {
            name: param.clone()
            for name, param in self.segformer.segformer.encoder.state_dict().items()
        }

    def enable_gradient_checkpointing(self) -> bool:
        """Enable gradient checkpointing on the encoder (saves VRAM).

        Returns:
            ``True`` if the operation succeeded, ``False`` otherwise.
        """
        enc = self.segformer.segformer.encoder
        if hasattr(enc, "gradient_checkpointing_enable"):
            enc.gradient_checkpointing_enable()
            return True
        return False

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
