"""Single-image SegFormer localizer for Phase 1 (binary building segmentation).

This module contains the :class:`LocalizerSegformer` used in Phase 1 of the
AFETSONAR training pipeline. Its sole job is to learn strong building priors
from the ``pre_disaster`` image; the trained encoder is then transferred to
the Siamese teacher for Phase 2 damage classification.

References
----------
Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021).
SegFormer: Simple and Efficient Design for Semantic Segmentation with
Transformers. *NeurIPS*.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class LocalizerSegformer(nn.Module):
    """Binary building localizer built on top of SegFormer-B3.

    Parameters
    ----------
    backbone_name:
        HuggingFace model id for the SegFormer backbone to load. Defaults to
        the ADE20K-pretrained B3 checkpoint.
    pretrained:
        If ``True`` (default) the pretrained weights are downloaded; otherwise
        the backbone is instantiated from its config with random weights.
    """

    def __init__(
        self,
        backbone_name: str = "nvidia/segformer-b3-finetuned-ade-512-512",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if pretrained:
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                backbone_name,
                num_labels=2,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(backbone_name)
            config.num_labels = 2
            self.segformer = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x:
            Image tensor of shape ``[B, 3, H, W]``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``[B, 2, H, W]`` upsampled to the input resolution.
        """
        _, _, h, w = x.shape
        outputs = self.segformer(x)
        logits = outputs.logits
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits

    def get_encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the encoder weights as a state dict (for Phase 2 transfer)."""
        encoder_state: Dict[str, torch.Tensor] = {}
        for name, param in self.segformer.segformer.encoder.state_dict().items():
            encoder_state[name] = param
        return encoder_state

    def enable_gradient_checkpointing(self) -> bool:
        """Enable gradient checkpointing on the encoder to save memory.

        Returns ``True`` if the encoder supported checkpointing, else ``False``.
        """
        if hasattr(self.segformer.segformer.encoder, "gradient_checkpointing_enable"):
            self.segformer.segformer.encoder.gradient_checkpointing_enable()
            return True
        return False

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
