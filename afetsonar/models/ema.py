"""Exponential Moving Average (EMA) model wrapper.

EMA maintains a shadow copy of the model weights that evolves as a weighted
running average of the training weights.  At inference time the shadow weights
consistently outperform the raw training weights, especially near the end of
training.

Usage::

    ema = ModelEMA(model, decay=0.999)
    for batch in loader:
        optimizer.step()
        ema.update(model)

    # evaluate with EMA weights
    backup = ema.apply_to(model)
    val_metric = evaluate(model, val_loader)
    ema.restore(model, backup)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ModelEMA:
    """Shadow-weight EMA for stabilising late-stage training.

    Args:
        model: The model whose parameters will be shadowed.
        decay: EMA decay coefficient.  Typical values: 0.999 (aggressive
            smoothing for long runs) or 0.9999 (light smoothing).

    Note:
        Only parameters with ``requires_grad=True`` are shadowed.
        Buffers (BatchNorm running stats) are **not** tracked — call
        ``model.eval()`` during inference to use the frozen BN stats.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights with the current model weights.

        Call once per optimiser step, **after** ``optimizer.step()``.

        Args:
            model: The model being trained.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Swap model weights with EMA shadow weights in-place.

        Returns a backup dict so the training weights can be restored
        afterwards via :meth:`restore`.

        Args:
            model: Model to modify in-place.

        Returns:
            Backup dict mapping parameter names to original training weights.
        """
        backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        """Restore training weights from backup (undo :meth:`apply_to`).

        Args:
            model: Model that was modified by :meth:`apply_to`.
            backup: Dict returned by :meth:`apply_to`.
        """
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
