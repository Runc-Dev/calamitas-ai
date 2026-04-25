"""Exponential Moving Average (EMA) weight averaging.

EMA maintains a shadow copy of the model's parameters that is updated as a
geometric average of the optimizer's trajectory. Evaluating against EMA
weights almost always gives a smoother and stronger final model, especially
with cosine-warm-restart schedules that rebound the loss.

References
----------
- Izmailov, P. et al. (2018). Averaging Weights Leads to Wider Optima and
  Better Generalization (SWA). *UAI*.
- He, K. et al. (2020). Momentum Contrast for Unsupervised Visual
  Representation Learning — popularized EMA for large models.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ModelEMA:
    """Maintain an exponential moving average of a model's parameters.

    Parameters
    ----------
    model:
        The live model whose parameters should be tracked.
    decay:
        EMA decay factor ``alpha``. Shadow = ``alpha * shadow + (1-alpha) * live``.
        A typical value is 0.999.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update the shadow weights toward the live weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Copy the shadow weights into ``model`` for evaluation.

        Returns
        -------
        dict
            A backup of the original parameters that must be passed to
            :meth:`restore` when the EMA evaluation is finished.
        """
        backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        """Restore the live weights after an EMA evaluation."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
