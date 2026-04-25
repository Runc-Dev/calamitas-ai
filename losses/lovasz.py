"""Lovász-Softmax loss — a direct differentiable surrogate of the Jaccard index.

This module implements the multi-class Lovász-Softmax loss described by
Berman, Triki & Blaschko (2018). Unlike cross-entropy (which is only a proxy
for accuracy), Lovász directly optimizes the intersection-over-union metric
that is reported at evaluation time, so it is a near-universal booster for
segmentation heads; both xView2 winners used it.

References
----------
Berman, M., Rannen Triki, A., & Blaschko, M. B. (2018). The Lovász-Softmax
Loss: A Tractable Surrogate for the Optimization of the Intersection-over-
Union Measure in Neural Networks. *CVPR*.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Lovász gradient helpers
# ----------------------------------------------------------------------

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute the Lovász extension of the Jaccard loss gradient.

    Implements Alg. 1 of Berman et al. 2018. ``gt_sorted`` is the binary
    ground-truth after sorting prediction errors in descending order.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _flatten_probas(
    probas: torch.Tensor,
    labels: torch.Tensor,
    ignore: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten ``[B, C, H, W]`` probas and ``[B, H, W]`` labels and drop ``ignore``."""
    if probas.dim() == 3:
        probas = probas.unsqueeze(1)
    _, c, _, _ = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, c)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    return probas[valid], labels[valid]


def _lovasz_softmax_flat(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: Union[str, Iterable[int]] = "present",
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-class Lovász-Softmax on flattened probabilities."""
    if probas.numel() == 0:
        return probas * 0.0

    c = probas.size(1)
    losses = []
    weights_list = []
    class_to_sum = list(range(c)) if classes in ("all", "present") else list(classes)

    for cls in class_to_sum:
        fg = (labels == cls).float()
        if classes == "present" and fg.sum() == 0:
            continue
        class_pred = probas[:, 0] if c == 1 else probas[:, cls]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
        if class_weights is not None:
            weights_list.append(class_weights[cls])

    if len(losses) == 0:
        return probas.sum() * 0.0

    losses_tensor = torch.stack(losses)
    if class_weights is not None and len(weights_list) > 0:
        weights = torch.stack(weights_list).to(losses_tensor.device)
        return (losses_tensor * weights).sum() / weights.sum()
    return losses_tensor.mean()


# ----------------------------------------------------------------------
# Public module
# ----------------------------------------------------------------------

class LovaszSoftmaxLoss(nn.Module):
    """Multi-class Lovász-Softmax loss as a PyTorch ``nn.Module``.

    Parameters
    ----------
    classes:
        ``"present"`` (only averaged over classes present in the batch, default),
        ``"all"`` (averaged over every class), or a sequence of class indices.
    ignore_index:
        Label value to exclude from the loss (typically ``255`` or ``-100``).
    class_weights:
        Optional per-class weights used when averaging the per-class losses.
    """

    def __init__(
        self,
        classes: Union[str, Sequence[int]] = "present",
        ignore_index: int = -100,
        class_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probas = F.softmax(logits, dim=1)
        flat_probas, flat_labels = _flatten_probas(probas, targets, self.ignore_index)
        return _lovasz_softmax_flat(
            flat_probas,
            flat_labels,
            classes=self.classes,
            class_weights=self.class_weights,
        )
