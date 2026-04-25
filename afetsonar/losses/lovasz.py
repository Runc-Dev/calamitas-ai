"""Lovász-Softmax loss — directly optimises mIoU.

Unlike cross-entropy (which optimises per-pixel accuracy), Lovász-Softmax
optimises the Jaccard index (IoU) as a surrogate.  It was used by the xView2
competition winners and is the primary loss term in AFETSONAR's combo loss.

References
----------
- Berman et al. 2018 — The Lovász-Softmax Loss: A Tractable Surrogate for
  the Optimization of the Intersection-Over-Union Measure in Neural Networks.
  CVPR 2018.  arXiv:1705.08790.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Low-level utilities
# ============================================================

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute the gradient of the Lovász extension w.r.t. sorted errors."""
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten spatial probabilities and labels; optionally filter ignore pixels."""
    if probas.dim() == 3:
        probas = probas.unsqueeze(1)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    return probas[valid], labels[valid]


def _lovasz_softmax_flat(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: Union[str, List[int]] = "present",
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lovász-Softmax on already-flattened probabilities/labels."""
    if probas.numel() == 0:
        return probas.sum() * 0.0

    C = probas.size(1)
    class_to_sum = list(range(C)) if classes in ("all", "present") else list(classes)
    losses, weights = [], []

    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        class_pred = probas[:, c] if C > 1 else probas[:, 0]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
        if class_weights is not None:
            weights.append(class_weights[c])

    if not losses:
        return probas.sum() * 0.0

    losses_t = torch.stack(losses)
    if class_weights is not None and weights:
        w = torch.stack(weights).to(losses_t.device)
        return (losses_t * w).sum() / w.sum()
    return losses_t.mean()


# ============================================================
# Public module
# ============================================================

class LovaszSoftmaxLoss(nn.Module):
    """Lovász-Softmax multi-class segmentation loss.

    Args:
        classes: Which classes to include in the mean.  ``"present"`` skips
            classes absent from a given batch (recommended); ``"all"`` always
            averages over all C classes.
        ignore_index: Label value to exclude from the loss computation.
        class_weights: Optional 1-D tensor of per-class weights (length C).
    """

    def __init__(
        self,
        classes: str = "present",
        ignore_index: int = -100,
        class_weights: Optional[List[float]] = None,
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
            self.class_weights: Optional[torch.Tensor] = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Lovász-Softmax loss.

        Args:
            logits: Raw logit tensor ``(B, C, H, W)``.
            targets: Integer label tensor ``(B, H, W)``.

        Returns:
            Scalar loss tensor.
        """
        probas = F.softmax(logits, dim=1)
        flat_p, flat_l = _flatten_probas(probas, targets, self.ignore_index)
        return _lovasz_softmax_flat(
            flat_p, flat_l,
            classes=self.classes,
            class_weights=self.class_weights,
        )
