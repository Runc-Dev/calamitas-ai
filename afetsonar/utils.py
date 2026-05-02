"""Visualization and debugging utilities.

Provides colour maps, image denormalization, and sample visualization
helpers used in notebooks and training scripts.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# ============================================================
# Colour map — 6 damage classes
# ============================================================

#: RGB colours per damage class (0–5).
DAMAGE_COLORS: list[list[int]] = [
    [0,   0,   0],   # 0 background      — black
    [0,   200, 0],   # 1 no_damage       — green
    [255, 230, 0],   # 2 minor_damage    — yellow
    [255, 128, 0],   # 3 major_damage    — orange
    [220, 0,   0],   # 4 destroyed       — red
    [128, 0,   128], # 5 unclassified    — purple
]

DAMAGE_LABELS: list[str] = [
    "background",
    "no_damage",
    "minor_damage",
    "major_damage",
    "destroyed",
    "unclassified",
]

DAMAGE_LABELS_TR: list[str] = [
    "arka plan",
    "sağlam",
    "az hasar",
    "ağır hasar",
    "yıkık",
    "belirsiz",
]

DAMAGE_CMAP = ListedColormap(np.array(DAMAGE_COLORS, dtype=np.float32) / 255.0)


# ============================================================
# Image utilities
# ============================================================

def denormalize(
    tensor: "torch.Tensor",
    mean: Optional[list[float]] = None,
    std: Optional[list[float]] = None,
) -> "torch.Tensor":
    """Reverse ImageNet normalisation for a CHW tensor.

    Args:
        tensor: Float tensor ``(3, H, W)`` normalised with ImageNet stats.
        mean: Channel means (default ImageNet).
        std: Channel stds (default ImageNet).

    Returns:
        Un-normalised tensor in ``[0, 1]``.
    """
    import torch  # lazy import — avoids hard dependency at module load time

    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    m = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    s = torch.tensor(std,  dtype=tensor.dtype).view(3, 1, 1)
    return (tensor * s + m).clamp(0.0, 1.0)


# ============================================================
# Sample visualisation
# ============================================================

def visualize_sample(
    sample: dict,
    save_path: Optional[str] = None,
    lang: str = "en",
) -> None:
    """Display a dataset sample with pre, post, mask, and overlay panels.

    Args:
        sample: Dict with keys ``"image"`` (tensor C×H×W), ``"mask"``
            (tensor H×W), ``"disaster_idx"``, ``"filename"``.
        save_path: If given, save the figure to this path.
        lang: ``"en"`` or ``"tr"`` for label language.
    """
    image = sample["image"]
    mask  = sample["mask"]
    labels = DAMAGE_LABELS_TR if lang == "tr" else DAMAGE_LABELS

    # Split 6-channel (teacher) or 3-channel (student) tensor
    if image.shape[0] == 6:
        pre  = denormalize(image[:3]).permute(1, 2, 0).numpy()
        post = denormalize(image[3:]).permute(1, 2, 0).numpy()
        panels = 4
    else:
        pre  = None
        post = denormalize(image).permute(1, 2, 0).numpy()
        panels = 3

    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 5))
    idx = 0

    if pre is not None:
        axes[idx].imshow(pre)
        axes[idx].set_title("Pre-disaster" if lang == "en" else "Afet öncesi")
        axes[idx].axis("off")
        idx += 1

    axes[idx].imshow(post)
    axes[idx].set_title("Post-disaster" if lang == "en" else "Afet sonrası")
    axes[idx].axis("off")
    idx += 1

    mask_np = mask.numpy() if hasattr(mask, "numpy") else np.array(mask)
    axes[idx].imshow(mask_np, cmap=DAMAGE_CMAP, vmin=0, vmax=5)
    axes[idx].set_title("Damage mask" if lang == "en" else "Hasar maskesi")
    axes[idx].axis("off")
    idx += 1

    # Overlay: blend damage colour on post image
    overlay = post.copy()
    for cls in range(1, 6):
        color = np.array(DAMAGE_COLORS[cls], dtype=np.float32) / 255.0
        region = mask_np == cls
        overlay[region] = 0.5 * overlay[region] + 0.5 * color
    axes[idx].imshow(overlay)
    axes[idx].set_title("Overlay")
    axes[idx].axis("off")

    title = str(sample.get("filename", ""))
    if "disaster_idx" in sample:
        di = sample["disaster_idx"]
        val = di.item() if hasattr(di, "item") else int(di)
        title += f"  |  disaster_idx={val}"
    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
