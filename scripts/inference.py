"""Single-image inference CLI.

Usage
-----
Run the student model on a post/pre image pair and optionally produce a
geo-referenced interactive map::

    python scripts/inference.py \\
        --post  path/to/post_disaster.png \\
        --pre   path/to/pre_disaster.png \\
        --model checkpoints/student/student_v1_best_ema.pth \\
        --output results/prediction.png \\
        --bbox  41.003,28.975,41.008,28.981 \\
        --map   results/map.html

Outputs
-------
- ``--output``  : PNG with colour-coded damage classes overlaid on the
  post-disaster image.
- ``--map``     : Self-contained HTML Folium map (requires ``--bbox``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# Class colour palette (BGR for OpenCV)
_PALETTE = {
    0: (128, 128, 128),   # background  — grey
    1: (0, 200, 0),       # no_damage   — green
    2: (0, 255, 255),     # minor       — yellow
    3: (0, 128, 255),     # major       — orange
    4: (0, 0, 255),       # destroyed   — red
    5: (200, 0, 200),     # unclassified — purple
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AFETSONAR single-image damage inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--post",   required=True, help="Post-disaster image path")
    p.add_argument("--pre",    default=None,  help="Pre-disaster image path (optional)")
    p.add_argument("--model",  required=True, help="Checkpoint path (.pth)")
    p.add_argument("--output", default="results/prediction.png", help="Output PNG path")
    p.add_argument("--map",    default=None,  help="Output HTML map path")
    p.add_argument(
        "--bbox",
        default=None,
        help="Geographic bbox: lat_min,lon_min,lat_max,lon_max",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--alpha",  type=float, default=0.5, help="Overlay transparency (0-1)")
    return p.parse_args()


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a class-index mask to an RGB colour image."""
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, bgr in _PALETTE.items():
        color_img[mask == cls] = bgr[::-1]  # BGR → RGB
    return color_img


def main() -> None:
    args = parse_args()

    # Import here so the script can be called without PYTHONPATH tricks
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from afetsonar import AfetsonarPipeline

    print(f"Loading model: {args.model}")
    pipeline = AfetsonarPipeline(args.model, device=args.device)

    print(f"Running inference on: {args.post}")
    mask = pipeline.predict(args.post, args.pre)

    # Save colourised overlay
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    post_img = cv2.imread(args.post)
    if post_img is None:
        print(f"Warning: cannot read {args.post} for overlay.")
        post_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    h, w = mask.shape
    post_resized = cv2.resize(post_img, (w, h))
    color_mask = colorize_mask(mask)
    overlay = cv2.addWeighted(
        post_resized,
        1 - args.alpha,
        cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR),
        args.alpha,
        0,
    )
    cv2.imwrite(args.output, overlay)
    print(f"Saved prediction overlay → {args.output}")

    # Class distribution
    unique, counts = np.unique(mask, return_counts=True)
    class_names = ["background", "no_damage", "minor", "major", "destroyed", "unclassified"]
    total = mask.size
    print("\nClass distribution:")
    for c, n in zip(unique, counts):
        name = class_names[c] if c < len(class_names) else f"cls_{c}"
        print(f"  {name:<15}: {n:>8}  ({n/total*100:.1f}%)")

    # Optional map
    if args.map and args.bbox:
        try:
            lat_min, lon_min, lat_max, lon_max = map(float, args.bbox.split(","))
            bbox = (lat_min, lon_min, lat_max, lon_max)
            html_path = pipeline.generate_map(
                post_path=args.post,
                pre_path=args.pre,
                bbox_latlon=bbox,
                hospitals=[],
                output_path=args.map,
            )
            print(f"Saved map → {html_path}")
        except Exception as e:
            print(f"Map generation failed: {e}")


if __name__ == "__main__":
    main()
