"""AFETSONAR demo — runs inference on bundled sample images.

Creates a colourised damage mask and a basic Folium map.
No GPU required (runs on CPU in ~2s on a modern laptop).

Usage::

    python demo/sample_inference.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running from the demo/ directory without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Windows consoles often default to a legacy codepage (e.g. cp1254) that
# cannot encode the Unicode symbols printed below.
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _make_sample_images(tmp_dir: Path) -> tuple[str, str]:
    """Create tiny synthetic pre/post images for a self-contained demo."""
    import cv2
    import numpy as np

    np.random.seed(0)
    pre  = np.random.randint(80, 200, (256, 256, 3), dtype=np.uint8)
    post = pre.copy()
    # Simulate a destroyed building patch
    post[100:140, 80:130] = [30, 20, 20]
    post[60:100,  110:160] = [50, 40, 30]

    pre_path  = str(tmp_dir / "pre_sample.png")
    post_path = str(tmp_dir / "post_sample.png")
    cv2.imwrite(pre_path,  cv2.cvtColor(pre,  cv2.COLOR_RGB2BGR))
    cv2.imwrite(post_path, cv2.cvtColor(post, cv2.COLOR_RGB2BGR))
    return pre_path, post_path


def main() -> None:
    import tempfile

    print("AFETSONAR — Demo inference (synthetic images, random weights)")
    print("=" * 60)

    # Create a temporary student model with random weights
    import torch
    from afetsonar.models import StudentSiameseSegformer

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Random checkpoint
        model = StudentSiameseSegformer(pretrained=False)
        ckpt_path = str(tmp / "demo_student.pth")
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        print(f"Demo checkpoint: {ckpt_path}")
        print(f"Parameters: {model.num_parameters()/1e6:.1f}M")

        # Sample images
        pre_path, post_path = _make_sample_images(tmp)
        print(f"Sample images: {post_path}")

        # Pipeline
        from afetsonar import AfetsonarPipeline
        pipeline = AfetsonarPipeline(ckpt_path, device="cpu")

        mask = pipeline.predict(post_path, pre_path)
        print(f"\nDamage mask shape : {mask.shape}")
        print(f"Unique classes    : {sorted(set(mask.flatten().tolist()))}")

        buildings = pipeline.mask_to_buildings(
            mask,
            bbox_latlon=(41.003, 28.975, 41.008, 28.981),
        )
        print(f"Buildings detected: {len(buildings)}")
        for b in buildings[:5]:
            print(
                f"  #{b['building_id']:2d} {b['damage_class_name']:<15} "
                f"area={b['area_m2']:.0f}m²  "
                f"lat={b.get('lat', 0):.5f}  lon={b.get('lon', 0):.5f}"
            )

        # Save outputs
        out_dir = ROOT / "results"
        out_dir.mkdir(exist_ok=True)

        import cv2
        import numpy as np
        PALETTE = np.array([
            [128, 128, 128], [0, 200, 0], [0, 255, 255],
            [0, 128, 255], [0, 0, 255], [200, 0, 200],
        ], dtype=np.uint8)
        color = PALETTE[mask.clip(0, 5)]
        cv2.imwrite(
            str(out_dir / "demo_mask.png"),
            cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
        )
        print(f"\n✅  Demo mask saved → {out_dir / 'demo_mask.png'}")

    print("\nDemo complete.  To run on real data:")
    print("  python scripts/run_pipeline.py \\")
    print("      --post  <post_image.png> \\")
    print("      --pre   <pre_image.png>  \\")
    print("      --model checkpoints/student/student_v1_best_ema.pth \\")
    print("      --bbox  41.003,28.975,41.008,28.981")


if __name__ == "__main__":
    main()
