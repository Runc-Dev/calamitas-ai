"""Full end-to-end pipeline CLI — image → interactive map.

Runs the complete AFETSONAR pipeline in a single command:
  1. Load student model checkpoint.
  2. Run damage inference on the post/pre image pair.
  3. Extract buildings, compute FEMA priority + survival scores.
  4. Assign rescue teams (K-means, priority-weighted).
  5. Build an 8-layer Folium HTML map.

Usage::

    python scripts/run_pipeline.py \\
        --post   data/samples/post.png \\
        --pre    data/samples/pre.png  \\
        --model  checkpoints/student/student_v1_best_ema.pth \\
        --bbox   41.003,28.975,41.008,28.981 \\
        --output results/afetsonar_map.html

Optional hospital list (JSON)::

    --hospitals '[{"name":"Cerrahpaşa","lat":41.0048,"lon":28.9510}]'
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AFETSONAR full pipeline: image → interactive map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--post",      required=True,  help="Post-disaster image path")
    p.add_argument("--pre",       default=None,   help="Pre-disaster image path")
    p.add_argument("--model",     required=True,  help="Checkpoint .pth path")
    p.add_argument(
        "--bbox", required=True,
        help="Geographic bounding box: lat_min,lon_min,lat_max,lon_max"
    )
    p.add_argument("--output",    default="results/afetsonar_map.html")
    p.add_argument(
        "--hospitals", default="[]",
        help='JSON array of hospital dicts, e.g. \'[{"name":"H","lat":41,"lon":29}]\''
    )
    p.add_argument("--n-teams",   type=int, default=5)
    p.add_argument("--device",    default="auto")
    p.add_argument("--config",    default=None,   help="Optional YAML config path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from afetsonar import AfetsonarPipeline
    from afetsonar.config import DefaultConfig

    # Config
    cfg = None
    if args.config and Path(args.config).exists():
        cfg = DefaultConfig.from_yaml(args.config)

    # Parse bbox
    try:
        lat_min, lon_min, lat_max, lon_max = map(float, args.bbox.split(","))
    except ValueError:
        print("ERROR: --bbox must be 'lat_min,lon_min,lat_max,lon_max'")
        sys.exit(1)

    # Parse hospitals
    try:
        hospitals = json.loads(args.hospitals)
    except json.JSONDecodeError:
        print("ERROR: --hospitals must be valid JSON array")
        sys.exit(1)

    # Default hospitals for Sultanahmet demo
    if not hospitals:
        hospitals = [
            {"id": 0, "name": "Cerrahpaşa Tıp Fakültesi",         "lat": 41.0048, "lon": 28.9510},
            {"id": 1, "name": "Haseki Eğitim ve Araştırma Hast.", "lat": 41.0117, "lon": 28.9447},
            {"id": 2, "name": "İstanbul Üniv. Tıp Fakültesi",     "lat": 41.0099, "lon": 28.9651},
            {"id": 3, "name": "Sultanahmet Devlet Hastanesi",     "lat": 41.0058, "lon": 28.9756},
        ]
        print(f"Using {len(hospitals)} default Istanbul hospitals.")

    # ── Pipeline ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AFETSONAR PIPELINE")
    print(f"{'='*60}")
    print(f"Post image : {args.post}")
    print(f"Pre image  : {args.pre or '(none — post duplicated)'}")
    print(f"Model      : {args.model}")
    print(f"BBox       : {lat_min:.4f},{lon_min:.4f} → {lat_max:.4f},{lon_max:.4f}")
    print(f"Output     : {args.output}")
    print()

    t_start = time.time()

    pipeline = AfetsonarPipeline(args.model, config=cfg, device=args.device)
    print(f"[1/4] Model loaded in {time.time()-t_start:.1f}s")

    t1 = time.time()
    analysis = pipeline.analyze(
        args.post, args.pre,
        bbox_latlon=(lat_min, lon_min, lat_max, lon_max),
    )
    mask      = analysis["mask"]
    buildings = analysis["buildings"]
    print(f"[2/4] Inference + analysis done in {time.time()-t1:.1f}s | {len(buildings)} buildings detected")

    if buildings:
        dmg_counts: dict = {}
        for b in buildings:
            dmg_counts[b["damage_class_name"]] = dmg_counts.get(b["damage_class_name"], 0) + 1
        for cls, n in sorted(dmg_counts.items()):
            print(f"      {cls:<20}: {n} buildings")

    t2 = time.time()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    html_path = pipeline.generate_map(
        post_path=args.post,
        pre_path=args.pre,
        bbox_latlon=(lat_min, lon_min, lat_max, lon_max),
        hospitals=hospitals,
        output_path=args.output,
        n_teams=args.n_teams,
    )
    print(f"[3/4] Map generated in {time.time()-t2:.1f}s")

    # Save mask PNG
    mask_out = Path(args.output).with_name("damage_mask.png")
    import cv2
    import numpy as np
    _PALETTE = np.array([
        [128, 128, 128], [0, 200, 0], [0, 255, 255],
        [0, 128, 255], [0, 0, 255], [200, 0, 200],
    ], dtype=np.uint8)
    color_mask = _PALETTE[mask.clip(0, 5)]
    cv2.imwrite(str(mask_out), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

    total = time.time() - t_start
    print(f"[4/4] All done in {total:.1f}s total")
    print(f"\n✅  Map   → {html_path}")
    print(f"✅  Mask  → {mask_out}")


if __name__ == "__main__":
    main()
