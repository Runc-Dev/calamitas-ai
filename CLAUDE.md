# AFETSONAR — Claude Code Guide

Drone/uydu görüntülerinden afet sonrası bina hasar tespiti ve kurtarma rota planlama.
Teknofest 2025.

## Project layout

```
afetsonar/          ← Python package (import afetsonar)
  models/           ← teacher.py, student.py, segformer.py, ema.py
  data/             ← dataset.py (XBDDatasetV2), augmentations.py, preprocessing.py
  losses/           ← lovasz.py, combo.py, distillation.py, localization.py
  routing/          ← priority.py, astar.py, tsp.py, helicopter.py, team_assignment.py
  geo/              ← utils.py, geotiff.py, map_builder.py
  evaluation/       ← metrics.py, ablation.py
  config.py         ← DefaultConfig dataclass, all constants
  pipeline.py       ← AfetsonarPipeline (end-to-end)
  utils.py          ← visualisation helpers
scripts/            ← CLI tools: train_teacher.py, inference.py, evaluate.py
tests/              ← pytest
configs/            ← YAML hyperparameters (default.yaml, teacher.yaml, student.yaml)
```

## Models

| Model | Backbone | Params | Latency | mIoU (test) | mF1 (test) |
|-------|----------|--------|---------|-------------|------------|
| Teacher (`SiameseTeacherSegformerV3`) | SegFormer-B3 | 50.3M | ~1140ms | 0.424 | 0.640 |
| Student (`StudentSiameseSegformer`) | SegFormer-B0 | 4.3M | 36ms | 0.395 | 0.617 |
| Localizer (`LocalizerSegformer`) | SegFormer-B3 | ~45M | — | Building IoU 0.756 | — |

KD efficiency: 93.2% (student retains 93% of teacher knowledge).

## Dataset

xBD (xView2 challenge), Tier 1 + Tier 3 = 9 168 images across 19 disaster events.
Train/Val/Test = 6418/1375/1375.

**6 damage classes:**  
0=background, 1=no_damage, 2=minor_damage, 3=major_damage, 4=destroyed, 5=unclassified

**Class pixel distribution:** bg 93%, no_damage 4.2%, minor 0.53%, major 0.92%, destroyed 1.1%, unclassified 0.25%

## Key design decisions

- **Siamese encoder** (shared weights for pre and post images) — not twin/dual encoder
- **Fusion** per encoder stage: `concat(pre, post, |post-pre|)` → 1×1 conv → original channels
- **Deep supervision**: aux damage heads at intermediate SegFormer stages
- **5-component KD loss** (student): 0.30·L_hard + 0.40·L_soft + 0.15·L_feature + 0.10·L_change + 0.05·L_disaster
- **Loss (teacher)**: 0.35·Lovász + 0.35·Dice + 0.30·Focal
- **EMA** (decay=0.999) — inference uses shadow weights

## Environment

- **Training**: Google Colab Pro+ H100 GPU
- **Inference**: edge device (Jetson Nano / Xavier) or CPU
- **Local machine**: code editing only, torch NOT installed locally
- Python 3.10+, transformers (HuggingFace), albumentations 1.x or 2.x

## Running training (Colab)

```bash
pip install -e .
python scripts/train_teacher.py --config configs/teacher.yaml
python scripts/train_student.py --config configs/student.yaml
```

## Running inference (local, no GPU)

```python
from afetsonar import AfetsonarPipeline
pipeline = AfetsonarPipeline("checkpoints/student/student_v1_best_ema.pth", device="cpu")
mask = pipeline.predict("post.png", "pre.png")
```

## Running tests

```bash
pip install -e ".[dev]"   # installs pytest + test deps
pytest tests/ -v
```

Tests that require torch are auto-skipped when torch is absent.

## 5-phase roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Repo structure | ✅ Done | Package + all modules |
| 2 — Auto pre-fetch | ✅ Done | AutoPreFetcher (Google Maps/Mapbox API) |
| 3 — Incremental training | ✅ Done | AfetsonarTrainer (resume, add_data, ablation) |
| 4 — Gradio web app | 🔲 Next | HuggingFace Spaces deploy |
| 5 — SoTA improvement | 🔲 | TTA, SWA, backbone upgrade, pseudo-labeling |

## Current performance plateau

Teacher hits mIoU ≈ 0.47 on validation, 0.424 on test. Root causes:
1. xBD label noise (polygon GT errors ±2-3px)
2. Class imbalance (minor_damage 0.53% of pixels)
3. minor ↔ major ↔ no_damage confusion

Planned fixes (Phase 3/5): TTA (+0.03–0.05), SWA (+0.01–0.03), Copy-Paste aug, pseudo-labeling.
