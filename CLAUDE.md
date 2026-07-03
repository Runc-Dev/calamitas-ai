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

| Model | Backbone | Params | Latency | mIoU (test) | mF1 (test) | mIoU (val) | mF1 (val) |
|-------|----------|--------|---------|-------------|------------|------------|-----------|
| Teacher (`SiameseTeacherSegformerV3`) | SegFormer-B3 | 50.3M | ~1140ms | 0.424 | 0.640 | 0.5586 | 0.6866 |
| Student (`StudentSiameseSegformer`) | SegFormer-B0 | 4.3M | 36ms | 0.395 | 0.617 | 0.439* | — |
| Localizer (`LocalizerSegformer`) | SegFormer-B3 | ~45M | — | Building IoU 0.756 | — | — | — |

*Student val_miou_no_bg (no background class).

KD efficiency: 93.2% (student retains 93% of teacher knowledge).

### Teacher v4 — per-class validation metrics (best epoch 70 / 80)

| Class | IoU | F1 |
|-------|-----|----|
| background | 0.990 | 0.995 |
| no_damage | 0.699 | 0.823 |
| minor_damage | 0.289 | 0.448 |
| major_damage | 0.438 | 0.609 |
| destroyed | 0.623 | 0.767 |
| unclassified | 0.314 | 0.478 |

Checkpoint: `teacher_v4_best_ema.pth` (Drive ID: `1866pHX-_PDErPnYERQw65cpXEKLELyoI`, 192 MB)

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

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Repo structure | ✅ Done | Package + all modules |
| 2 — Auto pre-fetch | ✅ Done | AutoPreFetcher (Google Maps/Mapbox API) |
| 3 — Incremental training | ✅ Done | AfetsonarTrainer (resume, add_data, ablation) |
| 4 — Gradio web app | ✅ Done | HuggingFace Spaces deploy (app.py) |
| 5 — SoTA improvement | ✅ Done | TTA (8 transforms + multi-scale), Copy-Paste aug |
| 6 — Global support + Mobile | ✅ Done | FastAPI backend (`api/`), React Native app (`mobile/`) |

## Drive alignment fixes (applied 2026-05-03)

| File | Before | After | Reason |
|------|--------|-------|--------|
| `config.py` IMAGE_SIZE | 768 | **512** | Drive training used 512 |
| `augmentations.py` default | 768 | **512** | Match Drive `augmentations_v2.py` |
| `astar.py` LON_M | Istanbul 41°N hardcode | **dynamic (mean lat of buildings)** | Global routing support |
| `config.py` POPULATION_DENSITY comment | "Fatih/Sultanahmet" | generic | Global support |

## Global Architecture (Phase 6)

```
api/
  main.py          ← FastAPI REST backend (POST /analyze, GET /health)
  requirements.txt

mobile/
  App.js           ← React Native (Expo) root
  app.json         ← Expo config (iOS + Android)
  src/
    screens/
      HomeScreen.js    ← Image picker + GPS + Analyze button
      ResultScreen.js  ← Mask image + stats + buildings + satellite map
    api/
      client.js        ← axios calls to FastAPI backend
    utils/
      colors.js        ← damage class colors (global, no city hardcoding)
```

### Running the API

```bash
pip install -r api/requirements.txt
AFETSONAR_CHECKPOINT=checkpoints/student/student_v1_best_ema.pth \
    uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Running the mobile app

```bash
cd mobile
npm install
# Edit src/api/client.js — set BASE_URL to your server IP
npx expo start
# Scan QR code with Expo Go (iOS/Android)
```

## Phase 5 — What was implemented

| Teknik | Dosya | Beklenen kazanım |
|--------|-------|-----------------|
| TTA (8 geometric + multi-scale) | `afetsonar/evaluation/tta.py` — `TTAWrapper` | +0.03–0.05 mF1 |
| Copy-Paste augmentation | `afetsonar/data/copy_paste.py` — `CopyPasteAugmentation`, `CopyPasteDataset` | +0.02–0.04 mF1 |

### TTAWrapper usage
```python
from afetsonar import AfetsonarPipeline
from afetsonar.evaluation.tta import TTAWrapper

pipeline = AfetsonarPipeline("checkpoints/student_v1_best_ema.pth")
tta = TTAWrapper(pipeline, n_augmentations=8)          # 8 transforms
tta = TTAWrapper(pipeline, scales=(0.75, 1.0, 1.25))   # multi-scale
mask = tta.predict("post.png", "pre.png")
```

### CopyPasteAugmentation usage
```python
from afetsonar.data.copy_paste import CopyPasteDataset, CopyPasteAugmentation

aug     = CopyPasteAugmentation(paste_probability=0.5, damage_classes_to_paste=(2, 3, 4))
dataset = CopyPasteDataset(train_dataset, aug)
# Use dataset as drop-in replacement — donor selected randomly per batch
```

## Current performance plateau

Teacher hits mIoU ≈ 0.47 on validation, 0.424 on test. Root causes:
1. xBD label noise (polygon GT errors ±2-3px)
2. Class imbalance (minor_damage 0.53% of pixels)
3. minor ↔ major ↔ no_damage confusion

With Phase 5 TTA: expected mF1 ≈ 0.67–0.69 (no retraining).
With Copy-Paste retraining: expected mF1 ≈ 0.69–0.73.
