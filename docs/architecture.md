# AFETSONAR Architecture

## Overview

```
Pre-disaster RGB ──┐                        ┌── Damage Logits (6 cls)
                   ├── Siamese Encoder ──── ┼── Change Logits (binary)
Post-disaster RGB ─┘   (SegFormer-B3/B0)   └── Disaster Type (5 cls)
                             │
                         [KD Loss]
                             │
                      Student (B0) → 36 ms edge inference
```

## Phase 1 — Building Localization

- **Model**: `LocalizerSegformer` (SegFormer-B3, 2-class output)
- **Input**: Single post-disaster RGB (3 channels)
- **Output**: Binary mask — background / building
- **Loss**: BCE + Dice (50/50)
- **Purpose**: Pre-trains the encoder for transfer to Phase 2

## Phase 2 — Teacher Training

- **Model**: `SiameseTeacherSegformerV3` (SegFormer-B3, ~50.3M params)
- **Input**: 6-channel tensor [pre_RGB | post_RGB]
- **Output**: 6-class damage mask + binary change mask + disaster type
- **Loss**: `TeacherLossV3` = 70% damage (Lovász+Dice+Focal+DeepSup) + 20% change + 10% disaster
- **Key techniques**:
  - Siamese feature fusion: `[pre | post | |pre - post|]` per stage
  - Deep supervision: 3 auxiliary heads at intermediate stages
  - EMA (decay=0.999) for model averaging
  - Cosine warm restarts: 3 cycles at epochs 25/50/75
  - Building-aware crop: 80% of patches contain building pixels
  - WeightedRandomSampler: 5× up-weight for damaged scenes

## Phase 3 — Student Distillation

- **Model**: `StudentSiameseSegformer` (SegFormer-B0, ~4.3M params)
- **Input**: Same 6-channel format
- **Loss**: 5-component KD loss:
  1. KD soft-label KL (α=0.30, T=4.0)
  2. CE hard-label (β=0.25)
  3. Feature matching MSE (γ=0.20)
  4. Attention transfer (δ=0.10)
  5. Combo damage Lovász+Dice+Focal (ε=0.15)
- **Result**: 12× smaller, 32.8× faster, 93% knowledge retention

## Routing Pipeline

```
Damage mask
    │
    ├── mask_to_buildings() ──── contour extraction (OpenCV)
    │
    ├── score_buildings() ──── FEMA priority + survival curve
    │
    ├── assign_teams() ──── priority-weighted K-means (n=5)
    │
    ├── apply_gradient_weights() ──── OSMnx graph + Shapely buffer
    │
    ├── nearest_neighbor_tsp() ──── Rosenkrantz et al. 1977
    │
    ├── astar_segment() ──── Hart et al. 1968 + Haversine heuristic
    │
    └── FoliumMapBuilder.save() ──── 8-layer interactive HTML
```

## Key Metrics (xBD test set, 1375 images)

| Model | mIoU_no_bg | mF1 | Destroyed IoU | Params | Latency |
|-------|-----------|-----|---------------|--------|---------|
| Teacher (B3) | 0.424 | 0.640 | 0.570 | 50.3M | ~1140ms |
| Student (B0) | 0.395 | 0.617 | 0.524 | 4.3M | 36ms ✅ |

## Ablation Journey

| Variant | mIoU_no_bg | Δ | Key addition |
|---------|-----------|---|--------------|
| v1 baseline | 0.298 | — | CE only |
| v2 | 0.325 | +0.027 | Lovász-Softmax |
| v3 | 0.405 | +0.080 | Deep supervision (biggest jump) |
| v4 (teacher) | 0.424 | +0.019 | EMA |
| Student (KD) | 0.395 | -0.029 | 12× smaller, 32.8× faster |
