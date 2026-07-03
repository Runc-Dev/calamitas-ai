# AFETSONAR 🛰️

**Drone-based Disaster Damage Assessment & Routing**  


---

## Overview

AFETSONAR (Turkish: *afet* = disaster, *sonar* = detection) is an end-to-end AI pipeline that turns satellite/drone imagery into an actionable rescue coordination map within **31 ms** — fast enough for real-time edge deployment.

```
Pre-disaster image ─┐
                    ├── Siamese SegFormer ──► 6-class damage mask
Post-disaster image ─┘         │
                                ▼
                   Building extraction (OpenCV contours)
                                │
                                ▼
                   FEMA priority + survival curve
                                │
                                ▼
               Priority-weighted K-means → 5 rescue teams
                                │
                                ▼
               A* routing (gradient-weighted OSM road graph)
                                │
                                ▼
               TSP ordering (Rosenkrantz 1977 nearest-neighbour)
                                │
                                ▼
          8-layer Folium interactive HTML map  (108 KB)
```

---

## Results

### Model Performance (xBD test set, 1375 images)

| Model | mIoU (no bg) | mF1 | Destroyed IoU | Params | Latency |
|-------|:-----------:|:---:|:-------------:|:------:|:-------:|
| Teacher (SegFormer-B3) | **0.424** | **0.640** | **0.570** | 50.3M | ~1140 ms |
| **Student (SegFormer-B0)** | **0.395** | **0.617** | **0.524** | **4.3M** | **36 ms ✅** |

> **Student = 12× smaller, 32.8× faster, 93% knowledge retention.**  
> The only edge-deployable solution on the xView2 leaderboard.

### Per-class IoU (Student)

| bg | no_damage | minor | major | destroyed | unclassified |
|:--:|:---------:|:-----:|:-----:|:---------:|:------------:|
| 0.988 | 0.650 | 0.272 | 0.396 | **0.524** | 0.136 |

### SoTA Comparison (xView2 dataset)

| Method | F1 | Params | Edge? |
|--------|:--:|:------:|:-----:|
| Durnov 2020 (1st place) | 0.74 | 100M+ | ❌ |
| Roy et al. 2021 | 0.68 | ~80M | ❌ |
| **AFETSONAR Teacher** | **0.640** | 50.3M | ❌ |
| **AFETSONAR Student** | **0.617** | **4.3M** | **✅** |

### Ablation Study

| Variant | mIoU_no_bg | Δ | Key addition |
|---------|:-----------:|:--:|--------------|
| v1 — baseline (CE only) | 0.298 | — | — |
| v2 — +Lovász-Softmax | 0.325 | +0.027 | Direct mIoU optimisation |
| v3 — +Deep Supervision | 0.405 | **+0.080** | 3 auxiliary heads ← biggest jump |
| v4 — +EMA (Teacher) | 0.424 | +0.019 | Exponential moving average |
| Student (KD) | 0.395 | — | 5-component distillation |

---

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/AFETSONAR.git
cd AFETSONAR
pip install -r requirements.txt
pip install -e .
```

### Single-image inference

```bash
python scripts/inference.py \
  --post  path/to/post_disaster.png \
  --pre   path/to/pre_disaster.png \
  --model checkpoints/student/student_v1_best_ema.pth \
  --output results/prediction.png \
  --bbox  41.003,28.975,41.008,28.981 \
  --map   results/map.html
```

### Full pipeline → interactive map

```bash
python scripts/run_pipeline.py \
  --post   path/to/post_disaster.png \
  --pre    path/to/pre_disaster.png \
  --model  checkpoints/student/student_v1_best_ema.pth \
  --bbox   41.003,28.975,41.008,28.981 \
  --output results/afetsonar_map.html
```

### Python API

```python
from afetsonar import AfetsonarPipeline

pipeline = AfetsonarPipeline("checkpoints/student/student_v1_best_ema.pth")

# Damage mask
mask = pipeline.predict("post.png", "pre.png")   # np.ndarray (H, W), values 0–5

# Full map
html_path = pipeline.generate_map(
    post_path="post.png",
    pre_path="pre.png",
    bbox_latlon=(41.003, 28.975, 41.008, 28.981),
    hospitals=[{"name": "Cerrahpaşa", "lat": 41.0048, "lon": 28.9510}],
    output_path="results/map.html",
)
```

### Run demo (no GPU, no data required)

```bash
python demo/sample_inference.py
```

---

## Damage Classes

| Index | Class | Weight |
|:-----:|-------|:------:|
| 0 | background | 0.05 |
| 1 | no_damage | 1.0 |
| 2 | minor_damage | **8.0** |
| 3 | major_damage | 5.0 |
| 4 | destroyed | **7.0** |
| 5 | unclassified | 0.5 |

---

## Training

### Full 3-phase pipeline

```bash
# Phase 1 — Building localization (encoder pre-training)
python scripts/train_localizer.py \
  --data-dir data/xbd --output-dir checkpoints/teacher

# Phase 2 — Teacher training (Siamese SegFormer-B3)
python scripts/train_teacher.py \
  --config configs/teacher.yaml --data-dir data/xbd

# Phase 3 — Student distillation (KD, SegFormer-B0)
python scripts/train_student.py \
  --teacher-ckpt checkpoints/teacher/teacher_v4_best_ema.pth \
  --config configs/student.yaml

# Evaluation
python scripts/evaluate.py \
  --model checkpoints/student/student_v1_best_ema.pth \
  --test-csv data/xbd/splits/test.csv

# Evaluation with Tier-1 TTA (8 transforms, optional multi-scale)
python scripts/evaluate.py \
  --model checkpoints/teacher/teacher_v4_best_ema.pth \
  --test-csv data/xbd/splits/test.csv \
  --tta --tta-scales 0.75 1.0 1.25
```

See `notebooks/09_tier1_tta_swa_eval.ipynb` for the ready-to-run Colab
workflow (baseline vs TTA vs TTA+multi-scale vs SWA).

### ONNX export (edge deployment)

```bash
pip install onnx onnxruntime   # or: pip install -e ".[onnx]"

python scripts/export_onnx.py \
  --checkpoint checkpoints/student/student_v1_best_ema.pth
# → student_v1_best_ema.onnx (16.5 MB), onnxruntime parity-checked
```

### Docker

```bash
docker build -t afetsonar -f docker/Dockerfile .
docker-compose -f docker/docker-compose.yml up afetsonar-pipeline
```

---

## Repository Structure

```
AFETSONAR/
├── afetsonar/               # Installable Python package
│   ├── models/              # Localizer, Teacher, Student, EMA
│   ├── losses/              # Lovász, Combo, KD, Localization
│   ├── data/                # XBDDatasetV2, augmentations, Copy-Paste
│   ├── routing/             # Priority, K-means, A*, TSP, LZ
│   ├── geo/                 # Geo utils, GeoTIFF, FoliumMapBuilder, AutoPreFetcher
│   ├── evaluation/          # Metrics, ablation tables, TTA
│   ├── training/            # AfetsonarTrainer (incremental fine-tuning)
│   ├── config.py            # All hyper-parameters
│   ├── deployment.py        # ONNX export / parity verification
│   └── pipeline.py          # AfetsonarPipeline (end-to-end)
├── scripts/                 # CLI training / inference / export scripts
├── api/                     # FastAPI REST backend (POST /analyze)
├── mobile/                  # React Native (Expo) client
├── app.py                   # Gradio web UI (HuggingFace Spaces)
├── notebooks/               # Training journey + 09 Tier-1 Colab notebook
├── configs/                 # YAML hyper-parameter files
├── tests/                   # pytest suite (174 tests, torch-optional)
├── docs/                    # Architecture, references, guides
├── demo/                    # Self-contained demo
├── docker/                  # Dockerfile + compose
└── results/                 # Evaluation outputs (CSV/JSON)
```

---

## Techniques Used

**Segmentation**
- SegFormer (Xie et al. 2021) — transformer backbone
- Siamese network — pre/post change detection
- Lovász-Softmax loss (Berman et al. 2018) — direct mIoU optimisation
- Deep supervision (3 auxiliary heads) — +0.080 mIoU jump
- EMA (Exponential Moving Average) — stable late-stage training
- OHEM (Online Hard Example Mining) — hard pixel focus
- Boundary-aware loss — building edge quality
- Building-aware crop — 80% patches contain building pixels
- WeightedRandomSampler — class imbalance correction
- Cosine warm restarts (Loshchilov & Hutter 2017) — plateau breaking

**Knowledge Distillation**
- 5-component KD loss: soft-label KL + CE + feature MSE + attention transfer + combo damage
- Temperature T=4 (Hinton et al. 2015)
- 12× parameter reduction, 32.8× speedup, 93% knowledge retention

**Routing**
- FEMA priority formula (P-154/P-1070) — survival probability × severity × area × population
- K-means clustering (priority-weighted) — rescue team zone assignment
- A* search (Hart et al. 1968) — optimal road routing
- Gradient edge weights (Shapely buffer) — impassable/slowed damage zones
- TSP nearest-neighbour (Rosenkrantz 1977) — multi-building visit order
- Voronoi diagrams — team responsibility zones
- NATO STANAG 3204 — helicopter LZ minimum dimensions (25×25 m)

---

## Dataset

**xBD** (Gupta et al. 2019) — 9,168 satellite images (Tier 1 + Tier 3), 6 damage classes.

```
data/xbd/
├── train/images/   pre + post disaster pairs
├── train/targets/  6-class segmentation masks
├── tier3/          additional training data
└── splits/         train.csv / val.csv / test.csv
```

Generate splits:
```python
from afetsonar.data import build_split_csv
build_split_csv("data/xbd", "data/xbd/splits")
```

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Interactive Map Layers

The output Folium map (`afetsonar_master_map.html`) contains 8 layers:

1. Satellite basemap (Esri World Imagery)
2. Damage assessment markers (colour by class, size ∝ priority)
3. Rescue team routing paths (colour by team)
4. Voronoi team zones
5. Hospital / assembly point markers
6. Helicopter landing zones (NATO STANAG 3204 compliant)
7. Alternative routes (k-shortest for destroyed/major buildings)
8. Street map toggle

---

## Scientific References

See [docs/scientific_references.md](docs/scientific_references.md) for the complete list.

Key papers: Xie 2021 (SegFormer) · Hinton 2015 (KD) · Berman 2018 (Lovász) · Hart 1968 (A*) · Gupta 2019 (xBD) · FEMA P-154 · NATO STANAG 3204

---

## Citation

```bibtex
@software{afetsonar2025,
  title   = {AFETSONAR: Drone-based Disaster Damage Assessment and Routing},
  year    = {2025},
  note    = {Teknofest 2025 — Built in 11 days},
  url     = {https://github.com/your-org/AFETSONAR}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
