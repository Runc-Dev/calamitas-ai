# AFETSONAR 🛰️

**İnsansız Hava Aracı Tabanlı Afet Hasar Değerlendirme ve Rota Planlama**  
Teknofest 2025 · 11 günde sıfırdan inşa edildi

---

## Genel Bakış

AFETSONAR, uydu/drone görüntülerini **31 ms** içinde eyleme dönüştürülebilir bir kurtarma koordinasyon haritasına çeviren uçtan uca bir yapay zeka pipeline'ıdır. Bu hız, gerçek zamanlı uç cihaz (edge) dağıtımına uygundur.

```
Afet öncesi görüntü ─┐
                      ├── Siamese SegFormer ──► 6 sınıflı hasar maskesi
Afet sonrası görüntü ─┘         │
                                 ▼
                    Bina tespiti (OpenCV kontur)
                                 │
                                 ▼
                    FEMA öncelik skoru + hayatta kalma eğrisi
                                 │
                                 ▼
                Ağırlıklı K-means → 5 kurtarma ekibi
                                 │
                                 ▼
                A* rota (gradyan ağırlıklı OSM yol grafı)
                                 │
                                 ▼
                TSP sıralama (Rosenkrantz 1977 en yakın komşu)
                                 │
                                 ▼
               8 katmanlı Folium interaktif HTML harita (108 KB)
```

---

## Sonuçlar

### Model Performansı (xBD test seti, 1375 görüntü)

| Model | mIoU (bg hariç) | mF1 | Yıkılmış IoU | Parametre | Gecikme |
|-------|:---------------:|:---:|:------------:|:---------:|:-------:|
| Teacher (SegFormer-B3) | **0.424** | **0.640** | **0.570** | 50.3M | ~1140 ms |
| **Student (SegFormer-B0)** | **0.395** | **0.617** | **0.524** | **4.3M** | **36 ms ✅** |

> **Student modeli: 12× daha küçük, 32.8× daha hızlı, %93 bilgi korunumu.**
> xView2 sıralamasındaki tek uç cihaza dağıtılabilir çözüm.

### Sınıf Bazında IoU (Student)

| bg | hasarsız | küçük | büyük | yıkılmış | sınıflandırılmamış |
|:--:|:--------:|:-----:|:-----:|:--------:|:------------------:|
| 0.988 | 0.650 | 0.272 | 0.396 | **0.524** | 0.136 |

### SoTA Karşılaştırması (xView2 veri seti)

| Yöntem | F1 | Parametre | Uç Cihaz? |
|--------|:--:|:---------:|:---------:|
| Durnov 2020 (1. sıra) | 0.74 | 100M+ | ❌ |
| Roy et al. 2021 | 0.68 | ~80M | ❌ |
| **AFETSONAR Teacher** | **0.640** | 50.3M | ❌ |
| **AFETSONAR Student** | **0.617** | **4.3M** | **✅** |

### Ablasyon Çalışması

| Varyant | mIoU_no_bg | Δ | Eklenen teknik |
|---------|:-----------:|:--:|----------------|
| v1 — temel (sadece CE) | 0.298 | — | — |
| v2 — +Lovász-Softmax | 0.325 | +0.027 | Doğrudan mIoU optimizasyonu |
| v3 — +Derin Denetim | 0.405 | **+0.080** | 3 yardımcı kafa ← en büyük artış |
| v4 — +EMA (Teacher) | 0.424 | +0.019 | Üstel hareketli ortalama |
| Student (KD) | 0.395 | — | 5 bileşenli damıtma |

---

## Hızlı Başlangıç

### Kurulum

```bash
git clone https://github.com/your-org/AFETSONAR.git
cd AFETSONAR
pip install -r requirements.txt
pip install -e .
```

### Tek görüntü çıkarımı

```bash
python scripts/inference.py \
  --post  afet_sonrasi.png \
  --pre   afet_oncesi.png \
  --model checkpoints/student/student_v1_best_ema.pth \
  --output results/tahmin.png \
  --bbox  41.003,28.975,41.008,28.981 \
  --map   results/harita.html
```

### Tam pipeline → interaktif harita

```bash
python scripts/run_pipeline.py \
  --post   afet_sonrasi.png \
  --pre    afet_oncesi.png \
  --model  checkpoints/student/student_v1_best_ema.pth \
  --bbox   41.003,28.975,41.008,28.981 \
  --output results/afetsonar_harita.html
```

### Python API

```python
from afetsonar import AfetsonarPipeline

pipeline = AfetsonarPipeline("checkpoints/student/student_v1_best_ema.pth")

# Hasar maskesi
mask = pipeline.predict("afet_sonrasi.png", "afet_oncesi.png")

# Tam harita
html_path = pipeline.generate_map(
    post_path="afet_sonrasi.png",
    pre_path="afet_oncesi.png",
    bbox_latlon=(41.003, 28.975, 41.008, 28.981),
    hospitals=[{"name": "Cerrahpaşa", "lat": 41.0048, "lon": 28.9510}],
    output_path="results/harita.html",
)
```

---

## Hasar Sınıfları

| İndeks | Sınıf | Eğitim Ağırlığı |
|:------:|-------|:---------------:|
| 0 | arka plan | 0.05 |
| 1 | hasarsız | 1.0 |
| 2 | küçük hasar | **8.0** |
| 3 | büyük hasar | 5.0 |
| 4 | yıkılmış | **7.0** |
| 5 | sınıflandırılmamış | 0.5 |

---

## Eğitim

### 3 aşamalı tam pipeline

```bash
# Aşama 1 — Bina lokalizasyonu (encoder ön eğitimi)
python scripts/train_localizer.py --data-dir data/xbd

# Aşama 2 — Teacher eğitimi (Siamese SegFormer-B3)
python scripts/train_teacher.py --config configs/teacher.yaml

# Aşama 3 — Student damıtma (KD, SegFormer-B0)
python scripts/train_student.py \
  --teacher-ckpt checkpoints/teacher/teacher_v4_best_ema.pth

# Değerlendirme
python scripts/evaluate.py \
  --model checkpoints/student/student_v1_best_ema.pth \
  --test-csv data/xbd/splits/test.csv
```

---

## Repo Yapısı

```
AFETSONAR/
├── afetsonar/               # Kurulabilir Python paketi
│   ├── models/              # Localizer, Teacher, Student, EMA
│   ├── losses/              # Lovász, Combo, KD, Lokalizasyon
│   ├── data/                # XBDDatasetV2, augmentasyonlar
│   ├── routing/             # Öncelik, K-means, A*, TSP, LZ
│   ├── geo/                 # Coğrafi araçlar, GeoTIFF, FoliumMapBuilder
│   ├── evaluation/          # Metrikler, ablasyon tabloları
│   ├── config.py            # Tüm hiperparametreler
│   └── pipeline.py          # AfetsonarPipeline (uçtan uca)
├── scripts/                 # CLI eğitim / çıkarım betikleri
├── notebooks/               # Orijinal 8 eğitim not defteri
├── configs/                 # YAML hiperparametre dosyaları
├── tests/                   # pytest test paketi
├── docs/                    # Mimari, referanslar, rehberler
├── demo/                    # Bağımsız demo
└── docker/                  # Dockerfile + compose
```

---

## Kullanılan Teknikler

**Segmentasyon**
- SegFormer (Xie et al. 2021) — transformer omurgası
- Siamese ağ — afet öncesi/sonrası değişim tespiti
- Lovász-Softmax kaybı (Berman et al. 2018) — doğrudan mIoU optimizasyonu
- Derin denetim (3 yardımcı kafa) — +0.080 mIoU artışı
- EMA (Üstel Hareketli Ortalama) — kararlı geç-aşama eğitimi
- OHEM (Çevrimiçi Zor Örnek Madenciliği)
- Sınır farkında kayıp — bina kenar kalitesi
- Bina farkında kırpma — yüksek hasar bölgelerine odaklanma
- WeightedRandomSampler — sınıf dengesizliğini giderme
- Kosinüs sıcak yeniden başlatmalar — plato kırma

**Bilgi Damıtma**
- 5 bileşenli KD kaybı: yumuşak etiket KL + CE + özellik MSE + dikkat transferi + combo hasar
- Sıcaklık T=4 (Hinton et al. 2015)
- 12× parametre azaltma, 32.8× hız artışı, %93 bilgi korunumu

**Rota Planlama**
- FEMA öncelik formülü (P-154/P-1070) — hayatta kalma × şiddet × alan × nüfus
- Öncelik ağırlıklı K-means — kurtarma ekibi bölge ataması
- A* arama (Hart et al. 1968) — optimal yol rotası
- Gradyan kenar ağırlıkları (Shapely tampon) — geçilemez/yavaş hasar bölgeleri
- TSP en yakın komşu (Rosenkrantz 1977) — çoklu bina ziyaret sırası
- Voronoi diyagramları — ekip sorumluluk bölgeleri
- NATO STANAG 3204 — helikopter iniş bölgesi minimum boyutları (25×25 m)

---

## Veri Seti

**xBD** (Gupta et al. 2019) — 9.168 uydu görüntüsü (Tier 1 + Tier 3), 6 hasar sınıfı.

---

## Testler

```bash
pip install pytest
pytest tests/ -v
```

---

## İnteraktif Harita Katmanları

Çıktı Folium haritası (`afetsonar_master_map.html`) 8 katman içerir:

1. Uydu altlık harita (Esri World Imagery)
2. Hasar değerlendirme işaretçileri (renge göre sınıf, boyuta göre öncelik)
3. Kurtarma ekibi rota yolları (ekibe göre renk)
4. Voronoi ekip bölgeleri
5. Hastane / toplanma noktası işaretçileri
6. Helikopter iniş bölgeleri (NATO STANAG 3204 uyumlu)
7. Alternatif rotalar (yıkılmış/büyük hasarlı binalar için k-en kısa yollar)
8. Sokak haritası geçişi

---

## Bilimsel Referanslar

Eksiksiz liste için [docs/scientific_references.md](docs/scientific_references.md) dosyasına bakın.

---

## Lisans

Apache 2.0 — [LICENSE](LICENSE) dosyasına bakın.
