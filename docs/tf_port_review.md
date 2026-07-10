# TF Port Kod İnceleme Raporu (2026-07-05)

TensorFlow/TPU portu öncesinde PyTorch kod tabanının satır satır incelemesi.
17 bulgu; önem sırasına göre. "Durum" kolonu: ✅ düzeltildi (commit'iyle),
🔀 tf-port branch'inde çözülüyor, 📋 backlog (kasıtlı ertelendi), ❎ doğrulandı-sorun-değil.

| # | Önem | Dosya | Bulgu | Durum |
|---|------|-------|-------|-------|
| 1 | **KRİTİK** | `afetsonar/data/dataset.py` `_load_image/_load_mask` | Eksik/bozuk dosya sessizce siyah görüntüye dönüşüyordu — eğitim verisini hiçbir uyarı olmadan zehirler. | ✅ master `6e9ba2e`: `FileNotFoundError` fırlatır; 4 yeni test |
| 2 | Orta | `losses/combo.py` + `training/trainer.py` | Deep-supervision list/tensor ayrımı model dışında (`isinstance` ile) çözülüyor — kırılgan sözleşme. | 📋 backlog (TF portunda `damage_logits` her zaman liste — sözleşme orada netleştirildi) |
| 3 | Orta | `losses/distillation.py` | KD loss'ta teacher(512)/student(256) kanal sayıları kurucu varsayılanı; model şekillerinden türetilmiyor. | 📋 student TF portu (T4) ile birlikte |
| 4 | Orta | `data/copy_paste.py:190-194` | `randint(0, H-rh)` kenar durumu şüphesi. | ❎ doğrulandı: `rh==H` iken `randint(0,0)=0` geçerli; `rh>H` zaten korunuyor. Hata yok |
| 5 | Düşük | `training/trainer.py` | Boş val loader'da metrikler NaN olarak geçmişe yazılabiliyordu. | ✅ kökü #12 ile çözüldü (metrikler artık sonlu) |
| 6 | Düşük | `data/augmentations.py` | Varsayılan `image_size=512`; teacher 768 ister — çağıran açıkça geçmezse sessiz kalite kaybı. | ✅ pratikte kapatıldı: pipeline/evaluate artık modele göre otomatik 768 (commit `d683dc2` öncesi); trainer'a config ile geçiliyor. Varsayılanı değiştirmek master'da davranış kırardı — 📋 not düşüldü |
| 7 | Düşük | `losses/combo.py` | `class_weights` uzunluğu `num_classes` ile doğrulanmıyordu → ilk batch'te kriptik crash. | ✅ master `6e9ba2e`: kurucuda `ValueError` + test |
| 8 | Düşük | `models/ema.py` | `restore()` backup anahtarlarının modelde var olduğunu doğrulamıyor. | 📋 backlog (tek çağıran trainer, anahtar seti sabit) |
| 9 | Bilgi | `training/trainer.py` | Son val batch'i küçükse uyarı yok. | 📋 önemsiz — metrikler doğru çalışıyor |
| 10 | Bilgi | `training/trainer.py` | history JSON'da `default=str` tip bilgisini bozabilir. | 📋 backlog |
| 11 | Orta | `training/trainer.py:_load_model` | `strict=False` atlanan anahtarları loglamıyordu → yanlış checkpoint sessizce "eğitilir"di. | ✅ master `6e9ba2e`: eşleşmeyen anahtar sayıları uyarı olarak basılıyor (pipeline'da zaten vardı) |
| 12 | Orta | `evaluation/metrics.py` | Hiç görülmemiş sınıflar NaN üretip ortalamalara/JSON'a sızabiliyordu. | ✅ master `6e9ba2e`: özet metrikler mevcut sınıflar üzerinden, boşsa 0.0; ayrıntı listelerinde NaN bilgi olarak kalır + testler |
| 13 | Düşük | çeşitli | `num_disaster_classes=5` çok yerde sabit; CSV'den türetilmiyor. | 📋 backlog |
| 14 | Düşük | `data/copy_paste.py` | `blend_alpha<0.5`'te kenar yumuşatmasız keskin dikişler. | 📋 backlog (varsayılan 1.0 kullanılıyor) |
| 15 | Bilgi | `models/teacher.py` | Deep-supervision gradyanları bellek maliyetli; decoder'da gradient checkpointing yok. | 📋 TPU tarafında bfloat16 zaten kazandırıyor; gerekirse T4'te |
| 16 | Düşük | `training/trainer.py` | `epochs<2`'de warmup tüm eğitimi kaplıyor. | 🔀 tf-port: yeni `WarmupCosine` programında doğru formül; torch tarafına dokunulmadı (Tier-2 TPU'da koşacak) |
| 17 | Bilgi | `training/trainer.py` | SWA `update_bn` tam bir epoch ek maliyet. | ❎ TF portunda SWA v1 kapsamı dışı (EMA yeterli); belgelendi |

## Porta özgü doğrulanan gerçekler

- `teacher_v4_best_ema.pth` yapısı: `{epoch: 74, model_state_dict: 708 anahtar, val_miou_no_bg: 0.4703, history}` — **EMA ayrı sözlük değil; state dict zaten EMA'lı ağırlıklar** (BN running stats dahil, 9 adet). Dönüşüm tek aşamalı.
- State dict önekleri: `encoder.*`, `decode_head.*`, `fusion_convs.N.*`, `aux_heads.N.*`, `change_head.*`, `disaster_head.*`.
- `evaluation/metrics.py` torch importu korumalı (numpy yolu var) → TF tarafında **değiştirilmeden** kullanılıyor.
- `data/copy_paste.py` ve `routing/`, `geo/` tamamen framework-bağımsız → TFRecord dönüştürücü ve TF eğitimi bunları aynen kullanır.
- xBD maskelerinde `ignore_index` (-100) kullanılmıyor (0-5 arası) → TF Lovász'ta ignore filtreleme statik olarak atlanabildi (dönüştürücü bunu assert eder).
