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

## TF portu hata ayıklama günlüğü (2026-07-06 gecesi, protokol formatında)

### TF-P1
1. **Kimlik:** TF-P1 · 2. **Dosya:** `afetsonar_tf/training/__init__.py:4`
3. **Belirti:** pytest toplama aşamasında `ModuleNotFoundError: afetsonar_tf.training.ema_tf`
4. **Kök neden:** `EmaShadows` `models/ema_tf.py`'de tanımlıyken `training/` altından import edilmesi
5. **Düzeltme:** import yolu `afetsonar_tf.models.ema_tf` olarak düzeltildi
6. **Test:** tüm tests_tf toplama + 22 test · 7. **Sonuç:** geçti · 8. **Yan etki:** yok · **Durum:** Çözüldü

### TF-P2
1. **Kimlik:** TF-P2 · 2. **Dosya:** `afetsonar_tf/data/augment_tf.py` `shared_color_jitter`
3. **Belirti:** tf.data map derlemesinde `ValueError: 'hue' must also be initialized in the else branch`
4. **Kök neden:** AutoGraph, `if tf.random.uniform([]) < p:` ifadesini `tf.cond`'a çevirir; yalnızca if-dalında tanımlanan tensörler (hue/sat/val) graf sözleşmesini bozar — klasik graph-tracing hatası
5. **Düzeltme:** dallanma tamamen kaldırıldı; olasılık, parametreleri kimlik dönüşümüne ölçekleyen maske ile uygulandı (`param * do_flag`) — XLA/TPU dostu, davranışsal olarak eşdeğer
6. **Test:** `test_tfrecord_pipeline` (3) + `test_train_smoke` (20 adım) · 7. **Sonuç:** 6/6 geçti · 8. **Yan etki:** atlanan durumda da hue/sat op'ları çalışır (CPU host'ta ihmal edilebilir maliyet) · **Durum:** Çözüldü

### TF-P3
1. **Kimlik:** TF-P3 · 2. **Dosya:** `afetsonar_tf/convert_weights.py` `load_backbone`
3. **Belirti:** parite testinde `RuntimeError: Backbone load not clean: {'missing_keys': ['decode_head.batch_norm.num_batches_tracked']}`
4. **Kök neden:** PT BatchNorm'un tamsayı adım sayacı (`num_batches_tracked`) TF'te var olmayan tek anahtar; sıkı eksiksizlik kontrolü bunu hata sayıyordu
5. **Düzeltme:** yalnızca `num_batches_tracked` son ekli anahtarlar açıkça muaf tutuldu; diğer her eksik/fazla anahtar hâlâ hata
6. **Test:** `test_parity_teacher` (3 test, 6 çıktı tensörü, atol 1e-3) · 7. **Sonuç:** 3/3 geçti · 8. **Yan etki:** yok (sayaç çıkarımda kullanılmıyor) · **Durum:** Çözüldü

### TF-P4
1. **Kimlik:** TF-P4 · 2. **Dosya:** ortam — Colab `drive.mount` (kod hatası değil)
3. **Belirti:** Notebook 11 hücre 1'de üç kez `ValueError: mount failed` (~2 dk zaman aşımı)
4. **Kök neden:** Drive OAuth onay penceresi ayrı Chrome penceresinde açılıyor; otomasyon oturumunun erişimi dışında kaldı ve onay 2 dk içinde tamamlanamadı. Colab'ın mount akışı onaysız zaman aşımına düşüyor
5. **Düzeltme:** Kullanıcı OAuth onayını bir kez elle verdi ("Devam Et"); izin hesapta kalıcı olduğu için sonraki mount'lar etkileşimsiz 8 sn'de tamamlandı
6. **Test:** hücre 1 yeniden koşuldu · 7. **Sonuç:** geçti (`AFETSONAR` klasör assert'i dahil) · 8. **Yan etki:** yok; TPU oturumunda (notebook 10) tekrar onay gerekmeyecek · **Durum:** Çözüldü

### TF-P5
1. **Kimlik:** TF-P5 · 2. **Dosya:** `scripts_tf/convert_to_tfrecords.py` `--split` argümanı ↔ `notebooks/11_tf_prep_cpu.ipynb` hücre 5
3. **Belirti:** `convert_to_tfrecords.py: error: argument --split: invalid choice: 'gate' (choose from train, val, test)` — gate shard'ları üretilmedi; `!` komutu hücreyi durdurmadığı için val dönüştürmesi devam etti
4. **Kök neden:** Notebook 11, gate alt kümesi için `--split gate` çağırıyor ama dönüştürücünün `choices` listesi yalnızca train/val/test içeriyordu — iki dosya arasında sözleşme uyuşmazlığı
5. **Düzeltme:** `choices` listesine `"gate"` eklendi (commit bu satırla birlikte). Shard adlandırması `{split}_dmg/-nodmg` kalıbından türediği için `gate_dmg-*` / `gate_nodmg-*` üretimi notebook 10'un `gate_*` glob'uyla birebir eşleşiyor — başka değişiklik gerekmedi
6. **Düzeltmenin doğruluğu:** eval yolunda split adı yalnızca dosya adı önekini belirler; parse şeması tüm split'lerde aynı (`parse_eval`)
7. **Test:** kalıcı çözüm olarak notebook 10'un shard-kopyalama hücresine öz-onarım eklendi — gate shard'ları yoksa TPU oturumu `tf_export/test200.csv`'den kendisi üretir (~2-3 dk) ve Drive'a geri kopyalar; `assert gate_files` ile sessiz geçiş engellendi · 8. **Sonuç:** TPU oturumunda doğrulanacak · 9. **Yan etki:** yok (gate shard'ları mevcutsa hücre davranışı değişmez) · **Durum:** Kısmen çözüldü (kod düzeltildi + öz-onarım; Colab doğrulaması TPU oturumunda)

### Parite kanıtı (Tur 3)
- `tests_tf`: **22/22 geçti** (TF 2.19.1, transformers 4.57.6, CPU fp32)
- Teacher paritesi: 6 çıktı tensörünün tamamı `max|tf−torch| ≤ 1e-3`
- Golden loss: CE/Dice/Focal ≤1e-4, Lovász ≤1e-3 toleransta eşleşti
- Torch tarafı: branch'te `pytest tests/` → **199/199** (tests_tf, torch ortamında otomatik atlanır)

## Porta özgü doğrulanan gerçekler

- `teacher_v4_best_ema.pth` yapısı: `{epoch: 74, model_state_dict: 708 anahtar, val_miou_no_bg: 0.4703, history}` — **EMA ayrı sözlük değil; state dict zaten EMA'lı ağırlıklar** (BN running stats dahil, 9 adet). Dönüşüm tek aşamalı.
- State dict önekleri: `encoder.*`, `decode_head.*`, `fusion_convs.N.*`, `aux_heads.N.*`, `change_head.*`, `disaster_head.*`.
- `evaluation/metrics.py` torch importu korumalı (numpy yolu var) → TF tarafında **değiştirilmeden** kullanılıyor.
- `data/copy_paste.py` ve `routing/`, `geo/` tamamen framework-bağımsız → TFRecord dönüştürücü ve TF eğitimi bunları aynen kullanır.
- xBD maskelerinde `ignore_index` (-100) kullanılmıyor (0-5 arası) → TF Lovász'ta ignore filtreleme statik olarak atlanabildi (dönüştürücü bunu assert eder).
