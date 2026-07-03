# AFETSONAR REST API — Web Entegrasyon Kılavuzu

Web sitesinin AFETSONAR'a bağlanması için özellik-bazlı REST API.
Her özellik **ayrı bir endpoint**: sadece ihtiyacın olanı çağır.

## Sunucuyu başlatma

```bash
pip install -r api/requirements.txt
pip install -e .          # afetsonar paketi (repo kökünde)

# Windows (PowerShell):
$env:AFETSONAR_CHECKPOINT = "checkpoints/student/student_v1_best_ema.pth"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Linux/macOS:
AFETSONAR_CHECKPOINT=checkpoints/student/student_v1_best_ema.pth \
    uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- İnteraktif dokümantasyon (Swagger): `http://<sunucu>:8000/docs`
- CORS tüm origin'lere açık — tarayıcıdan doğrudan `fetch` edilebilir.
- İlk istekte model yüklenir (birkaç saniye); sonrakiler hızlıdır.

## Endpoint özeti

| Endpoint | Metod | Ne yapar |
|----------|-------|----------|
| `/health` | GET | Sunucu ayakta mı, model yüklü mü |
| `/model-info` | GET | Model bilgisi (tip, parametre, sınıflar) |
| `/exif-gps` | POST | **Gerçek konum**: görüntünün EXIF'inden GPS çıkarır |
| `/predict` | POST | Sadece hasar maskesi (base64 PNG + istatistik) |
| `/buildings` | POST | Binalar: sınır çokgenleri + öncelik skorları |
| `/map` | POST | Hazır interaktif harita (Folium HTML) |
| `/routes` | POST | Ekip ataması + A* rotaları (JSON gövde) |
| `/analyze` | POST | Hepsi tek çağrıda (maske + binalar + GeoJSON) |

Tüm POST endpoint'leri (routes hariç) `multipart/form-data` alır:
dosya alanları `post_image` (zorunlu) ve `pre_image` (opsiyonel).

## ÖNEMLİ — sitede binaların yanlış çizilmesinin 2 sebebi ve çözümü

1. **Sınır verisi artık var.** Eskiden API sadece bina merkez noktası
   döndürüyordu; sınır çizecek veri yoktu. Artık her bina
   `polygon_latlon` (köşe listesi) taşıyor ve `/buildings?format=geojson`
   standart GeoJSON döndürüyor.
2. **Koordinat sırasına dikkat.** GeoJSON `[boylam, enlem]` (lon, lat)
   sırası kullanır; `polygon_latlon` ise `[enlem, boylam]` (lat, lon)
   sırasıdır. Leaflet'te `L.geoJSON` GeoJSON'u otomatik doğru işler;
   `L.polygon` ise `[lat, lon]` ister. İkisini karıştırma!

### Leaflet ile bina sınırlarını çizme (önerilen yol)

```html
<div id="map" style="height:600px"></div>
<script>
const map = L.map("map").setView([41.005, 28.977], 16);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png").addTo(map);

const colors = {no_damage:"#4caf50", minor_damage:"#ffeb3b",
                major_damage:"#ff9800", destroyed:"#f44336",
                unclassified:"#9c27b0"};

async function drawBuildings(file, lat, lon) {
  const fd = new FormData();
  fd.append("post_image", file);
  fd.append("lat", lat);
  fd.append("lon", lon);
  fd.append("format", "geojson");           // <-- GeoJSON iste

  const resp = await fetch("http://SUNUCU:8000/buildings", {method:"POST", body:fd});
  const geojson = await resp.json();

  L.geoJSON(geojson, {
    style: f => ({color: colors[f.properties.damage_class_name] || "#9e9e9e",
                  weight: 2, fillOpacity: 0.35}),
    onEachFeature: (f, layer) => layer.bindTooltip(
      `#${f.properties.building_id} — ${f.properties.damage_class_name}` +
      ` — ${Math.round(f.properties.area_m2)} m² — öncelik ${f.properties.priority_score}`)
  }).addTo(map);
}
</script>
```

## Örnek çağrılar

### 1) Gerçek konum çekme (EXIF GPS)

```js
const fd = new FormData();
fd.append("image", fileInput.files[0]);   // drone'un ORİJİNAL JPEG'i
const r = await (await fetch("http://SUNUCU:8000/exif-gps", {method:"POST", body:fd})).json();
// r = {success:true, found:true, lat:41.005, lon:28.977, altitude_m:120}
```

> ⚠️ EXIF yalnızca **orijinal dosyada** bulunur. Görüntüyü canvas'a çizip
> yeniden export eden, PNG'ye çeviren veya sıkıştıran her adım GPS
> verisini siler. Dosyayı kullanıcıdan aldığın haliyle gönder.

### 2) Sadece hasar maskesi

```js
const fd = new FormData();
fd.append("post_image", postFile);
// fd.append("pre_image", preFile);       // varsa doğruluk artar
// fd.append("use_tta", "true");          // ~8x yavaş, daha isabetli
const r = await (await fetch("http://SUNUCU:8000/predict", {method:"POST", body:fd})).json();
imgEl.src = "data:image/png;base64," + r.mask_png_b64;
// r.stats = sınıf bazında piksel yüzdeleri
```

### 3) Hazır harita (kendi harita kodu yazmak istemeyenler için)

```js
const fd = new FormData();
fd.append("post_image", postFile);
fd.append("lat", "41.005"); fd.append("lon", "28.977"); // veya boş bırak → EXIF
fd.append("hospitals_json", JSON.stringify([{name:"Cerrahpaşa", lat:41.0048, lon:28.951}]));
fd.append("response_format", "json");
const r = await (await fetch("http://SUNUCU:8000/map", {method:"POST", body:fd})).json();
document.getElementById("mapFrame").srcdoc = r.html;   // <iframe id="mapFrame">
```

Katmanlar: bina sınırları, hasar işaretleri, ekip bölgeleri, hastaneler,
A* kurtarma rotaları (OSM), helikopter iniş bölgeleri (OSM).
`include_routes=false` / `include_lz=false` ile ağ bağımlı katmanlar kapatılabilir.

### 4) Rota hesaplama (site kendi haritasını çiziyorsa)

```js
const body = {
  buildings: buildingsFromStep2,           // /buildings çıktısındaki dizi
  bbox: [41.003, 28.975, 41.008, 28.981],  // [lat_min, lon_min, lat_max, lon_max]
  n_teams: 3,
  hospitals: [{name:"Cerrahpaşa", lat:41.0048, lon:28.951}]
};
const r = await (await fetch("http://SUNUCU:8000/routes", {
  method:"POST", headers:{"Content-Type":"application/json"},
  body: JSON.stringify(body)})).json();
// r.routes[i].coords = [[lat,lon], ...]  → L.polyline(coords) ile çiz
// r.teams[i].assigned_hospital, r.buildings[i].team_id
```

### 5) Tek çağrıda her şey

```js
const fd = new FormData();
fd.append("post_image", postFile);        // lat/lon yoksa EXIF denenır
const r = await (await fetch("http://SUNUCU:8000/analyze", {method:"POST", body:fd})).json();
// r.mask_png_b64, r.stats, r.buildings (polygon_latlon dahil),
// r.geojson (L.geoJSON'a hazır), r.bbox, r.coord_source ("form"|"exif"|null)
```

## Hata yönetimi

- `422` — eksik/yanlış parametre (ör. `/map` için koordinat yok). `detail` alanını oku.
- `503` — model yüklenemedi (checkpoint yolu yanlış). `AFETSONAR_CHECKPOINT` ayarla.
- `500` — beklenmeyen hata; sunucu logunda traceback bulunur.
- `/routes` OSM'ye erişemezse `routes: []` + `route_error` döner, istek yine `200`'dür.

## Sık düşülen hatalar (checklist)

- [ ] GeoJSON `[lon, lat]`, Leaflet `[lat, lon]` — karıştırma.
- [ ] EXIF için orijinal JPEG gönder; canvas/PNG dönüşümü GPS'i siler.
- [ ] `bbox` sırası her yerde `[lat_min, lon_min, lat_max, lon_max]`.
- [ ] Maske görüntüsü `mask_width×mask_height` (512×512) — kendi görüntünün
      boyutuna ölçekleyerek bindir.
- [ ] Büyük görüntülerde ilk istek yavaş olabilir (model + CPU çıkarımı);
      arayüzde yükleniyor göstergesi kullan.
