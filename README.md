# sapi-ai

Web app to check cattle health. Random forest classifier over sensor and manual readings, Flask front end.

Built for smallholder farms. A farmer enters what they measured, gets a likely condition and what to do about it.

## Classes

```
sehat                 normal
kurang_nafsu_makan    low appetite
demam                 fever
kembung               bloat
mastitis              udder infection
```

## Features

```
suhu_tubuh        37.0 - 42.0   C
detak_jantung     40 - 120      bpm
nafas_per_menit   12 - 60       /min
aktivitas         0 - 100       index
rumput_kg         0 - 60        daily feed
air_liter         20 - 140      daily water
bobot_kg          150 - 800
umur_bulan        6 - 120
```

## Setup

```
pip install -r requirements.txt
python ml/latih_model.py
python app.py
```

Open `http://127.0.0.1:5000`. `latih_model.py` must run first — it writes `model/sapi.joblib`.

## Train

```
python ml/latih_model.py
```

4000 synthetic rows, 80/20 split, `RandomForestClassifier` with 300 trees, max depth 14, balanced class weights.

Result: **88.6% accuracy** on the held-out set.

```
                 precision  recall  f1    support
sehat                1.000  0.988  0.994    161
kurang_nafsu_makan   0.851  0.840  0.845    156
demam                0.969  0.969  0.969    159
kembung              0.776  0.728  0.752    162
mastitis             0.835  0.907  0.870    162
```

Feature importance:

```
suhu_tubuh         0.2428
aktivitas          0.2267
rumput_kg          0.1590
detak_jantung      0.1482
nafas_per_menit    0.1048
air_liter          0.0823
bobot_kg           0.0196
umur_bulan         0.0166
```

`kembung` and `kurang_nafsu_makan` get confused with each other in the matrix (20 and 23 cases) — the synthetic distributions overlap there. Real data would separate them.

## API

```
GET  /              form
POST /api/cek       { suhu_tubuh, detak_jantung, ... } -> prediction
GET  /api/info      model metadata
```

Every field is required. Values are range-checked against the table above and rejected with 400 if out of bounds or non-numeric.

```bash
curl -X POST http://127.0.0.1:5000/api/cek -H "Content-Type: application/json" -d "{\"suhu_tubuh\":40.3,\"detak_jantung\":90,\"nafas_per_menit\":39,\"aktivitas\":34,\"rumput_kg\":19,\"air_liter\":54,\"bobot_kg\":410,\"umur_bulan\":50}"
```

Returns `demam` at 98%.

## CLI

```
python ml/pakai_model.py --suhu 38.5 --detak 65 --nafas 25 --aktivitas 70 \
    --rumput 35 --air 80 --bobot 420 --umur 48
```

## Caveats

The training data is generated from hand-written distributions per class, not collected from real farms. Accuracy numbers reflect how separable those distributions are, not real-world performance. A production version needs actual veterinary records.

The advice text is generic. It is not a substitute for a vet.
