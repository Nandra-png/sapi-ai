"""Model klasifikasi kesehatan sapi.

Random forest dari scikit-learn. Fitur diambil dari data sensor (suhu, detak
jantung, aktivitas) plus beberapa pengukuran manual.
"""

import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

FITUR = [
    "suhu_tubuh",
    "detak_jantung",
    "nafas_per_menit",
    "aktivitas",
    "rumput_kg",
    "air_liter",
    "bobot_kg",
    "umur_bulan",
]

LABEL = ["sehat", "kurang_nafsu_makan", "demam", "kembung", "mastitis"]

ARAH = "model"


def bikin_data(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, len(FITUR)))
    y = np.zeros(n, dtype=int)

    for i in range(n):
        kelas = rng.integers(0, len(LABEL))
        y[i] = kelas

        if kelas == 0:
            suhu = rng.normal(38.5, 0.25)
            detak = rng.normal(65, 5)
            nafas = rng.normal(25, 3)
            aktivitas = rng.normal(70, 8)
            rumput = rng.normal(35, 4)
            air = rng.normal(80, 8)
        elif kelas == 1:
            suhu = rng.normal(38.6, 0.3)
            detak = rng.normal(67, 5)
            nafas = rng.normal(26, 3)
            aktivitas = rng.normal(42, 9)
            rumput = rng.normal(18, 5)
            air = rng.normal(58, 9)
        elif kelas == 2:
            suhu = rng.normal(40.2, 0.4)
            detak = rng.normal(88, 7)
            nafas = rng.normal(38, 5)
            aktivitas = rng.normal(35, 10)
            rumput = rng.normal(20, 6)
            air = rng.normal(55, 10)
        elif kelas == 3:
            suhu = rng.normal(38.9, 0.3)
            detak = rng.normal(74, 6)
            nafas = rng.normal(30, 4)
            aktivitas = rng.normal(40, 8)
            rumput = rng.normal(22, 6)
            air = rng.normal(60, 9)
        else:
            suhu = rng.normal(39.4, 0.35)
            detak = rng.normal(80, 6)
            nafas = rng.normal(32, 4)
            aktivitas = rng.normal(32, 9)
            rumput = rng.normal(25, 6)
            air = rng.normal(62, 9)

        bobot = rng.normal(420, 60)
        umur = rng.normal(48, 18)

        if kelas == 1:
            bobot -= rng.normal(25, 8)
        if kelas == 2:
            rumput *= 0.7
        if kelas == 4:
            aktivitas *= 0.8

        X[i] = [
            max(37.0, min(42.0, suhu)),
            max(40, min(120, detak)),
            max(12, min(60, nafas)),
            max(0, min(100, aktivitas)),
            max(0, min(60, rumput)),
            max(20, min(140, air)),
            max(150, min(800, bobot)),
            max(6, min(120, umur)),
        ]

    return X, y


def latih():
    X, y = bikin_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=7,
        n_jobs=-1,
    )
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    akurasi = accuracy_score(yte, pred)

    print(f"akurasi: {akurasi:.4f}\n")
    print(classification_report(yte, pred, target_names=LABEL, digits=3))
    print("confusion matrix:")
    print(confusion_matrix(yte, pred))

    urut = np.argsort(model.feature_importances_)[::-1]
    print("\nfitur paling ngaruh:")
    for i in urut:
        print(f"  {FITUR[i]:20} {model.feature_importances_[i]:.4f}")

    os.makedirs(ARAH, exist_ok=True)

    import joblib

    joblib.dump(model, os.path.join(ARAH, "sapi.joblib"))
    with open(os.path.join(ARAH, "info.json"), "w") as f:
        json.dump(
            {
                "fitur": FITUR,
                "label": LABEL,
                "akurasi": round(float(akurasi), 4),
                "n_latih": int(len(Xtr)),
                "n_uji": int(len(Xte)),
            },
            f,
            indent=2,
        )

    print(f"\nmodel disimpen di {ARAH}/sapi.joblib")


if __name__ == "__main__":
    latih()
