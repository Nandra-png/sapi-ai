"""Cek kesehatan satu sapi dari terminal.

python ml/pakai_model.py --suhu 38.5 --detak 65 --nafas 25 --aktivitas 70 \
    --rumput 35 --air 80 --bobot 420 --umur 48
"""

import argparse
import json
import os

import joblib
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, "..", "model", "sapi.joblib")
INFO = os.path.join(BASE, "..", "model", "info.json")

with open(INFO) as f:
    info = json.load(f)

FITUR = info["fitur"]
LABEL = info["label"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suhu", type=float, required=True)
    p.add_argument("--detak", type=float, required=True)
    p.add_argument("--nafas", type=float, required=True)
    p.add_argument("--aktivitas", type=float, required=True)
    p.add_argument("--rumput", type=float, required=True)
    p.add_argument("--air", type=float, required=True)
    p.add_argument("--bobot", type=float, required=True)
    p.add_argument("--umur", type=float, required=True)
    a = p.parse_args()

    x = np.array([[
        a.suhu, a.detak, a.nafas, a.aktivitas,
        a.rumput, a.air, a.bobot, a.umur,
    ]])

    model = joblib.load(MODEL)
    proba = model.predict_proba(x)[0]
    kelas = int(np.argmax(proba))

    print(f"\nhasil: {LABEL[kelas]} ({proba[kelas] * 100:.1f}%)\n")
    print("semua kemungkinan:")
    for i in np.argsort(proba)[::-1]:
        print(f"  {LABEL[i]:22} {proba[i] * 100:5.1f}%")


if __name__ == "__main__":
    main()
