"""Web buat cek kesehatan sapi.

Jalankan: python app.py
Buka http://127.0.0.1:5000
"""

import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, "model", "sapi.joblib")
INFO = os.path.join(BASE, "model", "info.json")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

if not os.path.exists(MODEL):
    raise SystemExit("model belum ada, jalanin: python ml/latih_model.py")

model = joblib.load(MODEL)
with open(INFO) as f:
    info = json.load(f)

FITUR = info["fitur"]
LABEL = info["label"]

BATAS = {
    "suhu_tubuh": (37.0, 42.0),
    "detak_jantung": (40, 120),
    "nafas_per_menit": (12, 60),
    "aktivitas": (0, 100),
    "rumput_kg": (0, 60),
    "air_liter": (20, 140),
    "bobot_kg": (150, 800),
    "umur_bulan": (6, 120),
}

SARAN = {
    "sehat": "gak ada kelainan, lanjutin aja pola yang sekarang",
    "kurang_nafsu_makan": "naikin pakan bertahap, cek rumputnya masih bagus apa gak",
    "demam": "pisahin dari kandang lain, ukur suhu tiap 4 jam, panggil dokter hewan kalo 2 hari gak turun",
    "kembung": "hentikan pakan fermentasi dulu, kasih jalan pagi, jangan kasih air banyak langsung",
    "mastitis": "cek ambingnya, susu yang keluar buang dulu, bersihin kandang, panggil dokter hewan",
}


@app.route("/")
def beranda():
    return render_template("index.html", fitur=FITUR, batas=BATAS, akurasi=info["akurasi"])


@app.route("/api/cek", methods=["POST"])
def cek():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "body harus json"}), 400

    baris = []
    for nama in FITUR:
        if nama not in data:
            return jsonify({"error": f"{nama} belum diisi"}), 400

        try:
            nilai = float(data[nama])
        except (TypeError, ValueError):
            return jsonify({"error": f"{nama} harus angka"}), 400

        if not np.isfinite(nilai):
            return jsonify({"error": f"{nama} bukan angka yang bener"}), 400

        bawah, atas = BATAS[nama]
        if nilai < bawah or nilai > atas:
            return jsonify({"error": f"{nama} harus antara {bawah} dan {atas}"}), 400

        baris.append(nilai)

    x = np.array([baris])
    proba = model.predict_proba(x)[0]
    kelas = int(np.argmax(proba))
    hasil = LABEL[kelas]

    peringkat = sorted(
        [
            {"label": LABEL[i], "yakin": round(float(proba[i]), 4)}
            for i in range(len(LABEL))
        ],
        key=lambda r: -r["yakin"],
    )

    return jsonify(
        {
            "hasil": hasil,
            "yakin": round(float(proba[kelas]), 4),
            "saran": SARAN[hasil],
            "semua": peringkat,
        }
    )


@app.route("/api/info")
def api_info():
    return jsonify(info)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
