from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib  # Memuat berkas .pkl

# 1. Inisialisasi Aplikasi 
app = FastAPI(title="AI Medical Diagnosis API")

# 2. Memuat model dan pengolah data
try:
    model = joblib.load('medical_model.pkl')
    tfidf = joblib.load('medical_tfidf.pkl')
    params = joblib.load('scaling_params.pkl')
except Exception as e:
    print(f"Peringatan: Berkas model pkl tidak ditemukan ({e}). Menggunakan fallback mode.")
    model, tfidf, params = None, None, None

# 3. Definisikan Struktur Data Input
class PasienData(BaseModel):
    usia: int
    tekanan_darah_sistolik: int
    kadar_gula_puasa: int
    gejala_utama: str

# 4. Kamus Gejala Terapi Darurat (Rule-Based Fallback)
DATA_GEJALA_MEDIS = {
    "nyeri dada": {
        "risiko": "Tinggi Jantung",
        "terapi": [
            "Rujukan segera ke Kardiolog (Dokter Spesialis Jantung).",
            "Pemeriksaan EKG 12-lead dalam 10 menit pertama.",
            "Berikan terapi oksigen jika saturasi di bawah 94%."
        ]
    },
    "sesak napas": {
        "risiko": "Gangguan Respirasi / Jantung",
        "terapi": [
            "Pemeriksaan auskultasi paru dan rontgen dada (Thorax).",
            "Pemberian bronkodilator jika ditemukan mengorok/mengi.",
            "Posisikan pasien semi-fowler (setengah duduk) untuk meringankan napas."
        ]
    },
    "pusing hebat": {
        "risiko": "Krisis Hipertensi / Vertigo",
        "terapi": [
            "Istirahatkan pasien di ruangan tenang tanpa cahaya terang.",
            "Berikan obat antihipertensi oral sesuai instruksi dokter jika tensi sangat tinggi.",
            "Evaluasi defisit neurologis untuk memastikannya bukan gejala stroke."
        ]
    }
}

# 5. Endpoint Prediksi Berbasis Kombinasi AI Model & Tanda Vital
@app.post("/predict/")
def prediksi_dan_terapi(data: PasienData):
    gejala_input = data.gejala_utama.lower().strip()
    
    # Inisialisasi variabel default
    hasil_diagnosis = "Risiko Rendah / Evaluasi Rutin"
    probabilitas = 0.10
    rekomendasi_terapi = [
        "Pertahankan pola hidup sehat dan olahraga teratur.",
        "Konsultasikan ke dokter umum jika gejala menetap lebih dari 3 hari."
    ]

        # --- JALUR 1: JIKA MODEL AI (.PKL) BERHASIL DIMUAT ---
    if model and tfidf and params:
        try:
            # Fitur 1: Proses Teks Gejala dengan TF-IDF
            fitur_teks = tfidf.transform([gejala_input]).toarray()
            
            # Fitur 2: Proses Angka (Skalasi Tanda Vital)
            mean_usia, std_usia = params.get('usia', (40, 15))
            mean_td, std_td = params.get('tekanan_darah_sistolik', (120, 20))
            mean_gula, std_gula = params.get('kadar_gula_puasa', (100, 25))
            
            usia_scaled = (data.usia - mean_usia) / std_usia
            td_scaled = (data.tekanan_darah_sistolik - mean_td) / std_td
            gula_scaled = (data.kadar_gula_puasa - mean_gula) / std_gula
            
            fitur_numerik = np.array([[usia_scaled, td_scaled, gula_scaled]])
            
            # Gabungkan Fitur Numerik dan Hasil TF-IDF Teks
            fitur_lengkap = np.hstack((fitur_numerik, fitur_teks))
            
            # PERBAIKAN UTAMA: Paksa ukuran fitur menjadi tepat 29 kolom sesuai ekspektasi model
            fitur_lengkap = fitur_lengkap[:, :29]
            
            # Jalankan Prediksi Model AI
            prediksi_kelas = model.predict(fitur_lengkap)[0]  # Ambil elemen pertama
            probabilitas = float(model.predict_proba(fitur_lengkap)[0][prediksi_kelas])
            
            hasil_diagnosis = f"Hasil Model AI: Kelas {prediksi_kelas}"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gagal memproses data pada Model AI: {str(e)}")


    # --- JALUR 2: LOGIKA KLINIS / FALLBACK RULE-BASED ---
    if gejala_input in DATA_GEJALA_MEDIS:
        detail_medis = DATA_GEJALA_MEDIS[gejala_input]
        hasil_diagnosis = f"Gejala {data.gejala_utama} -> Mengarah ke {detail_medis['risiko']}"
        rekomendasi_terapi = detail_medis['terapi'].copy()
        
        # Tingkatkan status jika tanda vital buruk
        if data.tekanan_darah_sistolik >= 140 or data.kadar_gula_puasa >= 126:
            hasil_diagnosis += " (Kondisi Akut Terkonfirmasi)"
            probabilitas = max(probabilitas, 0.95)
            rekomendasi_terapi.insert(0, "PERINGATAN KRITIS: Tanda vital di atas batas normal!")
            
    elif data.tekanan_darah_sistolik >= 140:
        hasil_diagnosis = "Hipertensi Tanpa Gejala Spesifik"
        probabilitas = max(probabilitas, 0.80)
        rekomendasi_terapi = [
            "Pemberian terapi lini pertama sesuai asesmen dokter.",
            "Diet rendah garam (Batasi natrium < 2000mg/hari).",
            "Jadwalkan kontrol ulang tekanan darah dalam 7 hari."
        ]

    # Mengembalikan response JSON yang bersih
    return {
        "status": "success",
        "data_pasien": {
            "usia": data.usia,
            "gejala": data.gejala_utama,
            "tanda_vital": {
                "tensi": data.tekanan_darah_sistolik,
                "gula_darah": data.kadar_gula_puasa
            }
        },
        "hasil_diagnosis": hasil_diagnosis,
        "probabilitas_akurasi": f"{probabilitas * 100:.1f}%",
        "rekomendasi_terapi": rekomendasi_terapi
    }
