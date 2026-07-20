from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import traceback

app = FastAPI(title="AI Medical Diagnosis API")

# Memuat model dan pengolah data
try:
    model = joblib.load('medical_model.pkl')
    tfidf = joblib.load('medical_tfidf.pkl')
    params = joblib.load('scaling_params.pkl')
    print("--- INFO PKL ---")
    print("Tipe data params:", type(params))
    if isinstance(params, dict):
        print("Keys yang tersedia di scaling_params.pkl:", list(params.keys()))
except FileNotFoundError as e:
    print(f"Error loading files: {e}. Pastikan semua file .pkl ada di direktori yang sama.")

# Schema Validasi Input dengan Pydantic
class DiagnosisInput(BaseModel):
    keluhan: str = Field(
        ..., 
        json_schema_extra={"example": "Pusing dan demam sejak kemarin malam"}
    )
    suhu: float = Field(
        ..., 
        description="Suhu tubuh dalam Celsius",
        json_schema_extra={"example": 38.5}
    )
    tensi: str = Field(
        ..., 
        description="Tekanan darah format 'sistole/diastole' atau angka langsung",
        json_schema_extra={"example": "120/80"}
    )

@app.get("/")
def read_root():
    return {"message": "Welcome to AI Medical Diagnosis API! Go to /docs to try the API."}

@app.post("/diagnose")
async def get_diagnosis(input_data: DiagnosisInput):
    try:
        keluhan = input_data.keluhan
        suhu = input_data.suhu
        tensi = input_data.tensi
        
        # 1. Preprocessing Teks
        text_vector = tfidf.transform([keluhan.lower()]).toarray()
        
        # 2. Ambil nilai min & max untuk normalisasi suhu
        # Jika params berbentuk dict, ambil key 'min'/'max'. Jika object scikit-learn (scaler), sesuaikan.
        if isinstance(params, dict):
            p_min = params.get('min', 0)
            p_max = params.get('max', 1)
        else:
            p_min = getattr(params, 'data_min_', [0])[0]
            p_max = getattr(params, 'data_max_', [1])[0]

        suhu_norm = (suhu - p_min) / (p_max - p_min) if (p_max - p_min) != 0 else 0
        
        # 3. Handle Tensi secara Fleksibel
        # Jika tensi dikirim berupa '120/80', kita ambil sistolenya saja (120) 
        # agar jumlah fitur numerik tetap 2 (suhu & tensi) seperti kode awalmu.
        try:
            if '/' in str(tensi):
                tensi_val = float(tensi.split('/')[0])
            else:
                tensi_val = float(tensi)
        except ValueError:
            tensi_val = 120.0 # fallback jika format salah
            
        # 4. Susun Fitur Numerik (Kembali ke 2 Fitur sesuai kode awalmu)
        numeric_features = np.array([[suhu_norm, tensi_val]])
        
        # 5. Gabungkan Fitur
        features_combined = np.hstack((text_vector, numeric_features))
        
        # 6. Prediksi
        prediction = model.predict(features_combined)
        
        hasil_diagnosis = prediction[0]
        if hasattr(hasil_diagnosis, "item"): 
            hasil_diagnosis = hasil_diagnosis.item()

        return {
            "status": "success",
            "diagnosis": str(hasil_diagnosis),
            "rekomendasi_awal": "Silahkan konsultasi lebih lanjut dengan dokter spesialis terkait."
        }
        
    except Exception as e:
        # PENTING: Mencetak detail error baris mana yang rusak di terminal
        print("\n=== TERJADI ERROR DI SERVER ===")
        traceback.print_exc()
        print("===============================\n")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal: {str(e)}")