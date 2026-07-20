from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="AI Medical Diagnosis API")

# Memuat model dan pengolah data
model = joblib.load('medical_model.pkl')
tfidf = joblib.load('medical_tfidf.pkl')
params = joblib.load('scaling_params.pkl')

@app.post("/diagnose")
async def get_diagnosis(data: dict):
    # 1. Mengambil input dari user
    keluhan = data['keluhan']
    suhu = data['suhu']
    tensi = data['tensi']
    
    # 2. Preprocessing & Vectorization Teks
    text_vector = tfidf.transform([keluhan.lower()]).toarray()
    
    # 3. Normalisasi Data Numerik
    suhu_norm = (suhu - params['min']) / (params['max'] - params['min'])
    numeric_features = np.array([[suhu_norm, tensi]])
    
    # 4. Menggabungkan Fitur (Teks + Numerik)
    features_combined = np.hstack((text_vector, numeric_features))
    
    # 5. Prediksi Diagnosis
    prediction = model.predict(features_combined)
    
    return {
        "status": "success",
        "diagnosis": prediction[0],
        "rekomendasi_awal": "Silahkan konsultasi lebih lanjut dengan dokter spesialis terkait."
    }