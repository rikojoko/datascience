import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Multitask Medical NLP & Clinical API",
    description="API Multitask yang diselaraskan dengan arsitektur preprocessor asli.",
    version="1.0.4"
)

# 1. Sesuaikan Tipe Data Input dengan Skema Dataset Training Anda
class KeluhanInput(BaseModel):
    symptom: str
    usia: int               # Numerik -> Masuk ke StandardScaler
    tekanan_darah: str      # String (Format "120/80") -> Masuk ke OneHotEncoder
    kadar_gula: int         # Numerik -> Masuk ke StandardScaler

MODEL_PATH = "model_multitask_diagnosa.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"File '{MODEL_PATH}' tidak ditemukan!")

with open(MODEL_PATH, 'rb') as file:
    pipeline_model = pickle.load(file)

@app.get("/")
def index():
    return {"status": "Online", "message": "API Sinkron dengan Preprocessor Model."}

@app.post("/predict-multitask")
def predict_multitask(payload: KeluhanInput):
    if not payload.symptom.strip():
        raise HTTPException(status_code=400, detail="Teks keluhan tidak boleh kosong.")
    
    try:
        # 2. Bangun DataFrame dengan Nama Kolom dan Tipe Data yang Sesuai Persis
        input_data = pd.DataFrame([{
            "symptom_description": str(payload.symptom),
            "usia": int(payload.usia),
            "tekanan_darah": str(payload.tekanan_darah), # Wajib String format "sistolik/diastolik"
            "kadar_gula": int(payload.kadar_gula)
        }])
        
        # 3. Prediksi menggunakan MultiOutputClassifier Pipeline
        # Output dari model ini berbentuk matriks 2D: [[ 'diagnosa', 'terapi' ]]
        hasil_prediksi = pipeline_model.predict(input_data)[0]
        
        diagnosa = str(hasil_prediksi[0]).upper()
        terapi = str(hasil_prediksi[1])

        return {
            "status": "Success",
            "multitask_output": {
                "task_1_diagnosis": diagnosa,
                "task_2_therapy": terapi
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses Pipeline model: {str(e)}")