import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Konfigurasi Path Model Multi-Output Anda
MODEL_PATH = "model_multitask_diagnosa.pkl"

def load_medical_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as file:
        return pickle.load(file)

# Memuat model saat server startup
pipeline_model = load_medical_model()


# Route Utama: Tampilan Awal Sistem (Keadaan Standby)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Route Analisis: Memproses Form POST dan Mengembalikan Dua Output Sekaligus
@app.route('/analyze', methods=['POST'])
def analyze():
    if not pipeline_model:
        return render_template('assistant.html', error="Berkas 'model_multitask_diagnosa.pkl' tidak ditemukan di direktori server backend.")

    # 1. Mengekstrak seluruh parameter masukan dari elemen Form HTML
    symptom = request.form.get('symptom', '')
    usia = request.form.get('usia', 25)
    tekanan_darah = request.form.get('tekanan_darah', '120/80')
    kadar_gula = request.form.get('kadar_gula', 100)

    # Simpan kembali data input agar tidak hilang saat halaman merender ulang hasil
    data_inputs = {
        "symptom": symptom,
        "usia": usia,
        "tekanan_darah": tekanan_darah,
        "kadar_gula": kadar_gula
    }

    try:
        # 2. Reshape ke bentuk DataFrame 2D (Sesuai skema preprocessor ColumnTransformer Anda)
        input_df = pd.DataFrame([{
            "symptom_description": str(symptom),
            "usia": int(usia),
            "tekanan_darah": str(tekanan_darah), # Tipe kategori wajib string
            "kadar_gula": int(kadar_gula)
        }])

        # 3. Prediksi menggunakan MultiOutputClassifier Pipeline Anda
        # Mengembalikan matriks 2D berisi baris data hasil: [['diagnosa', 'terapi']]
        prediction_matrix = pipeline_model.predict(input_df)[0]
        
        # Ekstrak kedua target output secara independen
        data_outputs = {
            "diagnosis": str(prediction_matrix[0]).upper().replace('_', ' '),
            "therapy": str(prediction_matrix[1])
        }

        # 4. Kirim kembali data masukan dan hasil keluaran ganda ke Jinja2 Template
        return render_template('index.html', inputs=data_inputs, outputs=data_outputs)

    except Exception as e:
        return render_template('index.html', inputs=data_inputs, error=str(e))


if __name__ == '__main__':
    # Jalankan server lokal di port 5000 dengan mode debug aktif
    app.run(debug=True, port=8000)