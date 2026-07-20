import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Memuat model multitask yang telah dilatih sebelumnya
MODEL_PATH = "model_multitask_diagnosa.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        pipeline_model = pickle.load(file)
else:
    pipeline_model = None
    print(f"⚠️ Peringatan: File {MODEL_PATH} tidak ditemukan. Harap letakkan di folder yang sama.")

# Route 1: Halaman Awal (Form Kosong)
@app.route('/', methods=['GET'])
def index():
    # Merender file HTML di dalam folder templates
    return render_template('index.html')

# Route 2: Memproses Data Form POST dan Mengembalikan Hasil Ke Halaman yang Sama
@app.route('/diagnosa', methods=['POST'])
def diagnosa():
    if not pipeline_model:
        return render_template('index.html', error="Model .pkl belum dimuat di server backend!")

    # 1. Menangkap data yang dikirimkan oleh pengguna lewat elemen form HTML
    symptom = request.form.get('symptom', '')
    usia = request.form.get('usia', 25)
    tekanan_darah = request.form.get('tekanan_darah', '120/80')
    kadar_gula = request.form.get('kadar_gula', 100)

    try:
        # 2. Bungkus ke DataFrame (Format harus sesuai persis dengan preprocessor scikit-learn Anda)
        input_data = pd.DataFrame([{
            "symptom_description": str(symptom),
            "usia": int(usia),
            "tekanan_darah": str(tekanan_darah), # String format "120/80"
            "kadar_gula": int(kadar_gula)
        }])
        
        # 3. Prediksi menggunakan model MultiOutputClassifier
        hasil_prediksi = pipeline_model.predict(input_data)[0]
        
        data_output = {
            "task_1_diagnosis": str(hasil_prediksi[0]).upper(),
            "task_2_therapy": str(hasil_prediksi[1])
        }

        # 4. Kembalikan data ke template agar disisipkan langsung ke halaman HTML
        return render_template(
            'index.html', 
            hasil=data_output,
            input_symptom=symptom,
            input_usia=usia,
            input_td=tekanan_darah,
            input_gula=kadar_gula
        )

    except Exception as e:
        return render_template('index.html', error=str(e), input_symptom=symptom)

if __name__ == '__main__':
    # Jalankan server lokal Flask di port 5000
    app.run(debug=True, port=5000)