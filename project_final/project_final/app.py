from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Muat model dan pengolah data
try:
    model = joblib.load('project_final/medical_model.pkl')
    tfidf = joblib.load('project_final/medical_tfidf.pkl')
    params = joblib.load('project_final/scaling_params.pkl')
except Exception as e:
    print(f"Peringatan: Gagal memuat file model pkl. Pastikan file pkl ada di folder yang sama. Detail: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diagnose_web', methods=['POST'])
def diagnose_web():
    try:
        # Ambil data dari form HTML dengan aman
        keluhan = request.form.get('keluhan', '')
        suhu_raw = request.form.get('suhu', '')
        tensi_raw = request.form.get('tensi', '')

        # Validasi input kosong
        if not keluhan or not suhu_raw or not tensi_raw:
            return render_template('index.html', error="Harap isi semua kolom keluhan, suhu, dan tensi!")

        # Konversi tipe data
        suhu = float(suhu_raw)
        tensi = float(tensi_raw)

        # Preprocessing & Prediksi
        text_vector = tfidf.transform([keluhan.lower()]).toarray()
        
        # Normalisasi suhu dengan pengaman pembagian dengan nol
        rentang_suhu = params['max'] - params['min']
        suhu_norm = (suhu - params['min']) / rentang_suhu if rentang_suhu != 0 else 0
        
        numeric_features = np.array([[suhu_norm, tensi]])
        features = np.hstack((text_vector, numeric_features))

        # Kamus terapi / rekomendasi
        terapi_map = {
            'Flu/Infeksi Saluran Napas': 'Istirahat total, minum air hangat, dan konsumsi vitamin C.',
            'Jantung': 'Segera periksa ke dokter spesialis jantung, hindari aktivitas berat.',
            'Hipotensi': 'Konsumsi makanan mengandung garam dan minum air putih yang cukup.',
            'Apendisitis': 'Segera ke UGD untuk pemeriksaan bedah, jangan menunda.',
            'Alergi Kulit': 'Hindari pemicu alergi, gunakan krim antihistamin, dan konsultasikan ke dokter kulit.'
        }

        # Melakukan prediksi
        prediction = model.predict(features)[0]
        rekomendasis = terapi_map.get(prediction, 'Konsultasikan gejala Anda ke dokter umum.')

        # PERBAIKAN UTAMA: Mengembalikan respon valid ke halaman index.html dengan membawa variabel hasil
        return render_template(
            'index.html', 
            prediction=prediction, 
            rekomendasis=rekomendasis,
            keluhan=keluhan,
            suhu=suhu,
            tensi=tensi
        )

    except ValueError:
        # Menangani jika pengguna memasukkan huruf/karakter aneh pada kolom suhu & tensi
        return render_template('index.html', error="Input Suhu dan Tensi harus berupa angka yang valid!")
    except Exception as e:
        # Menangani error tak terduga lainnya
        return render_template('index.html', error=f"Terjadi kesalahan sistem: {str(e)}")

if __name__ == '__main__':
    # Memindahkan port ke 5001 untuk menghindari konflik port 5000 yang terkunci
    print("Membuka Server Flask di http://127.0.0.1:5001")
    app.run(port=5001, debug=True)