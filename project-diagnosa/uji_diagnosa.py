import pickle
import pandas as pd

# 1. Muat kembali model baru
with open('model_diagnosa_multimodal.pkl', 'rb') as file:
    model = pickle.load(file)

# 2. Siapkan data pasien baru (Format harus berupa DataFrame dengan kolom yang sesuai)
data_pasien_baru = pd.DataFrame({
    'symptom_description': [
        "Sering buang air kecil, lemas, dan berat badan turun drastis belakangan ini.",
        "Nyeri dada kiri menjalar ke tangan disertai keringat dingin hebat."
    ],
    'usia': [52, 58],
    'tekanan_darah': ["135/85", "165/100"],
    'kadar_gula': [260, 135] # Pasien pertama memiliki kadar gula sangat tinggi (indikasi diabetes)
})

# 3. Lakukan prediksi
hasil_prediksi = model.predict(data_pasien_baru)

# 4. Tampilkan Hasil
for i, prediksi in enumerate(hasil_prediksi):
    print(f"Pasien {i+1}:")
    print(f"  Gejala   : {data_pasien_baru['symptom_description'].iloc[i]}")
    print(f"  Diagnosa : {prediksi}\n")