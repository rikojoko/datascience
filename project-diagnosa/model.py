import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

print("=== MEMULAI PROSES PEMBUATAN MODEL MEDIS (.PKL) ===")

# 1. MEMBUAT DATA LATIHAN TIRUAN (DUMMY DATA - SEBANYAK 10 DATA PASIEN)
data_medis = {
    'usia': [55, 28, 60, 35, 50, 42, 65, 38, 45, 52],
    'tekanan_darah_sistolik': [150, 115, 160, 120, 140, 130, 170, 118, 125, 145],
    'kadar_gula_puasa': [135, 90, 140, 95, 120, 110, 150, 88, 105, 130],
    'gejala_utama': [
        'nyeri dada menjalar ke lengan', 
        'batuk ringan dan flu', 
        'sesak napas berat saat beraktivitas', 
        'pusing biasa karena lelah', 
        'pusing hebat berputar vertigo', 
        'demam dan batuk berdahak',
        'nyeri dada terasa ditekan benda berat',
        'sesak napas tiba tiba',
        'mual dan pusing ringan',
        'badan lemas dan sering haus'
    ],
    # Target: 1 = Risiko Tinggi (Butuh Tindakan Cepat), 0 = Risiko Rendah (Evaluasi Rutin)
    'target_diagnosis': [1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data_medis)

# 2. PROSES FITUR TEKS (TF-IDF)
# Batasi max_features menjadi 26 agar total kolom pas 29 (3 numerik + 26 teks = 29)
tfidf = TfidfVectorizer(max_features=26)
X_text = tfidf.fit_transform(df['gejala_utama']).toarray()

# 3. PROSES FITUR NUMERIK & HITUNG SCALING PARAMS
# Menyimpan nilai rata-rata (mean) dan standar deviasi (std) untuk scaling manual di FastAPI
params = {
    'usia': (float(df['usia'].mean()), float(df['usia'].std())),
    'tekanan_darah_sistolik': (float(df['tekanan_darah_sistolik'].mean()), float(df['tekanan_darah_sistolik'].std())),
    'kadar_gula_puasa': (float(df['kadar_gula_puasa'].mean()), float(df['kadar_gula_puasa'].std()))
}

# Lakukan penskalaan (scaling) data numerik
usia_scaled = (df['usia'] - params['usia'][0]) / params['usia'][1]
td_scaled = (df['tekanan_darah_sistolik'] - params['tekanan_darah_sistolik'][0]) / params['tekanan_darah_sistolik'][1]
gula_scaled = (df['kadar_gula_puasa'] - params['kadar_gula_puasa'][0]) / params['kadar_gula_puasa'][1]

X_numeric = np.column_stack((usia_scaled, td_scaled, gula_scaled))

# 4. GABUNGKAN SEMUA FITUR (TOTAL KINI PAS 29 KOLOM)
X_combined = np.hstack((X_numeric, X_text))
y = df['target_diagnosis'].values

print(f"-> Ukuran data latihan: {X_combined.shape[0]} baris, {X_combined.shape[1]} kolom.")

# 5. LATIH MODEL RANDOM FOREST CLASSIFIER
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_combined, y)

# 6. SIMPAN SEMUA FILENYA MENJADI .PKL
joblib.dump(model, 'medical_model.pkl')
joblib.dump(tfidf, 'medical_tfidf.pkl')
joblib.dump(params, 'scaling_params.pkl')

print("\n[SUKSES]: Berhasil membuat dan menyimpan 3 berkas pkl baru:")
print("- medical_model.pkl")
print("- medical_tfidf.pkl")
print("- scaling_params.pkl")
print("=====================================================")
