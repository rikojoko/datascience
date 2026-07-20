import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Dataset Baru dengan Tambahan Fitur Medis
data_diagnosa = {
    'symptom_description': [
        "Pasien mengeluh demam tinggi, batuk kering, dan nyeri tenggorokan selama tiga hari.",
        "Mengalami hidung tersumbat, bersin-bersin, mata gatal setelah terpapar debu.",
        "Merasa lelah terus-menerus, kehilangan minat pada hobi, sulit tidur selama berminggu-minggu.",
        "Sakit kepala hebat, mual, muntah, dan sensitif terhadap cahaya.",
        "Kulit gatal kemerahan, bengkak di beberapa area setelah makan seafood.",
        "Perasaan sedih yang mendalam, putus asa, dan perubahan nafsu makan signifikan.",
        "Gejala mirip flu tapi lebih ringan, dengan pilek dan sedikit demam.",
        "Sering cemas tanpa alasan jelas, detak jantung cepat, tangan berkeringat.",
        "Sakit perut hebat, diare, dan muntah berkali-kali.", 
        "Batuk berdahak, sesak napas, dan demam rendah.", 
        "Gatal-gatal di seluruh tubuh, ruam merah, dan bengkak bibir setelah makan udang.", 
        "Perasaan panik mendadak, jantung berdebar, dan kesulitan bernapas tanpa sebab jelas.", 
        "Demam dan nyeri otot serta sendi, disertai ruam merah.", 
        "Sering buang air kecil, mudah haus, dan cepat lapar, berat badan turun tanpa sebab.", 
        "Mata kabur, sakit kepala, leher kaku, dan sensitif terhadap cahaya.", 
        "Tidak bisa tidur, nafsu makan hilang, merasa tidak berharga.", 
        "Sakit tenggorokan, batuk kering, hidung tersumbat, dan nyeri kepala ringan.", 
        "Demam tinggi, menggigil, nyeri badan, dan batuk yang parah.", 
        "Kulit kering bersisik, gatal parah, terutama di malam hari.", 
        "Nyeri ulu hati, mual, perut kembung, dan sering bersendawa.", 
        "Kesulitan berkonsentrasi, sering lupa, mudah tersinggung.", 
        "Lemas, pucat, pusing, dan detak jantung tidak teratur.", 
        "Nyeri dada, sesak napas, berkeringat dingin, menjalar ke lengan kiri." 
    ],
    'usia': [25, 19, 34, 28, 22, 40, 30, 26, 35, 50, 18, 29, 21, 55, 45, 38, 27, 42, 24, 33, 31, 48, 60],
    'tekanan_darah': ["120/80", "110/70", "120/80", "130/85", "115/75", "120/80", "118/78", "130/90", "110/70", "135/85", "120/80", "140/90", "110/70", "140/90", "150/95", "115/75", "120/80", "130/85", "120/80", "125/80", "135/85", "100/65", "160/100"],
    'kadar_gula': [90, 85, 95, 100, 90, 110, 95, 105, 88, 115, 92, 120, 85, 250, 130, 90, 95, 100, 88, 105, 110, 75, 140],
    'diagnosis': [
        'influenza', 'alergi', 'depresi', 'migrain', 'alergi', 'depresi',
        'flu_biasa', 'gangguan_kecemasan', 'diare', 'bronkitis', 'alergi',
        'gangguan_kecemasan', 'demam_berdarah', 'diabetes', 'migrain',
        'depresi', 'flu_biasa', 'influenza', 'eksim', 'maag', 'stres',
        'anemia', 'serangan_jantung'
    ]
}

df = pd.DataFrame(data_diagnosa)

# 2. Mengatur Pemrosesan Spesifik per Tipe Kolom (ColumnTransformer)
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(ngram_range=(1, 2)), 'symptom_description'), # Teks ke matriks angka
        ('num', StandardScaler(), ['usia', 'kadar_gula']),                    # Normalisasi angka skala besar
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['tekanan_darah'])    # Mengubah teks kategori (str) ke kolom biner
    ]
)

# 3. Menggabungkan Preprocessor dan Classifier ke dalam Satu Pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(C=1.0, random_state=42, max_iter=1000))
])

# 4. Melatih Model dengan Seluruh Fitur
X = df[['symptom_description', 'usia', 'tekanan_darah', 'kadar_gula']]
y = df['diagnosis']
model_pipeline.fit(X, y)

# 5. Menyimpan Model ke file .pkl
nama_file = 'model_diagnosa_multimodal.pkl'
with open(nama_file, 'wb') as file:
    pickle.dump(model_pipeline, file)

print(f"Model berhasil diperbarui dan disimpan sebagai {nama_file}")