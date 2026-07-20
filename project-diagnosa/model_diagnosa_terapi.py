import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

# 1. Dataset Baru dengan Target Ganda: 'diagnosis' & 'terapi'
data_klinis = {
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
    
    # Target 1
    'diagnosis': [
        'influenza', 'alergi', 'depresi', 'migrain', 'alergi', 'depresi',
        'flu_biasa', 'gangguan_kecemasan', 'diare', 'bronkitis', 'alergi',
        'gangguan_kecemasan', 'demam_berdarah', 'diabetes', 'migrain',
        'depresi', 'flu_biasa', 'influenza', 'eksim', 'maag', 'stres',
        'anemia', 'serangan_jantung'
    ],
    # Target 2 (Diprediksi langsung oleh AI)
    'terapi': [
        'Istirahat total, minum air hangat, dan konsumsi Paracetamol.',
        'Hindari debu/pemicu dan konsumsi antihistamin.',
        'Konseling psikologis dan perbaiki pola tidur rutin.',
        'Istirahat di kamar gelap dan minum pereda nyeri kepala.',
        'Hindari seafood pemicu dan kompres dingin area bengkak.',
        'Rujukan ke psikolog/psikiater untuk terapi CBT.',
        'Perbanyak minum cairan hangat dan istirahat cukup.',
        'Latih teknik deep breathing dan kurangi kafein.',
        'Pemberian oralit cairan elektrolit dan diet makanan lunak.',
        'Gunakan humidifier udara dan hindari asap rokok.',
        'Konsumsi antihistamin segera dan hindari udang.',
        'Tenangkan pernapasan dan rujukan evaluasi psikologis.',
        'Bed rest total, pantau trombosit, dan minum cairan isotonik.',
        'Diet rendah karbohidrat/gula dan kontrol rutin ke dokter.',
        'Kompres hangat dahi dan kurangi paparan cahaya gadget.',
        'Dukungan sosial keluarga dan konsultasi medis ke psikiater.',
        'Minum vitamin C dan jaga hidrasi tubuh tetap baik.',
        'Isolasi mandiri, bed rest, dan minum air minimal 2 liter.',
        'Gunakan sabun non-parfum dan aplikasikan pelembap kulit.',
        'Makan porsi kecil tapi sering, hindari asam dan pedas.',
        'Lakukan relaksasi stres dan manajemen waktu kerja.',
        'Konsumsi makanan tinggi zat besi seperti daging merah/bayam.',
        '🚨 EMERGENCY: Segera bawa ke UGD dan berikan oksigen tambahan!'
    ]
}

df = pd.DataFrame(data_klinis)

# 2. Preprocessing Data Input
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(ngram_range=(1, 2)), 'symptom_description'),
        ('num', StandardScaler(), ['usia', 'kadar_gula']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['tekanan_darah'])
    ]
)

# 3. Menggunakan MultiOutputClassifier agar model bisa memprediksi > 1 kolom target
model_multi_target = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(LogisticRegression(C=1.0, random_state=42, max_iter=1000)))
])

# 4. Pemisahan Fitur (X) dan Dua Target (y)
X = df[['symptom_description', 'usia', 'tekanan_darah', 'kadar_gula']]
y = df[['diagnosis', 'terapi']] # Dua kolom target sekaligus!

# 5. Training Model
model_multi_target.fit(X, y)

# 6. Simpan Model Multi-output ke File .pkl
nama_file = 'model_multitask_diagnosa.pkl'
with open(nama_file, 'wb') as file:
    pickle.dump(model_multi_target, file)

print(f"🔥 Sukses! Model multi-output disimpan sebagai: {nama_file}")