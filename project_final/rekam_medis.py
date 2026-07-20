import pandas as pd
import numpy as np

# Simulasi data rekam medis pasien
data_pasien = {
    'keluhan_utama': [
        'Demam tinggi sudah 3 hari disertai batuk kering',
        'Nyeri dada sebelah kiri menjalar ke lengan',
        'Pusing, mual, dan lemas setelah beraktivitas',
        'Nyeri perut hebat di bagian bawah kanan',
        'Gatal-gatal merah di seluruh area punggung',
        None # Data kosong untuk simulasi cleaning
    ],
    'suhu_tubuh': [38.5, 36.6, 37.0, 38.2, 36.5, 37.5],
    'tekanan_darah_sistolik': [120, 150, 90, 110, 120, np.nan],
    'diagnosis': ['Flu/Infeksi Saluran Napas', 'Gangguan Jantung', 'Hipotensi', 'Apendisitis', 'Alergi Kulit', 'Flu'],
    'saran_terapi': ['Istirahat & Parasetamol', 'Rujuk Spesialis Jantung', 'Minum cukup & Glukosa', 'Rujuk Bedah', 'Antihistamin', 'Istirahat']
}

df_pasien = pd.DataFrame(data_pasien)
df_pasien.to_csv('patient_records.csv', index=False)