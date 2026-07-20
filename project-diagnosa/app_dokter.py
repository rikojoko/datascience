import streamlit as st
import numpy as np
import joblib  # Wajib untuk memuat file .pkl langsung

# 1. Konfigurasi Halaman Web
st.set_page_config(
    page_title="Sistem AI Asisten Medis",
    page_icon="🏥",
    layout="centered"
)

# 2. MEMUAT MODEL YANG TERSIMPAN SECARA LANGSUNG
@st.cache_resource # Fitur Streamlit agar model tidak dimuat ulang setiap tombol diklik (bikin hemat memori)
def load_medical_models():
    try:
        model = joblib.load('medical_model.pkl')
        tfidf = joblib.load('medical_tfidf.pkl')
        params = joblib.load('scaling_params.pkl')
        return model, tfidf, params
    except Exception as e:
        st.error(f"Gagal memuat file .pkl! Pastikan file berada di folder yang sama. Error: {e}")
        return None, None, None

model, tfidf, params = load_medical_models()

# 3. Kamus Gejala Terapi Darurat (Rule-Based Fallback)
DATA_GEJALA_MEDIS = {
    "nyeri dada": {
        "risiko": "Tinggi Jantung",
        "terapi": [
            "Rujukan segera ke Kardiolog (Dokter Spesialis Jantung).",
            "Pemeriksaan EKG 12-lead dalam 10 menit pertama.",
            "Berikan terapi oksigen jika saturasi di bawah 94%."
        ]
    },
    "sesak napas": {
        "risiko": "Gangguan Respirasi / Jantung",
        "terapi": [
            "Pemeriksaan auskultasi paru dan rontgen dada (Thorax).",
            "Pemberian bronkodilator jika ditemukan mengorok/mengi.",
            "Posisikan pasien semi-fowler (setengah duduk) untuk meringankan napas."
        ]
    },
    "pusing hebat": {
        "risiko": "Krisis Hipertensi / Vertigo",
        "terapi": [
            "Istirahatkan pasien di ruangan tenang tanpa cahaya terang.",
            "Berikan obat antihipertensi oral sesuai instruksi dokter jika tensi sangat tinggi.",
            "Evaluasi defisit neurologis untuk memastikannya bukan gejala stroke."
        ]
    }
}

# 4. Tampilan Judul Dashboard Dokter
st.title("🏥 AI Medical Assistant & Therapy Dashboard")
st.markdown("---")

st.subheader("📋 Input Data Klinis Pasien")
col1, col2 = st.columns(2)

with col1:
    usia = st.number_input("Usia Pasien (Tahun):", min_value=1, max_value=120, value=45, step=1)
    tensi = st.slider("Tekanan Darah Sistolik (mmHg):", min_value=70, max_value=220, value=120, step=1)

with col2:
    gula_darah = st.slider("Kadar Gula Darah Puasa (mg/dL):", min_value=50, max_value=400, value=100, step=1)
    gejala = st.text_input("Keluhan / Gejala Utama:", value="nyeri dada")

# 5. Tombol Proses Eksekusi Medis
if st.button("🚀 Jalankan Diagnosis AI", type="primary"):
    gejala_input = gejala.lower().strip()
    
    # Nilai default jika model gagal
    hasil_diagnosis = "Risiko Rendah / Evaluasi Rutin"
    probabilitas = 0.10
    rekomendasi_terapi = [
        "Pertahankan pola hidup sehat dan olahraga teratur.",
        "Konsultasikan ke dokter umum jika gejala menetap lebih dari 3 hari."
    ]

    # PROSES JALUR MODEL .PKL LANGSUNG
    if model and tfidf and params:
        try:
            # Ekstraksi Fitur Teks
            fitur_teks = tfidf.transform([gejala_input]).toarray()
            
            # Penskalaan Fitur Numerik
            mean_usia, std_usia = params.get('usia', (40, 15))
            mean_td, std_td = params.get('tekanan_darah_sistolik', (120, 20))
            mean_gula, std_gula = params.get('kadar_gula_puasa', (100, 25))
            
            usia_scaled = (usia - mean_usia) / std_usia
            td_scaled = (tensi - mean_td) / std_td
            gula_scaled = (gula_darah - mean_gula) / std_gula
            
            fitur_numerik = np.array([[usia_scaled, td_scaled, gula_scaled]])
            
            # Gabungkan dan Potong 29 Kolom
            fitur_lengkap = np.hstack((fitur_numerik, fitur_teks))
            fitur_lengkap = fitur_lengkap[:, :29]
            
            # Prediksi Model
            prediksi_kelas = model.predict(fitur_lengkap)[0]
            probabilitas = float(model.predict_proba(fitur_lengkap)[0][prediksi_kelas])
            
            hasil_diagnosis = f"Hasil Model AI: Risiko Tinggi (Kelas {prediksi_kelas})" if prediksi_kelas == 1 else f"Hasil Model AI: Risiko Rendah (Kelas {prediksi_kelas})"
        except Exception as e:
            st.error(f"Gagal memproses data pada Model AI: {str(e)}")

    # JALUR 2: FILTER KLINIS (RULE-BASED SAFETY NET)
    if gejala_input in DATA_GEJALA_MEDIS:
        detail_medis = DATA_GEJALA_MEDIS[gejala_input]
        hasil_diagnosis = f"Gejala {gejala} -> Mengarah ke {detail_medis['risiko']}"
        rekomendasi_terapi = detail_medis['terapi'].copy()
        
        if tensi >= 140 or gula_darah >= 126:
            hasil_diagnosis += " (Kondisi Akut Terkonfirmasi)"
            probabilitas = max(probabilitas, 0.95)
            rekomendasi_terapi.insert(0, "PERINGATAN KRITIS: Tanda vital di atas batas normal!")
            
    elif tensi >= 140:
        hasil_diagnosis = "Hipertensi Tanpa Gejala Spesifik"
        probabilitas = max(probabilitas, 0.80)
        rekomendasi_terapi = [
            "Pemberian terapi lini pertama sesuai asesmen dokter.",
            "Diet rendah garam (Batasi natrium < 2000mg/hari).",
            "Jadwalkan kontrol ulang tekanan darah dalam 7 hari."
        ]

    # TAMPILKAN HASIL DI DASHBOARD
    st.markdown("---")
    st.subheader("🩺 Hasil Analisis Medis")
    st.info(f"**Kesimpulan Diagnosis:** {hasil_diagnosis}")
      
    
    # TAMPILKAN HASIL DI DASHBOARD BERDASARKAN TINGKAT RISIKO
    st.markdown("---")
    st.subheader("🩺 Hasil Analisis & Stratifikasi Risiko Medis")

    # Logika Penentuan Tingkat Risiko Klinis
    if "Kondisi Akut Terkonfirmasi" in hasil_diagnosis or (model and 'Kelas 1' in hasil_diagnosis):
        tingkat_risiko = "TINGGI (HIGH RISK)"
        warna_box = st.error
        pesan_risiko = "Pasien membutuhkan penanganan darurat segera atau rujukan ke dokter spesialis."
    elif tensi >= 130 or gula_darah >= 100 or "Hipertensi" in hasil_diagnosis:
        tingkat_risiko = "SEDANG (MODERATE RISK)"
        warna_box = st.warning
        pesan_risiko = "Pasien memerlukan observasi berkala, perubahan pola hidup, atau terapi lini pertama."
    else:
        tingkat_risiko = "RENDAH (LOW RISK)"
        warna_box = st.success
        pesan_risiko = "Kondisi pasien relatif stabil. Disarankan untuk tetap menjaga pola hidup sehat."

    # 1. Tampilkan Box Risiko Berwarna
    warna_box(f"### Kategori Risiko: {tingkat_risiko}\n\n{pesan_risiko}")

        # 2. Tampilkan Detail Analisis dengan Kolom Visual
    col_hasil1, col_hasil2 = st.columns(2)
    with col_hasil1:
        # PERBAIKAN: Hapus [0] karena prediksi_kelas sudah berupa angka langsung (bukan list)
        st.metric(label="Kesimpulan Diagnosis AI", value=f"Kelas {prediksi_kelas}" if model and 'prediksi_kelas' in locals() else "Evaluasi Gejala")
    with col_hasil2:
        st.metric(label="Tingkat Keyakinan Model AI", value=f"{probabilitas * 100:.1f}%")

    # 3. Tampilkan Rekomendasi Terapi Medis
    st.markdown("### 💊 Panduan Rencana Terapi & Penanganan:")
    for i, terapi in enumerate(rekomendasi_terapi, 1):
        st.write(f"**{i}.** {terapi}")
        
    st.markdown(" ")
    st.caption("⚠️ *Disclaimer: Hasil diagnosis stratifikasi risiko ini hanya alat bantu keputusan (Clinical Decision Support System). Keputusan akhir tetap berada di tangan dokter spesialis.*")
