import streamlit as st
import pandas as pd
import pickle
import os

# --- KELOLA LOAD MODEL ---
@st.cache_resource
def load_model():
    nama_file = 'model_diagnosa_multimodal.pkl'
    if os.path.exists(nama_file):
        with open(nama_file, 'rb') as file:
            return pickle.load(file)
    return None

model = load_model()

# --- DATABASE TERAPI BERDASARKAN DIAGNOSA ---
# Menyesuaikan dengan daftar kelas penyakit di dataset
kamus_terapi = {
    'influenza': {
        'obat': "Antipiretik/analgesik (Paracetamol), dekongestan, atau antivirus (sesuai resep dokter jika parah).",
        'tindakan': "Istirahat total (bed rest), minum air putih minimal 2 liter/hari, konsumsi makanan hangat, dan isolasi mandiri agar tidak menular."
    },
    'alergi': {
        'obat': "Antihistamin (misal: Cetirizine, Loratadine) dan salep kortikosteroid ringan jika ada ruam kulit.",
        'tindakan': "Identifikasi dan hindari pemicu alergi (seafood, debu, dingin). Kompres dingin pada area kulit yang bengkak/gatal."
    },
    'depresi': {
        'obat': "Terapi farmakologi (Antidepresan) *hanya* diberikan melalui resep dokter spesialis kejiwaan (Psikiater).",
        'tindakan': "Sangat disarankan melakukan konseling/psikoterapi dengan psikolog/psikiater, bercerita kepada orang terpercaya, menjaga rutinitas tidur, dan olahraga ringan."
    },
    'migrain': {
        'obat': "Analgesik (Ibuprofen, Paracetamol), atau obat golongan triptan untuk migrain berat.",
        'tindakan': "Istirahat di ruangan yang gelap, tenang, dan bebas dari kebisingan. Kompres dingin di bagian belakang leher atau dahi."
    },
    'flu_biasa': {
        'obat': "Obat flu bebas (pemberhentian gejala pilek/batuk) dan vitamin C untuk meningkatkan imun.",
        'tindakan': "Perbanyak minum air hangat, istirahat yang cukup, dan konsumsi sup hangat untuk melegakan tenggorokan."
    },
    'gangguan_kecemasan': {
        'obat': "Obat penenang (Anxiolytics) jika diresepkan oleh dokter pada kondisi akut.",
        'tindakan': "Latih teknik pernapasan dalam (*deep breathing*), meditasi, kurangi konsumsi kafein/alkohol, dan lakukan terapi perilaku kognitif (CBT)."
    },
    'diare': {
        'obat': "Oralit untuk mencegah dehidrasi, suplemen Zinc, dan obat anti-diare bila direkomendasikan dokter.",
        'tindakan': "Minum cairan elektrolit secara berkala, hindari makanan pedas, berlemak, dan produk susu sementara waktu. Makan makanan lunak seperti bubur."
    },
    'bronkitis': {
        'obat': "Mukolitik/Ekspektoran (pengencer dahak), obat pereda batuk, inhaler jika sesak, atau antibiotik jika ada infeksi bakteri.",
        'tindakan': "Gunakan *humidifier* (pelembap udara), hindari asap rokok dan polusi udara, serta perbanyak minum air hangat."
    },
    'demam_berdarah': {
        'obat': "Paracetamol untuk demam (HINDARI Ibuprofen/Aspirin karena memicu pendarahan).",
        'tindakan': "Rawat inap sangat disarankan untuk monitoring trombosit. Minum cairan dalam jumlah banyak (air, jus, larutan isotonik) dan istirahat total."
    },
    'diabetes': {
        'obat': "Obat Hipoglikemik Oral (Metformin, dll) atau terapi Insulin sesuai instruksi ketat dokter spesialis penyakit dalam.",
        'tindakan': "Batasi konsumsi karbohidrat sederhana dan gula, lakukan olahraga kardio teratur, serta cek kadar gula darah secara berkala."
    },
    'eksim': {
        'obat': "Pelembap kulit khusus (*emollient*), salep kortikosteroid untuk meredakan radang hebat.",
        'tindakan': "Gunakan sabun mandi berbahan lembut tanpa parfum, hindari menggaruk area yang gatal, dan gunakan pakaian berbahan katun longgar."
    },
    'maag': {
        'obat': "Antasida, H2 Blocker (Ranitidin), atau PPI (Omeprazole) diminum sebelum makan.",
        'tindakan': "Makan dalam porsi kecil namun sering, hindari makanan pedas, asam, kopi, dan stres. Jangan langsung berbaring setelah makan."
    },
    'stres': {
        'obat': "Suplemen vitamin B-kompleks atau herbal penenang (seperti teh chamomile).",
        'tindakan': "Lakukan manajemen waktu yang baik, sempatkan waktu untuk hobi (*self-care*), olahraga rutin, dan lakukan teknik relaksasi."
    },
    'anemia': {
        'obat': "Suplemen Zat Besi (Iron/Sangobion) dan Asam Folat.",
        'tindakan': "Tingkatkan konsumsi makanan kaya zat besi (daging merah, hati ayam, bayam) dan makanan tinggi vitamin C untuk membantu penyerapan zat besi."
    },
    'serangan_jantung': {
        'obat': "Aspirin kunyah (jika tersedia dan diinstruksikan darurat medis) sambil menunggu bantuan datang.",
        'tindakan': "🚨 **KONDISI DARURAT MEDIS!** Segera hubungi ambulans (119) atau bawa pasien ke UGD rumah sakit terdekat. Posisikan pasien duduk rileks, longgarkan pakaiannya, dan temani terus."
    }
}

# --- ATURAN HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="AI Diagnosis & Terapi Asisten",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Asisten Diagnosa & Terapi Medis (AI)")
st.write("Masukkan data klinis pasien untuk mendapatkan estimasi diagnosa awal beserta saran terapinya.")

if model is None:
    st.error("⚠️ File `model_diagnosa_multimodal.pkl` tidak ditemukan!")
else:
    # --- FORM INPUTAN PASIEN ---
    with st.form(key='form_pasien'):
        st.subheader("📋 Data Umum & Klinis Pasien")
        
        symptom_description = st.text_area(
            label="Deskripsi Gejala / Keluhan Pasien",
            placeholder="Contoh: Sakit perut hebat, diare, dan muntah berkali-kali...",
        )
        
        col1, col2 = st.columns(2)
        with col1:
            usia = st.number_input("Usia Pasien (Tahun)", min_value=1, max_value=120, value=30)
        with col2:
            kadar_gula = st.number_input("Kadar Gula Darah sewaktu (mg/dL)", min_value=50, max_value=500, value=100)
            
        tekanan_darah = st.selectbox(
            label="Tekanan Darah (Sistolik/Diastolik)",
            options=["120/80", "110/70", "115/75", "118/78", "125/80", "130/85", "130/90", "135/85", "140/90", "150/95", "160/100"]
        )
        
        submit_button = st.form_submit_button(label='Analisis Diagnosa & Terapi')

    # --- PROSES PREDIKSI & PENAMPILAN TERAPI ---
    if submit_button:
        if not symptom_description.strip():
            st.warning("⚠️ Mohon isi deskripsi gejala pasien terlebih dahulu.")
        else:
            data_input_baru = pd.DataFrame({
                'symptom_description': [symptom_description],
                'usia': [usia],
                'tekanan_darah': [tekanan_darah],
                'kadar_gula': [kadar_gula]
            })
            
            try:
                # Prediksi Penyakit
                prediksi = model.predict(data_input_baru)[0]
                probabilitas = model.predict_proba(data_input_baru)
                nilai_yakin = max(probabilitas[0]) * 100
                
                hasil_diagnosa = prediksi.replace('_', ' ').title()
                
                # Tampilkan Hasil Diagnosa
                st.success("### 📊 Hasil Analisis AI")
                st.metric(label="Estimasi Diagnosa Penyakit", value=hasil_diagnosa)
                st.caption(f"Tingkat keyakinan model (*Confidence Score*): **{nilai_yakin:.2f}%**")
                
                # Tampilkan Rencana Terapi berdasarkan hasil diagnosa
                st.subheader("💊 Rekomendasi Terapi & Penanganan Awal")
                if prediksi in kamus_terapi:
                    terapi = kamus_terapi[prediksi]
                    
                    # Menggunakan struktur kolom/card agar menarik
                    st.markdown(f"**💊 Terapi Farmakologi (Obat/Suplemen):**")
                    st.info(terapi['obat'])
                    
                    st.markdown(f"**🏃‍♂️ Terapi Non-Farmakologi (Tindakan/Gaya Hidup):**")
                    st.success(terapi['tindakan'])
                else:
                    st.info("Rekomendasi terapi spesifik belum tersedia untuk penyakit ini.")
                
                # Warning disclaimer medis
                st.divider()
                st.warning(
                    "**Disclaimer:** Rekomendasi terapi ini bersifat edukasi penanganan pertama berbasis AI. "
                    "Segala bentuk pemberian obat keras harus melalui resep dan pemeriksaan langsung oleh Dokter."
                )
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")