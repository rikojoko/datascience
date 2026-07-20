import streamlit as st
import pandas as pd
import time

# --- 1. PENGATURAN HALAMAN & HEADER ---
st.set_page_config(page_title="Demo UI Streamlit", page_icon="🚀", layout="centered")
st.title("🚀 Eksplorasi Elemen UI & Event Streamlit")
st.write("Skrip ini menunjukkan cara membuat komponen input dan menangkap interaksi pengguna secara real-time.")

st.markdown("---")

# --- 2. ELEMEN UI: INPUT DATA (FORM) ---
st.subheader("📋 Form Registrasi Pasien (Contoh Input)")

# Input Teks (Text Input)
nama = st.text_input("Nama Lengkap Pasien:", placeholder="Masukkan nama sesuai KTP...")

# Input Angka (Number Input)
usia = st.number_input("Usia (Tahun):", min_value=0, max_value=120, value=25)

# Pilihan Dropdown (Selectbox)
jenis_kelamin = st.selectbox("Jenis Kelamin:", ["Laki-laki", "Perempuan", "Tidak Ingin Menyebutkan"])

# Geseran Angka (Slider)
Skala_nyeri = st.slider("Skala Nyeri yang Dirasakan (1 - 10):", min_value=1, max_value=10, value=5)

# Kotak Centang (Checkbox)
setujuSK = st.checkbox("Saya menyetujui semua data di atas diisi secara benar.")

st.markdown("---")

# --- 3. EVENT HANDLING: MENANGKAP TOMBOL DIKLIK (Button Event) ---
st.subheader("⚡ Event Trigger")

# Event: Tombol diklik
if st.button("Proses Data Pasien", type="primary"):
    # Validasi input di dalam event tombol
    if not nama.strip():
        st.warning("⚠️ Nama pasien tidak boleh kosong!")
    elif not setujuSK:
        st.error("❌ Anda harus menyetujui syarat & ketentuan sebelum memproses.")
    else:
        # Menampilkan animasi loading (Spinner Event/Status)
        with st.spinner("Sedang mendaftarkan ke database sistem..."):
            time.sleep(1.5) # Simulasi proses jeda waktu
        
        # Sukses! Menampilkan Elemen Output
        st.success(f"🎉 Pasien bernama **{nama}** berhasil diproses!")
        
        # Menampilkan data ke dalam struktur tabel (Data Display Element)
        st.markdown("### 📊 Rangkuman Data Hasil Input:")
        data_hasil = {
            "Parameter": ["Nama", "Usia", "Gender", "Skala Nyeri"],
            "Nilai Input": [nama, f"{usia} Tahun", jenis_kelamin, f"{Skala_nyeri} / 10"]
        }
        st.table(pd.DataFrame(data_hasil))

# --- 4. EVENT HANDLING SECARA REAL-TIME (TENTATIVE DISPLAY) ---
# Tampilan ini akan berubah langsung secara otomatis TANPA perlu menekan tombol submit
st.sidebar.subheader("⚙️ Panel Kontrol Sidebar")
tampilkan_tips = st.sidebar.toggle("Tampilkan Tips Medis Hari Ini")

if tampilkan_tips:
    st.sidebar.info("💡 **Tips:** Minum air putih minimal 2 liter sehari dapat membantu menjaga konsentrasi dan hidrasi ginjal tetap optimal!")