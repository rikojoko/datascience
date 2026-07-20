import streamlit as st
import pandas as pd
import pickle
import os

# --- LOAD MODEL MULTI-OUTPUT ---
@st.cache_resource
def load_multitask_model():
    nama_file = 'model_multitask_diagnosa.pkl'
    if os.path.exists(nama_file):
        with open(nama_file, 'rb') as file:
            return pickle.load(file)
    return None

model = load_multitask_model()

# --- SETUP UI ---
st.set_page_config(page_title="AI Multi-Task Diagnosis", page_icon="🧠", layout="centered")
st.title("🧠 AI Multi-Output Assistant")
st.write("Model ini memprediksi **Diagnosa** sekaligus **Terapi** secara langsung menggunakan Machine Learning.")

if model is None:
    st.error("⚠️ File `model_multitask_diagnosa.pkl` tidak ditemukan! Jalankan script training dulu.")
else:
    # --- FORM INPUT ---
    with st.form(key='form_klinis'):
        st.subheader("📋 Input Kondisi Klinis")
        symptom_description = st.text_area("Keluhan Utama / Gejala", placeholder="Ketik gejala di sini...")
        
        col1, col2 = st.columns(2)
        with col1:
            usia = st.number_input("Usia", min_value=1, max_value=120, value=30)
        with col2:
            kadar_gula = st.number_input("Kadar Gula Darah (mg/dL)", min_value=50, max_value=500, value=100)
            
        tekanan_darah = st.selectbox("Tekanan Darah", ["120/80", "110/70", "130/85", "140/90", "160/100"])
        
        submit = st.form_submit_button("Jalankan Prediksi AI Ganda")

    # --- EKSEKUSI PREDIKSI ---
    if submit:
        if not symptom_description.strip():
            st.warning("Isi keluhan gejala terlebih dahulu.")
        else:
            # Siapkan DataFrame input
            data_pasien = pd.DataFrame({
                'symptom_description': [symptom_description],
                'usia': [usia],
                'tekanan_darah': [tekanan_darah],
                'kadar_gula': [kadar_gula]
            })
            
            # Model mengembalikan array 2 dimensi berisi [Diagnosa, Terapi]
            hasil_prediksi = model.predict(data_pasien)[0]
            
            prediksi_diagnosa = hasil_prediksi[0].replace('_', ' ').title()
            prediksi_terapi = hasil_prediksi[1]
            
            # Tampilkan Hasil Visual ke Dashboard
            st.success("### 📊 Hasil Prediksi Model Machine Learning")
            
            st.metric(label="Hasil Diagnosa Penyakit", value=prediksi_diagnosa)
            
            st.markdown("#### 💊 Rekomendasi Tindakan / Terapi (Hasil Prediksi AI):")
            st.info(prediksi_terapi)
            
            st.divider()
            st.caption("ℹ️ *Kedua output di atas diproduksi langsung secara matematis oleh model `.pkl` Anda berdasarkan pola dari data latih.*")