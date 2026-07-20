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

# --- SETUP UI streamlit ---
# --- EKSEKUSI PREDIKSI ---