import requests
import json

# URL endpoint FastAPI lokal Anda
URL = "http://127.0.0.1:8000/predict/"

# Data simulasi pasien yang akan dikirim ke API
data_pasien = {
    "usia": 58,
    "tekanan_darah_sistolik": 155,
    "kadar_gula_puasa": 135,
    "gejala_utama": "nyeri dada"
}

print("=== MEMULAI UJI COBA API MEDIS LOKAL ===")
print(f"Mengirim data pasien ke: {URL}\n")

try:
    # Mengirim permintaan POST dengan data format JSON
    response = requests.post(URL, json=data_pasien, timeout=5)
    
    # Memeriksa apakah server merespons dengan status sukses (200 OK)
    if response.status_code == 200:
        hasil = response.json()
        
        # Menampilkan output secara rapi di terminal
        print("[STATUS]: Berhasil Terhubung ke API\n")
        print(f"Hasil Diagnosis : {hasil['hasil_diagnosis']}")
        print(f"Tingkat Akurasi : {hasil['probabilitas_akurasi']}")
        print("\nRekomendasi Terapi Medis:")
        
        for i, terapi in enumerate(hasil['rekomendasi_terapi'], 1):
            print(f"{i}. {terapi}")
            
    else:
        print(f"[ERROR]: Server merespons dengan kode status {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("[ERROR]: Gagal terhubung ke server!")
    print("Pastikan server FastAPI Anda sudah berjalan menggunakan perintah: uvicorn main:app --reload")
except Exception as e:
    print(f"[ERROR]: Terjadi kesalahan sistem: {str(e)}")

print("\n=== UJI COBA SELESAI ===")