import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_koneksi_api():
    try:
        response = requests.get(f"{BASE_URL}/")
        return response.status_code == 200
    except:
        return False

def test_prediksi_multitask(teks_gejala, usia, tekanan_darah, kadar_gula):
    print(f"=== Menguji Input Gejala -> '{teks_gejala[:35]}...' ===")
    
    # PERBAIKAN: Tekanan darah dikirim sebagai string sesuai format training
    payload = {
        "symptom": teks_gejala,
        "usia": usia,
        "tekanan_darah": tekanan_darah, 
        "kadar_gula": kadar_gula
    }
    
    response = requests.post(f"{BASE_URL}/predict-multitask", json=payload)
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))
    print("-" * 50)

if __name__ == "__main__":
    if test_koneksi_api():
        print("✓ Server aktif. Memulai pengujian data klinis...\n")
        
        # Uji Kasus 1: Diare
        test_prediksi_multitask(
            teks_gejala="Sakit perut hebat, diare, dan muntah berkali-kali.",
            usia=35,
            tekanan_darah="110/70",
            kadar_gula=88
        )
        
        # Uji Kasus 2: Serangan Jantung
        test_prediksi_multitask(
            teks_gejala="Nyeri dada, sesak napas, berkeringat dingin, menjalar ke lengan kiri.",
            usia=60,
            tekanan_darah="160/100",
            kadar_gula=140
        )
    else:
        print("✗ Server FastAPI belum aktif. Jalankan uvicorn terlebih dahulu!")