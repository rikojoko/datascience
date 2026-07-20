import requests

url = "http://127.0.0.1:5000/diagnose"
data_pasien = {
    "keluhan": "Nyeri perut hebat di bagian bawah kanan",
    "suhu": 38.5,
    "tensi": 110
}

response = requests.post(url, json=data_pasien)
print(response.json())