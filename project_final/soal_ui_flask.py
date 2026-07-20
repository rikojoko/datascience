import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Konfigurasi Path Model Multi-Output Anda


# Memuat model saat server startup



# Route Utama: Tampilan Awal Sistem (Keadaan Standby)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Route Analisis: Memproses Form POST dan Mengembalikan Dua Output Sekaligus
@app.route('/prediksi', methods=['POST'])
def prediksi():
    pass
        


if __name__ == '__main__':
    # Jalankan server lokal di port 5000 dengan mode debug aktif
    app.run(debug=True, port=8000)