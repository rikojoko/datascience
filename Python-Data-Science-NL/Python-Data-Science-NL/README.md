# Python Data Science

## Natural Language Processing (NLP) Fundamentals

Natural Language Processing (NLP) adalah cabang ilmu komputer dan kecerdasan buatan yang fokus pada bagaimana mesin memahami, memproses, dan menghasilkan bahasa manusia. NLP sering digunakan untuk analisis teks, chatbot, klasifikasi sentimen, penerjemahan, dan sistem pencarian.

### 1. Apa itu NLP?

NLP membantu komputer untuk bekerja dengan data teks seperti artikel, ulasan, email, dan percakapan. Tujuannya adalah mengubah bahasa manusia menjadi bentuk yang dapat diproses oleh algoritma.

Contoh penerapan NLP:
- Analisis sentimen pada ulasan produk
- Chatbot dan asisten virtual
- Pengenalan topik dari dokumen
- Klasifikasi email spam atau bukan spam
- Penerjemahan bahasa otomatis

### 2. Tahapan dasar dalam NLP

Berikut adalah tahapan umum yang sering digunakan dalam proyek NLP:

1. Pengumpulan data teks
2. Pembersihan teks
3. Tokenisasi
4. Penghilangan kata stopwords
5. Stemming atau lemmatization
6. Representasi teks menjadi angka
7. Pelatihan model

### 3. Contoh preprocessing teks sederhana

Sebelum teks diproses lebih lanjut, biasanya kita membersihkan teks terlebih dahulu agar lebih rapi dan konsisten.

```python
import re
from collections import Counter

text = "Natural Language Processing membantu komputer memahami bahasa manusia. NLP sangat berguna dalam analisis teks."

# Mengubah teks menjadi huruf kecil dan menghapus tanda baca
text_clean = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()

# Memecah teks menjadi kata-kata
tokens = text_clean.split()

# Kata-kata yang tidak terlalu penting
stop_words = {"dan", "di", "yang", "untuk", "dalam", "adalah", "pada", "sangat"}

# Menghapus stop words
filtered_tokens = [token for token in tokens if token not in stop_words]

# Menghitung frekuensi kata
word_freq = Counter(filtered_tokens)

print("Teks bersih:", text_clean)
print("Token:", tokens)
print("Token setelah filtering:", filtered_tokens)
print("Frekuensi kata:", word_freq)
```

### 4. Penjelasan singkat contoh

- `re.sub(...)` digunakan untuk membersihkan teks dari tanda baca.
- `split()` memecah teks menjadi token per kata.
- `stop_words` berisi kata-kata yang biasanya tidak memberikan makna penting.
- `Counter` digunakan untuk menghitung seberapa sering setiap kata muncul.

### 5. Kenapa NLP penting?

NLP penting karena banyak data yang tersedia dalam bentuk teks, terutama di media sosial, email, dokumen, dan website. Dengan NLP, kita bisa mengekstrak informasi berharga dari data tersebut dan membuat sistem yang lebih cerdas.

### 6. Instalasi dependensi

Sebelum menjalankan contoh NLP, pastikan Anda telah menginstal semua library yang dibutuhkan.

### 7. Kesimpulan

NLP adalah fondasi penting dalam analisis teks dan kecerdasan buatan. Dengan memahami dasar-dasarnya, Anda dapat mulai membangun aplikasi seperti chatbot, analisis sentimen, dan sistem pencarian berbasis teks.

#### Cara install

Jalankan perintah berikut di terminal Anda:

```bash
pip install -r requirements.txt
```

Jika Anda menggunakan virtual environment, aktifkan terlebih dahulu sebelum menjalankan perintah di atas.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Catatan

- Untuk sistem Windows, gunakan `venv\Scripts\activate`.
- Jika mengalami masalah saat install, pastikan pip sudah terbaru:

```bash
python -m pip install --upgrade pip
```
## solusi codespace sinkro repo 
1. Cek Status Git
```
cd /workspaces/Python-Data-Science
git status
```
2. Cek Remote Repository
```
git remote -v
```
3. Cek Branch Saat Ini
```
git branch -a
```
### Kemungkinan Masalah & Solusi:
A. Branch tidak tersinkronisasi:
```
git fetch origin
git pull origin NeuralNetwork
```
B. Ada perubahan lokal yang belum di-commit:
```
git add .
git commit -m "Update materi neural network dan requirements"
git push origin NeuralNetwork
```
C. Jika ada konflik dengan default branch (Main):
```
git fetch origin
git merge origin/Main
# Resolve conflicts jika ada
git push origin NeuralNetwork
```
D. Reset ke remote state (jika ingin mulai bersih):
```
git fetch origin
git reset --hard origin/NeuralNetwork
```
4. Verifikasi Autentikasi
Pastikan GitHub credentials sudah ter-setup:
```
git config --list | grep user
gh auth status
```
Jika belum login, gunakan:
```
gh auth login
```
menjalankan perintah 
```
git status  
git remote -v
```


