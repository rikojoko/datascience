

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


