# Multi-Label Hate Speech and Abusive Language Classification

Sistem klasifikasi multi-label untuk mendeteksi **Hate Speech** dan **Abusive Language** pada tweet berbahasa Indonesia menggunakan metode **Long Short-Term Memory (LSTM)** dengan **GloVe Word Embedding**.

## Deskripsi

Project ini merupakan implementasi model Deep Learning untuk melakukan klasifikasi multi-label terhadap teks berbahasa Indonesia. Sistem menerima input berupa sebuah kalimat atau tweet, kemudian memprediksi apakah teks tersebut mengandung:

- Hate Speech
- Abusive Language

Model dibangun menggunakan arsitektur LSTM dengan representasi kata GloVe dan diimplementasikan menggunakan FastAPI sebagai backend serta React (Vite) sebagai frontend.

---

## Fitur

- Klasifikasi Hate Speech
- Klasifikasi Abusive Language
- Prediksi probabilitas masing-masing label
- REST API menggunakan FastAPI
- Antarmuka web sederhana untuk melakukan prediksi

---

## Teknologi

### Backend

- Python
- FastAPI
- PyTorch
- NLTK
- Pandas
- NumPy

### Frontend

- React
- Vite
- Tailwind CSS
- Axios

---

## Struktur Folder

```text
backend/
│
├── app/
├── model/
├── requirements.txt
└── venv/

frontend/
│
├── src/
├── public/
└── package.json
```

---

# Cara Menjalankan Project

## 1. Clone Repository

```bash
git clone https://github.com/kanahayeay/PDL-KLP8-KLASIFIKASI.git
cd PDL-KLP8-KLASIFIKASI
```

---

## 2. Menjalankan Backend

Masuk ke folder backend.

```bash
cd backend
```

Buat virtual environment.

```bash
python -m venv venv
```

Aktifkan virtual environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependency.

```bash
pip install -r requirements.txt
```

Download resource NLTK.

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords')"
```

Jalankan backend.

```bash
uvicorn app.main:app --reload
```

Backend akan berjalan di:

```
http://127.0.0.1:8000
```

Swagger API:

```
http://127.0.0.1:8000/docs
```

---

## 3. Menjalankan Frontend

Buka terminal baru.

Masuk ke folder frontend.

```bash
cd frontend
```

Install dependency.

```bash
npm install
```

Jalankan aplikasi.

```bash
npm run dev
```

Frontend biasanya dapat diakses pada:

```
http://localhost:5173
```

---

# Cara Menggunakan

1. Jalankan backend.
2. Jalankan frontend.
3. Masukkan teks pada halaman utama.
4. Klik tombol **Predict**.
5. Sistem akan menampilkan hasil prediksi Hate Speech dan Abusive Language beserta probabilitasnya.

---

# Contoh Output

Input

```
dasar bodoh
```

Output

```json
{
  "hate_speech": {
    "label": 1,
    "probability": 0.8978
  },
  "abusive": {
    "label": 1,
    "probability": 0.9985
  }
}
```

---

## Tim Pengembang

Kelompok 8

- Albertin Caecilia Djema
- Sherly Martha Revania
- Ni Ketut Sukardiasih
- Theresia Margaretha Purba
- Skye Kanahaya Endrawan
- I Gede Landip Anggareksa

Program Studi Informatika  
Universitas Udayana
