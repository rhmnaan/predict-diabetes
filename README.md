# 🧠 Predict Diabetes

Aplikasi web Flask sederhana untuk memprediksi risiko diabetes menggunakan dua model Machine Learning: **Random Forest** dan **K-Nearest Neighbors (KNN)**. Aplikasi ini memungkinkan pengguna memasukkan data kesehatan mereka dan melihat prediksi dari kedua model, serta membandingkan kinerja keduanya.

---

## 🚀 Teknologi yang Digunakan

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" />
  <img src="https://img.shields.io/badge/TailwindCSS-06B6D4?style=for-the-badge&logo=tailwind-css&logoColor=white" />
  <img src="https://img.shields.io/badge/Jinja2-B41717?style=for-the-badge&logo=jinja&logoColor=white" />
  <img src="https://img.shields.io/badge/Joblib-00ADD8?style=for-the-badge" />
</p>

---

## 📚 Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Model yang Digunakan](#-model-yang-digunakan)
- [Dataset](#-dataset)
- [Struktur Proyek](#-struktur-proyek)
- [Instalasi](#-instalasi)
- [Penggunaan](#-penggunaan)
- [Deployment (Hosting)](#-deployment-hosting)
- [Kontribusi](#-kontribusi)
- [Lisensi](#-lisensi)
- [Penafian](#-penafian)

---

## ✅ Fitur Utama

- 🔹 **Antarmuka Web Sederhana:** Formulir interaktif untuk memasukkan data kesehatan.
- 🔹 **Dua Model Prediksi:** Random Forest & K-Nearest Neighbors digunakan secara bersamaan.
- 🔹 **Tampilan Kinerja Model:** Halaman khusus membandingkan metrik seperti Akurasi, Presisi, Recall, F1-Score, dan ROC AUC.
- 🔹 **Tampilan Dataset:** Halaman untuk menampilkan sebagian data dari dataset pelatihan.
- 🔹 **Preprocessing Konsisten:** Penanganan nilai hilang dan outlier diterapkan secara konsisten di pelatihan dan evaluasi.

---

## 🤖 Model yang Digunakan

- **Random Forest:** Algoritma ensemble berbasis banyak pohon keputusan. Tidak sensitif terhadap skala fitur.
- **K-Nearest Neighbors (KNN):** Mengklasifikasikan berdasarkan mayoritas tetangga terdekat. Sangat sensitif terhadap skala fitur — membutuhkan normalisasi (StandardScaler).

Kedua model dilatih pada dataset yang telah dibersihkan dengan penanganan nilai 0 dan outlier.

---

## 📊 Dataset

Dataset yang digunakan adalah **Pima Indian Diabetes Database** dari [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), yang berisi data medis dari wanita keturunan Indian-Pima berusia ≥21 tahun.

### Fitur Dataset:

- `Pregnancies`: Jumlah kehamilan  
- `Glucose`: Konsentrasi glukosa plasma  
- `BloodPressure`: Tekanan darah diastolik  
- `SkinThickness`: Ketebalan lipatan kulit  
- `Insulin`: Kadar insulin serum  
- `BMI`: Indeks massa tubuh  
- `DiabetesPedigreeFunction`: Fungsi silsilah diabetes  
- `Age`: Usia pasien  
- `Outcome`: 1 = Positif diabetes, 0 = Negatif diabetes  

---

## 📁 Struktur Proyek

