# Import library yang diperlukan untuk analisis data, machine learning, dan penyimpanan model.
import pandas as pd # Digunakan untuk manipulasi dan analisis data (DataFrame).
import numpy as np # Digunakan untuk operasi numerik, terutama untuk menangani nilai NaN.
from sklearn.model_selection import train_test_split # Untuk membagi dataset menjadi set pelatihan dan pengujian.
from sklearn.ensemble import RandomForestClassifier # Algoritma Random Forest untuk klasifikasi.
from sklearn.metrics import accuracy_score # Untuk menghitung akurasi
import joblib # Untuk menyimpan (serialize) model ke file.

print("--- Memulai Pelatihan dan Penyimpanan Model Random Forest Secara Lokal ---")

# 1. Muat Dataset
# Memuat file 'diabetes.csv' ke dalam DataFrame Pandas.
# Jika file tidak ditemukan, akan menampilkan pesan error dan keluar dari program.
try:
    data = pd.read_csv('diabetes.csv')
    print("Dataset 'diabetes.csv' berhasil dimuat.")
except FileNotFoundError:
    print("Error: File 'diabetes.csv' tidak ditemukan. Pastikan file ada di direktori yang sama.")
    exit()

# 2. Penanganan Nilai Hilang
# Mengidentifikasi kolom-kolom yang mungkin memiliki nilai '0' yang sebenarnya adalah nilai hilang (missing values).
# Mengganti nilai '0' dengan NaN, lalu mengisi NaN dengan nilai median dari masing-masing kolom.
cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_impute:
    data[col] = data[col].replace(0, np.nan)
    data[col].fillna(data[col].median(), inplace=True)
print("Penanganan nilai 0 (missing values) dengan median selesai.")

# 3. Pemisahan Fitur (X) dan Target (y)
# Memisahkan dataset menjadi fitur (variabel independen) dan target (variabel dependen 'Outcome').
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# 4. Pemisahan Data Latih dan Uji (Gunakan random_state yang konsisten: 42)
# Penting: Pastikan random_state ini sama dengan yang digunakan di app.py untuk evaluasi!
# xtr (X_train): Fitur data untuk melatih model
# xte (X_test): Fitur data untuk menguji model
# ytr (y_train): Label target untuk melatih
# yte (y_test): Label target untuk menguji
xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Ukuran data latih: {xtr.shape}, Ukuran data uji: {xte.shape}")

# 5. Inisialisasi dan Latih Model Random Forest
# Membuat instance RandomForestClassifier dengan parameter yang sama seperti di aplikasi Flask.
clf_rf = RandomForestClassifier(
    n_estimators=100,         # Jumlah pohon keputusan.
    random_state=42,          # Untuk reproduktibilitas.
    class_weight='balanced'   # Menangani ketidakseimbangan kelas.
)
clf_rf.fit(xtr, ytr) # Latih Random Forest pada data pelatihan (tidak perlu discale)
print("Model Random Forest selesai dilatih.")

# 6. Evaluasi Akurasi pada Data Uji
# Membuat prediksi pada data uji (xte)
y_pred_rf = clf_rf.predict(xte)
# Menghitung akurasi model pada data uji
accuracy_rf = accuracy_score(yte, y_pred_rf)
print(f"Akurasi Random Forest pada data uji: {accuracy_rf:.4f}")

# 7. Menyimpan Model Random Forest
# Menentukan nama file untuk model Random Forest.
# joblib.dump() digunakan untuk menyimpan objek model yang sudah dilatih ke file.
rf_model_filename = 'random_forest_diabetes_model.joblib'

joblib.dump(clf_rf, rf_model_filename)

print(f"\nModel Random Forest '{rf_model_filename}' berhasil disimpan di direktori lokal Anda.")
print("Anda sekarang bisa menjalankan 'app.py'.")
