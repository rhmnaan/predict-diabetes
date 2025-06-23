# Import library yang diperlukan untuk analisis data, machine learning.
import pandas as pd # Untuk manipulasi dan analisis data (DataFrame).
import numpy as np # Untuk operasi numerik.

from sklearn.model_selection import train_test_split # Untuk membagi dataset.
from sklearn.preprocessing import StandardScaler # Untuk penskalaan fitur.
from sklearn.neighbors import KNeighborsClassifier # Algoritma K-Nearest Neighbors.
from sklearn.metrics import (
    confusion_matrix, # Untuk evaluasi Confusion Matrix.
    accuracy_score,   # Untuk menghitung akurasi.
    precision_score,  # Untuk menghitung presisi.
    recall_score,     # Untuk menghitung recall.
    f1_score,         # Untuk menghitung F1-score.
    roc_auc_score,    # Untuk menghitung ROC AUC score.
    classification_report # Untuk laporan klasifikasi detail.
)
import joblib # Untuk menyimpan (serialize) model dan scaler.

# --- Bagian Khusus Google Colab (dapat diaktifkan jika berjalan di Colab) ---
# from google.colab import files


print("--- Memulai Pelatihan dan Penyimpanan Model KNN Secara Lokal (Sesuai Colab) ---")

# 1. Muat Dataset
# Memuat file 'diabetes.csv' ke dalam DataFrame Pandas.
try:
    data = pd.read_csv('diabetes.csv')
    print("Dataset 'diabetes.csv' berhasil dimuat.")
except FileNotFoundError:
    print("Error: File 'diabetes.csv' tidak ditemukan. Pastikan file ada di direktori yang sama.")
    exit()

print("\n--- Informasi Awal Dataset ---")
data.info()
print("\nRingkasan Statistik Deskriptif:")
print(data.describe())
print(f"\nBentuk Dataset (baris, kolom): {data.shape}")
print(f"\nJumlah nilai null di setiap kolom:\n{data.isnull().sum()}")


# 3. Deteksi dan Hapus Outlier dengan IQR (Interquartile Range)
print("\n--- Deteksi dan Hapus Outlier dengan IQR ---")
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# Hapus baris yang memiliki outlier
# Perhatian: Ini memodifikasi DataFrame 'data' secara in-place untuk langkah selanjutnya
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
print(f"Bentuk Dataset setelah outlier dihapus (baris, kolom): {data.shape}")
print("Contoh 5 baris pertama data setelah outlier dihapus:")
print(data.head()) # Ganti display() dengan print()

# 4. Pemisahan Fitur (X) dan Target (y)
# Gunakan 'data' yang sudah dibersihkan dari outlier
X = data.iloc[:, 0:8]
y = data.iloc[:, 8]

# 5. Pemisahan Data Latih dan Uji (Train-Test Split)
# xtr (X_train): Fitur data untuk melatih model
# xte (X_test): Fitur data untuk menguji model
# ytr (y_train): Label target untuk melatih
# yte (y_test): Label target untuk menguji
# Penting: random_state=42 digunakan untuk reproduktibilitas. Stratify tidak ada di kode Colab Anda.
xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nUkuran data latih (X_train): {xtr.shape}, Ukuran data uji (X_test): {xte.shape}")

# 6. Inisialisasi dan Latih StandardScaler
# FIX: Inisialisasi scaler di sini sebelum digunakan.
sc = StandardScaler()
xtr_scaled = sc.fit_transform(xtr) # FIT dan TRANSFORM data pelatihan
xte_scaled = sc.transform(xte)     # TRANSFORM data uji menggunakan scaler yang sama

print("\nStandardScaler berhasil dilatih dan data latih/uji discale.")
print("Contoh 10 baris pertama data latih setelah normalisasi:")
print(pd.DataFrame(xtr_scaled, columns=X.columns).head(10))

# 7. Gabungkan Data Latih dan Uji yang Sudah Discale (Untuk Tampilan)
train_data_df = pd.DataFrame(xtr_scaled, columns=X.columns)
train_data_df['Outcome'] = ytr.values
train_data_df['Set'] = 'Train'

test_data_df = pd.DataFrame(xte_scaled, columns=X.columns)
test_data_df['Outcome'] = yte.values
test_data_df['Set'] = 'Test'

split_data = pd.concat([train_data_df, test_data_df], ignore_index=True)
print("\nContoh 5 baris pertama data yang sudah dibagi dan discale:")
print(split_data.head()) # Ganti display() dengan print()

# 8. Inisialisasi dan Latih Model K-Nearest Neighbors (KNN)
# Menggunakan K=30, metrik jarak Euclidean.
clf_knn = KNeighborsClassifier(n_neighbors=30, p=2, metric='euclidean')
# FIX: Gunakan data yang sudah discale untuk pelatihan KNN
clf_knn.fit(xtr_scaled, ytr)
print("\nModel K-Nearest Neighbors selesai dilatih.")

# 9. Prediksi dan Evaluasi KNN pada Data Uji
# FIX: Gunakan data yang sudah discale untuk prediksi KNN
pred_knn = clf_knn.predict(xte_scaled)
# Dapatkan probabilitas untuk ROC AUC
proba_knn = clf_knn.predict_proba(xte_scaled)[:, 1] # Probabilitas kelas positif

print("\n--- Evaluasi K-Nearest Neighbors ---")
print("Confusion Matrix:\n", confusion_matrix(yte, pred_knn))

accuracy_knn = accuracy_score(yte, pred_knn)
precision_knn = precision_score(yte, pred_knn)
recall_knn = recall_score(yte, pred_knn)
f1_knn = f1_score(yte, pred_knn)
roc_auc_knn = roc_auc_score(yte, proba_knn)


print(f"Akurasi: {accuracy_knn:.4f}")
print(f"Presisi (Kelas Positif): {precision_knn:.4f}")
print(f"Recall (Kelas Positif): {recall_knn:.4f}")
print(f"F1 Score (Kelas Positif): {f1_knn:.4f}")
print(f"ROC AUC Score: {roc_auc_knn:.4f}")

# report_knn = classification_report(yte, pred_knn, output_dict=True) # Dihapus karena metrik dihitung manual
# print(f"Presisi (Kelas Positif): {report_knn['1']['precision']:.4f}")
# print(f"Recall (Kelas Positif): {report_knn['1']['recall']:.4f}")
# print(f"F1 Score (Kelas Positif): {report_knn['1']['f1-score']:.4f}")


# 10. Menyimpan Model KNN dan Scaler
# Menyimpan model KNN yang sudah dilatih dan objek StandardScaler.
knn_model_filename = 'knn_diabetes_model.joblib'
scaler_filename = 'standard_scaler_diabetes.joblib'

joblib.dump(clf_knn, knn_model_filename)
print(f"\nModel KNN berhasil disimpan sebagai '{knn_model_filename}'.")

joblib.dump(sc, scaler_filename)
print(f"Scaler berhasil disimpan sebagai '{scaler_filename}'.")

# --- Bagian Download (Hanya berfungsi di Google Colab) ---
# try:
#     files.download(knn_model_filename)
#     files.download(scaler_filename)
#     print(f"\nFile '{knn_model_filename}' dan '{scaler_filename}' telah dikirim untuk diunduh.")
#     print("Periksa folder unduhan (Downloads) di komputer Anda.")
# except NameError:
#     print("\nFungsi 'files.download' hanya tersedia di Google Colab.")
#     print("Anda dapat menemukan file model dan scaler di direktori lokal Anda.")

print("\nPelatihan dan penyimpanan model KNN selesai.")
