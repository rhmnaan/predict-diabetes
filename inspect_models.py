import joblib
import pandas as pd # Dibutuhkan jika objek yang dimuat adalah DataFrame atau memiliki ketergantungan pada Pandas
import numpy as np  # Dibutuhkan jika objek yang dimuat memiliki ketergantungan pada NumPy
from sklearn.ensemble import RandomForestClassifier # Dibutuhkan jika objek yang dimuat adalah RandomForestClassifier
from sklearn.preprocessing import StandardScaler    # Dibutuhkan jika objek yang dimuat adalah StandardScaler
from sklearn.neighbors import KNeighborsClassifier  # Dibutuhkan jika objek yang dimuat adalah KNeighborsClassifier

print("--- Membaca Isi File .joblib ---")

# --- Bagian 1: Membaca Model Random Forest ---
rf_model_filename = 'random_forest_diabetes_model.joblib'

try:
    # Memuat model Random Forest
    loaded_rf_model = joblib.load(rf_model_filename)
    print(f"\nBerhasil memuat model: {rf_model_filename}")
    print(f"Tipe objek yang dimuat: {type(loaded_rf_model)}")

    # Contoh melihat beberapa atribut dari model Random Forest
    if isinstance(loaded_rf_model, RandomForestClassifier):
        print(f"Jumlah estimator (pohon): {loaded_rf_model.n_estimators}")
        print(f"Fitur yang digunakan saat pelatihan: {loaded_rf_model.n_features_in_}")
        # Jika Anda melatih model dengan kolom DataFrame, Anda bisa mencoba ini (tidak selalu ada):
        # print(f"Nama fitur: {loaded_rf_model.feature_names_in_}") # Atribut ini mungkin tidak selalu ada tergantung versi scikit-learn
        print("\nBeberapa feature importances teratas (jika tersedia):")
        # Contoh sederhana untuk feature importances jika dilatih dengan nama kolom
        # Asumsi: Anda tahu nama kolom asli
        try:
            # Ini hanya contoh. Anda perlu tahu nama fitur asli Anda.
            # Misalnya, jika X.columns dari data asli Anda adalah:
            # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            feature_importances = pd.Series(loaded_rf_model.feature_importances_, index=feature_names)
            print(feature_importances.sort_values(ascending=False).head())
        except Exception as e:
            print(f"Tidak dapat menampilkan feature importances secara detail: {e}")
            print("Pastikan Anda memiliki nama fitur asli yang sesuai.")
    else:
        print("Objek yang dimuat bukan RandomForestClassifier.")

except FileNotFoundError:
    print(f"Error: File '{rf_model_filename}' tidak ditemukan.")
except Exception as e:
    print(f"Terjadi error saat memuat atau membaca {rf_model_filename}: {e}")


# --- Bagian 2: Membaca Scaler (StandardScaler) ---
scaler_filename = 'standard_scaler_diabetes.joblib'

try:
    # Memuat objek StandardScaler
    loaded_scaler = joblib.load(scaler_filename)
    print(f"\nBerhasil memuat scaler: {scaler_filename}")
    print(f"Tipe objek yang dimuat: {type(loaded_scaler)}")

    # Contoh melihat beberapa atribut dari StandardScaler
    if isinstance(loaded_scaler, StandardScaler):
        print(f"Mean dari fitur (dipelajari): {loaded_scaler.mean_}")
        print(f"Variance dari fitur (dipelajari): {loaded_scaler.var_}")
        print(f"Skala dari fitur (standar deviasi): {loaded_scaler.scale_}")
        print(f"Jumlah fitur yang diharapkan: {loaded_scaler.n_features_in_}")
    else:
        print("Objek yang dimuat bukan StandardScaler.")

except FileNotFoundError:
    print(f"Error: File '{scaler_filename}' tidak ditemukan.")
except Exception as e:
    print(f"Terjadi error saat memuat atau membaca {scaler_filename}: {e}")


# --- Bagian 3: Membaca Model KNN ---
knn_model_filename = 'knn_diabetes_model.joblib'

try:
    # Memuat model KNN
    loaded_knn_model = joblib.load(knn_model_filename)
    print(f"\nBerhasil memuat model: {knn_model_filename}")
    print(f"Tipe objek yang dimuat: {type(loaded_knn_model)}")

    # Contoh melihat beberapa atribut dari model KNN
    if isinstance(loaded_knn_model, KNeighborsClassifier):
        print(f"Jumlah tetangga (n_neighbors): {loaded_knn_model.n_neighbors}")
        print(f"Metrik jarak: {loaded_knn_model.metric}")
        print(f"P-value untuk metrik Minkowski: {loaded_knn_model.p}")
    else:
        print("Objek yang dimuat bukan KNeighborsClassifier.")

except FileNotFoundError:
    print(f"Error: File '{knn_model_filename}' tidak ditemukan.")
except Exception as e:
    print(f"Terjadi error saat memuat atau membaca {knn_model_filename}: {e}")

