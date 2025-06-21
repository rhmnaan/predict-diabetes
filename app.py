import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify

# --- Inisialisasi Aplikasi Flask ---
app = Flask(__name__)

# --- Lokasi Model dan Data ---
RF_MODEL_PATH = 'random_forest_diabetes_model.joblib'
KNN_MODEL_PATH = 'knn_diabetes_model.joblib'
SCALER_PATH = 'standard_scaler_diabetes.joblib' # Path untuk StandardScaler
DATA_PATH = 'diabetes.csv'

# --- Load Model Random Forest ---
try:
    loaded_rf_model = joblib.load(RF_MODEL_PATH)
    print(f"Model Random Forest '{RF_MODEL_PATH}' berhasil dimuat.")
except FileNotFoundError:
    print(f"ERROR: Model Random Forest '{RF_MODEL_PATH}' tidak ditemukan. Pastikan file model ada.")
    exit()

# --- Load Model KNN dan Scaler ---
try:
    loaded_knn_model = joblib.load(KNN_MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH) # Muat scaler
    print(f"Model KNN '{KNN_MODEL_PATH}' dan Scaler '{SCALER_PATH}' berhasil dimuat.")
except FileNotFoundError:
    print(f"ERROR: Model KNN '{KNN_MODEL_PATH}' atau Scaler '{SCALER_PATH}' tidak ditemukan. Pastikan file ada.")
    exit()

# --- Median untuk Imputasi (Sama seperti saat pelatihan) ---
# PENTING: Nilai median ini HARUS berasal dari data *pelatihan*
# yang digunakan untuk melatih model Anda. Ini bisa dihitung sekali dan di-hardcode,
# atau disimpan bersama model. Untuk demo ini, kita akan hitung dari dataset asli lagi.
# Dalam aplikasi nyata, lebih baik menyimpan ini secara terpisah atau bersama model.
try:
    temp_df_for_medians = pd.read_csv(DATA_PATH)
    cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_impute:
        temp_df_for_medians[col] = temp_df_for_medians[col].replace(0, np.nan)
    
    # Simpan median ke dalam dictionary
    MEDIANS = {col: temp_df_for_medians[col].median() for col in cols_to_impute}
    # Dapatkan urutan kolom yang diharapkan dari dataset asli (tanpa 'Outcome')
    EXPECTED_COLUMNS = temp_df_for_medians.drop('Outcome', axis=1).columns.tolist()
    print("Median untuk imputasi dan urutan kolom diharapkan berhasil dihitung/didapat.")
except FileNotFoundError:
    print(f"ERROR: '{DATA_PATH}' tidak ditemukan. Median tidak dapat dihitung dan tabel tidak dapat ditampilkan.")
    exit()

# --- Fungsi Preprocessing ---
def preprocess_input(data_dict):
    """
    Melakukan preprocessing pada data input dari form.
    Menerima dictionary, mengembalikan DataFrame yang siap diprediksi (belum discale untuk KNN).
    """
    df_input = pd.DataFrame([data_dict])
    
    # Pastikan semua kolom numerik yang mungkin memiliki 0 diubah menjadi NaN
    for col in MEDIANS.keys():
        df_input[col] = df_input[col].replace(0, np.nan)
        # Imputasi NaN dengan median yang sudah dihitung
        df_input[col].fillna(MEDIANS[col], inplace=True)
        
    # Pastikan urutan kolom sesuai dengan saat model dilatih
    # Ini penting agar model menerima input dalam urutan yang benar
    return df_input[EXPECTED_COLUMNS]

# --- Route untuk Halaman Utama (Home Page - Informasi Saja) ---
@app.route('/')
def home():
    return render_template('home.html')

# --- Route untuk Halaman Prediksi (Form Input dan Hasil) ---
@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

# --- Route untuk Memproses Prediksi (Ketika Form Disubmit) ---
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Mengambil data dari form
        try:
            data = {
                'Pregnancies': int(request.form['Pregnancies']),
                'Glucose': int(request.form['Glucose']),
                'BloodPressure': int(request.form['BloodPressure']),
                'SkinThickness': int(request.form['SkinThickness']),
                'Insulin': int(request.form['Insulin']),
                'BMI': float(request.form['BMI']),
                'DiabetesPedigreeFunction': float(request.form['DiabetesPedigreeFunction']),
                'Age': int(request.form['Age'])
            }
        except ValueError:
            return render_template('predict.html', error_message="Input tidak valid. Pastikan semua kolom diisi dengan angka.")

        # Preprocess data input (unscaled)
        processed_data_unscaled = preprocess_input(data)

        # --- Prediksi dengan Random Forest ---
        rf_prediction = loaded_rf_model.predict(processed_data_unscaled)[0]
        rf_proba = loaded_rf_model.predict_proba(processed_data_unscaled)[:, 1][0]
        
        rf_result = "Diabetes" if rf_prediction == 1 else "Tidak Diabetes"
        rf_confidence = f"Probabilitas Diabetes: {rf_proba:.2%}"

        # --- Prediksi dengan KNN ---
        # Data harus discale untuk KNN menggunakan scaler yang sudah dimuat
        processed_data_scaled = loaded_scaler.transform(processed_data_unscaled)
        
        knn_prediction = loaded_knn_model.predict(processed_data_scaled)[0]
        knn_proba = loaded_knn_model.predict_proba(processed_data_scaled)[:, 1][0]
        
        knn_result = "Diabetes" if knn_prediction == 1 else "Tidak Diabetes"
        knn_confidence = f"Probabilitas Diabetes: {knn_proba:.2%}"

        # Kembali ke halaman prediksi dengan hasil dari kedua model
        return render_template('predict.html', 
                               rf_prediction_text=f"Random Forest: {rf_result}", 
                               rf_confidence_text=rf_confidence,
                               knn_prediction_text=f"K-Nearest Neighbors: {knn_result}", 
                               knn_confidence_text=knn_confidence)

# --- Route untuk Menampilkan Tabel Data ---
@app.route('/table')
def show_table():
    try:
        # Baca dataset asli
        df_full = pd.read_csv(DATA_PATH)
        # Konversi DataFrame ke HTML table
        # .to_html() akan menghasilkan string HTML dari DataFrame
        # .head(20) untuk hanya menampilkan 20 baris pertama agar tidak terlalu panjang
        diabetes_table_html = df_full.head(15).to_html(classes='data-table', index=False)
        return render_template('table.html', diabetes_table=diabetes_table_html)
    except FileNotFoundError:
        return render_template('table.html', diabetes_table="<p style='text-align:center; color:red;'>Error: File 'diabetes.csv' tidak ditemukan.</p>")

# --- Jalankan Aplikasi Flask ---
if __name__ == '__main__':
    app.run(debug=True)
