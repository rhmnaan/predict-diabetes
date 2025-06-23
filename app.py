import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

app = Flask(__name__)

# --- File Path ---
RF_MODEL_PATH = 'random_forest_diabetes_model.joblib'
KNN_MODEL_PATH = 'knn_diabetes_model.joblib'
SCALER_PATH = 'standard_scaler_diabetes.joblib'
DATA_PATH = 'diabetes.csv'

# --- Load Model ---
try:
    loaded_rf_model = joblib.load(RF_MODEL_PATH)
    loaded_knn_model = joblib.load(KNN_MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    print("Model RF, KNN, dan scaler berhasil dimuat.")
except FileNotFoundError as e:
    print(f"File tidak ditemukan: {e}")
    exit()

# --- Evaluasi RANDOM FOREST (tanpa hapus outlier, dengan imputasi median) ---
try:
    rf_data = pd.read_csv(DATA_PATH)
    rf_cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in rf_cols_to_impute:
        rf_data[col] = rf_data[col].replace(0, np.nan)
        rf_data[col].fillna(rf_data[col].median(), inplace=True)

    X_rf = rf_data.drop('Outcome', axis=1)
    y_rf = rf_data['Outcome']
    _, X_test_rf, _, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf)

    y_pred_rf = loaded_rf_model.predict(X_test_rf)
    y_proba_rf = loaded_rf_model.predict_proba(X_test_rf)[:, 1]

    rf_metrics = {
        'accuracy': accuracy_score(y_test_rf, y_pred_rf),
        'precision': precision_score(y_test_rf, y_pred_rf),
        'recall': recall_score(y_test_rf, y_pred_rf),
        'f1_score': f1_score(y_test_rf, y_pred_rf),
        'roc_auc': roc_auc_score(y_test_rf, y_proba_rf)
    }
except Exception as e:
    print(f"Gagal evaluasi Random Forest: {e}")
    rf_metrics = None

# --- Evaluasi KNN (hapus outlier, tidak imputasi, discale) ---
try:
    knn_data = pd.read_csv(DATA_PATH)
    Q1 = knn_data.quantile(0.25)
    Q3 = knn_data.quantile(0.75)
    IQR = Q3 - Q1
    knn_data = knn_data[~((knn_data < (Q1 - 1.5 * IQR)) | (knn_data > (Q3 + 1.5 * IQR))).any(axis=1)]

    X_knn = knn_data.drop('Outcome', axis=1)
    y_knn = knn_data['Outcome']
    _, X_test_knn, _, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)

    X_test_knn_scaled = loaded_scaler.transform(X_test_knn)
    y_pred_knn = loaded_knn_model.predict(X_test_knn_scaled)
    y_proba_knn = loaded_knn_model.predict_proba(X_test_knn_scaled)[:, 1]

    knn_metrics = {
        'accuracy': accuracy_score(y_test_knn, y_pred_knn),
        'precision': precision_score(y_test_knn, y_pred_knn),
        'recall': recall_score(y_test_knn, y_pred_knn),
        'f1_score': f1_score(y_test_knn, y_pred_knn),
        'roc_auc': roc_auc_score(y_test_knn, y_proba_knn)
    }
except Exception as e:
    print(f"Gagal evaluasi KNN: {e}")
    knn_metrics = None

# --- Preprocessing Input untuk Prediksi ---
def preprocess_input(data_dict, use_imputation=False):
    df = pd.DataFrame([data_dict])
    if use_imputation:
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            df[col] = df[col].replace(0, np.nan)
            df[col].fillna(df[col].median(), inplace=True)
    return df

# --- Routes ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
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
        return render_template('predict.html', error_message="Input tidak valid.")

    # --- Prediksi Random Forest (pakai imputasi) ---
    input_rf = preprocess_input(data, use_imputation=True)
    rf_pred = loaded_rf_model.predict(input_rf)[0]
    rf_proba = loaded_rf_model.predict_proba(input_rf)[:, 1][0]
    rf_result = "Diabetes" if rf_pred == 1 else "Tidak Diabetes"

    # --- Prediksi KNN (tanpa imputasi, tapi discale) ---
    input_knn = preprocess_input(data, use_imputation=False)
    input_knn_scaled = loaded_scaler.transform(input_knn)
    knn_pred = loaded_knn_model.predict(input_knn_scaled)[0]
    knn_proba = loaded_knn_model.predict_proba(input_knn_scaled)[:, 1][0]
    knn_result = "Diabetes" if knn_pred == 1 else "Tidak Diabetes"

    return render_template('predict.html',
        rf_prediction_text=f"Random Forest: {rf_result}",
        rf_confidence_text=f"Probabilitas: {rf_proba:.2%}",
        knn_prediction_text=f"K-Nearest Neighbors: {knn_result}",
        knn_confidence_text=f"Probabilitas: {knn_proba:.2%}"
    )

@app.route('/table')
def show_table():
    try:
        df = pd.read_csv(DATA_PATH)
        table_html = df.head(20).to_html(classes='data-table', index=False)
        return render_template('table.html', diabetes_table=table_html)
    except FileNotFoundError:
        return render_template('table.html', diabetes_table="<p style='color:red;'>Data tidak ditemukan.</p>")

@app.route('/models')
def show_model_performance():
    return render_template('models.html', rf_metrics=rf_metrics, knn_metrics=knn_metrics)

# if __name__ == '__main__':
#     app.run(debug=True)
