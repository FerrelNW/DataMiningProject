# Import render_template to serve HTML files
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import traceback
import numpy as np # Ditambahkan untuk mengenali tipe data

app = Flask(__name__)
CORS(app)

# --- PHASE 1: Load all artifacts on server start ---
try:
    model = joblib.load("campaign_model.pkl")
    preprocessor = joblib.load("preprocessor_pipeline.pkl")
    feature_importance_df = joblib.load("feature_importance.pkl")
    # Muat juga urutan kolom yang benar
    model_columns = joblib.load("model_columns.pkl")
    print(">>> All artifacts loaded successfully.")
except Exception as e:
    print(f"!!! ERROR loading artifacts: {e}")
    model, preprocessor, feature_importance_df, model_columns = None, None, None, None

# --- PHASE 2: Create Routes ---

# Route to serve the main page (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Route to serve the dashboard page
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Route for dashboard data API
@app.route('/api/dashboard_data')
def get_dashboard_data():
    try:
        df_raw = pd.read_csv('customer_segmentation.csv')
        df_raw['Income'].fillna(df_raw['Income'].median(), inplace=True)
        edu_dist = df_raw['Education'].value_counts()
        df_raw['TotalSpending'] = df_raw.loc[:, 'MntWines':'MntGoldProds'].sum(axis=1)
        income_spending_data = df_raw[['Income', 'TotalSpending']].dropna().to_dict('records')
        income_spending_data_chartjs = [{'x': d['Income'], 'y': d['TotalSpending']} for d in income_spending_data]
        spending_marital = df_raw.groupby('Marital_Status')['TotalSpending'].mean().sort_values(ascending=False)
        spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        spending_composition = df_raw[spending_cols].sum().sort_values(ascending=False)

        dashboard_data = {
            "education_dist": {"labels": edu_dist.index.tolist(), "values": edu_dist.values.tolist()},
            "income_spending": {"data": income_spending_data_chartjs},
            "spending_by_marital": {"labels": spending_marital.index.tolist(), "values": spending_marital.values.tolist()},
            "spending_composition": {"labels": spending_composition.index.tolist(), "values": spending_composition.values.tolist()}
        }
        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route to handle prediction logic (FINAL FIX HERE)
@app.route('/predict', methods=['POST'])
def predict():
    if model and preprocessor and feature_importance_df is not None and model_columns is not None:
        try:
            json_data = request.get_json()
            print(f"\n[INFO] Received new data: {json_data}")

            # Buat DataFrame dari data input
            input_df_unordered = pd.DataFrame([json_data])

            # --- INI PERBAIKAN UTAMA ---
            # Pastikan DataFrame input memiliki kolom yang sama persis DAN URUTAN YANG SAMA
            # seperti data yang digunakan saat training (X)
            input_df = pd.DataFrame(columns=model_columns)
            input_df = pd.concat([input_df, input_df_unordered], ignore_index=True)
            input_df = input_df[model_columns] # Paksa urutan kolom sama

            # --- Logika XAI (sudah benar) ---
            top_features = feature_importance_df.head(5)
            reasons = []
            processed_feature_names = preprocessor.get_feature_names_out()

            for feature_name_processed in top_features['feature']:
                original_col = feature_name_processed.split('__')[1]
                if 'cat__' in feature_name_processed:
                    original_col = '_'.join(original_col.split('_')[:-1])

                if original_col in input_df.columns:
                    user_value = input_df[original_col].values[0]
                    if isinstance(user_value, np.integer): user_value = int(user_value)
                    elif isinstance(user_value, np.floating): user_value = float(user_value)
                    reasons.append({'feature': original_col, 'value': user_value})
                if len(reasons) >= 3: break

            # --- Proses data dan lakukan prediksi (sudah benar) ---
            processed_input = preprocessor.transform(input_df)
            prediction_prob = model.predict_proba(processed_input)[0][1]

            print(f"[INFO] Prediction probability: {prediction_prob:.4f}")
            return jsonify({
                'prediction_probability': prediction_prob,
                'reasons': reasons
            })

        except Exception as e:
            print(f"!!! ERROR during prediction: {e}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Model artifacts not loaded. Check server logs.'}), 500

# --- PHASE 3: Run the Server ---
if __name__ == '__main__':
    print(">>> Starting Flask server at http://1.0.0.1:5000")
    app.run(port=5000, debug=True)