# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)

# === Load model ===
preprocessor = joblib.load('models/preprocessor_pipeline.pkl')
model = joblib.load('models/campaign_model.pkl')
feature_importance = joblib.load('models/feature_importance.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        X_processed = preprocessor.transform(input_df)
        prediction_prob = model.predict_proba(X_processed)[0][1]
        prediction_label = int(model.predict(X_processed)[0])

        response_data = {
            'prediction_probability': float(prediction_prob),
            'prediction_label': prediction_label,
            'reasons': [
                {'feature': 'Income', 'value': data.get('Income', 0)},
                {'feature': 'TotalSpending', 'value': data.get('TotalSpending', 0)},
                {'feature': 'Recency', 'value': data.get('Recency', 0)}
            ]
        }
        return jsonify(response_data)

    except Exception as e:
        print("!!! ERROR during prediction:", e)
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard_data', methods=['GET'])
def dashboard_data():
    try:
        df = pd.read_csv('customer_segmentation.csv')
        df['Income'] = df['Income'].fillna(0)

        # === 1. Buat Fitur Turunan ===
        
        spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        existing_spend_cols = [col for col in spending_cols if col in df.columns]
        df['TotalSpending'] = df[existing_spend_cols].sum(axis=1)

        df['Age'] = 2025 - df['Year_Birth']
        
        df['TotalChildren'] = df['Kidhome'] + df['Teenhome']
        df['Has_Children'] = (df['TotalChildren'] > 0).map({True: 'With Children', False: 'No Children'})

        # === 2. Hitung KPI (Key Performance Indicators) ===
        # === PERBAIKAN: Tambahkan .item() untuk konversi ke Python float ===
        kpis = {
            'total_customers': len(df), # len() sudah Python int, jadi aman
            'avg_income': df['Income'].mean().item(),
            'avg_spending': df['TotalSpending'].mean().item(),
            'conversion_rate': (df['Response'].mean() * 100).item()
        }

        # === 3. Siapkan Data Grafik ===
        
        age_bins = [18, 29, 39, 49, 59, 69, 100]
        age_labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
        age_dist = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False).value_counts().sort_index()
        
        # === PERBAIKAN: Tambahkan .item() untuk konversi ke Python int ===
        channel_totals = {
            'Web': df['NumWebPurchases'].sum().item(),
            'Catalog': df['NumCatalogPurchases'].sum().item(),
            'Store': df['NumStorePurchases'].sum().item(),
            'Deals': df['NumDealsPurchases'].sum().item()
        }
        
        family_dist = df['Has_Children'].value_counts()
        
        spending_comp_labels = [col for col in existing_spend_cols if col in ['MntWines', 'MntFruits', 'MntGoldProds', 'MntMeatProducts', 'MntSweetProducts']]
        spending_comp_values = [df[col].sum().item() for col in spending_comp_labels] # .item() sudah ada di sini
        
        spending_by_marital = df.groupby('Marital_Status')['TotalSpending'].mean()

        sample_n = min(1000, len(df))
        
        if sample_n > 0:
            df_sample = df.sample(n=sample_n, random_state=1, replace=False)
            income_spending_data = [
                {'x': row['Income'], 'y': row['TotalSpending']} 
                for _, row in df_sample.iterrows()
            ]
        else:
            income_spending_data = []

        # === 4. Kirim Semua Data ===
        data = {
            'kpis': kpis, # Sekarang sudah aman
            'age_dist': {
                'labels': age_dist.index.tolist(),
                'values': age_dist.values.tolist() # .tolist() sudah otomatis konversi
            },
            'channel_totals': {
                'labels': list(channel_totals.keys()),
                'values': list(channel_totals.values()) # Sekarang sudah aman
            },
            'family_dist': {
                'labels': family_dist.index.tolist(),
                'values': family_dist.values.tolist() # .tolist() sudah otomatis konversi
            },
            'spending_composition': {
                'labels': spending_comp_labels,
                'values': spending_comp_values
            },
            'spending_by_marital': {
                'labels': spending_by_marital.index.tolist(),
                'values': spending_by_marital.values.tolist() # .tolist() sudah otomatis konversi
            },
            'income_spending': {
                'data': income_spending_data
            }
        }
        return jsonify(data)

    except Exception as e:
        print("Dashboard data error:", e)
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)