# predictive_model.py
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

# === 1. Load dataset ===
df = pd.read_csv("customer_segmentation.csv")

target_col = 'Response'
if target_col not in df.columns:
    raise ValueError(f"Column '{target_col}' not found in dataset. Please check column names.")

# === 2. Tambahkan fitur turunan agar cocok dengan form input ===

# -- Age
if 'Year_Birth' in df.columns:
    df['Age'] = 2025 - df['Year_Birth']
else:
    print("⚠️ Warning: 'Year_Birth' column missing, Age set to 0.")
    df['Age'] = 0

# -- TotalSpending
spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
existing_spend_cols = [c for c in spending_cols if c in df.columns]
if existing_spend_cols:
    df['TotalSpending'] = df[existing_spend_cols].sum(axis=1)
else:
    print("⚠️ Warning: No spending columns found, TotalSpending set to 0.")
    df['TotalSpending'] = 0

# -- Customer_Tenure_Days
if 'Dt_Customer' in df.columns:
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
    latest_date = df['Dt_Customer'].max()
    df['Customer_Tenure_Days'] = (latest_date - df['Dt_Customer']).dt.days
else:
    print("⚠️ Warning: 'Dt_Customer' missing, Customer_Tenure_Days set to 0.")
    df['Customer_Tenure_Days'] = 0

# -- TotalChildren
if {'Kidhome', 'Teenhome'}.issubset(df.columns):
    df['TotalChildren'] = df['Kidhome'] + df['Teenhome']
else:
    print("⚠️ Warning: 'Kidhome' or 'Teenhome' missing, TotalChildren set to 0.")
    df['TotalChildren'] = 0

# === 3. Pilih fitur sesuai form web ===
selected_features = [
    'Age', 'Income', 'Education', 'Marital_Status', 'TotalSpending', 'Recency',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
    'NumWebVisitsMonth', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
    'AcceptedCmp4', 'AcceptedCmp5', 'Complain', 'Customer_Tenure_Days', 'TotalChildren'
]

missing = [f for f in selected_features if f not in df.columns]
if missing:
    print(f"⚠️ Missing columns auto-created: {missing}")
    for col in missing:
        df[col] = 0

X = df[selected_features]
y = df[target_col]

# === 4. Tangani missing values ===
df = df.replace(r'^\s*$', np.nan, regex=True)
X = X.fillna(0)
y = y.fillna(0)

# === 5. Pisahkan numerik & kategorikal ===
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

# === 6. Preprocessing ===
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

# === 7. Model ===
model = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# === 8. Combine pipeline ===
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# === 9. Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 10. Train ===
print("\n🚀 Training model (features aligned with web form)...")
pipeline.fit(X_train, y_train)

# === 11. Evaluate ===
y_pred = pipeline.predict(X_test)
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# === 12. Feature importance ===
feature_names = (
    num_features +
    list(pipeline.named_steps['preprocessor']
         .named_transformers_['cat']
         .named_steps['encoder']
         .get_feature_names_out(cat_features))
)

importances = pipeline.named_steps['classifier'].feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# === 13. Save artifacts ===
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline.named_steps['preprocessor'], 'models/preprocessor_pipeline.pkl')
joblib.dump(pipeline.named_steps['classifier'], 'models/campaign_model.pkl')
joblib.dump(feature_names, 'models/model_columns.pkl')
joblib.dump(feature_importance, 'models/feature_importance.pkl')

print("\n🎯 Model training complete and saved successfully in /models/")
