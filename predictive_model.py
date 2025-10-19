# predictive_model_v2.py
# ======================================================
# Improved Predictive Model for Marketing Campaign Response
# By: [Your Name]
# ======================================================

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# ======================================================
# 1. Load and Clean Dataset
# ======================================================
print(">>> Loading and cleaning data...")
df = pd.read_csv('customer_segmentation.csv')

# Handle missing income
df['Income'] = df['Income'].fillna(df['Income'].median())

# Drop unnecessary columns
df.drop(['ID', 'Z_CostContact', 'Z_Revenue'], axis=1, inplace=True)

# ======================================================
# 2. Feature Engineering (Smart New Features)
# ======================================================
print(">>> Performing feature engineering...")

# Date processing
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
latest_date = df['Dt_Customer'].max() + pd.DateOffset(days=1)

df['Age'] = latest_date.year - df['Year_Birth']
df['Customer_Tenure_Days'] = (latest_date - df['Dt_Customer']).dt.days
df['TotalSpending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                          'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
df['TotalChildren'] = df['Kidhome'] + df['Teenhome']

# Standardize education & marital status
df['Education'] = df['Education'].replace({
    'Graduation': 'Graduate', 'PhD': 'Postgraduate', 'Master': 'Postgraduate',
    '2n Cycle': 'Graduate', 'Basic': 'Undergraduate'
})
df['Marital_Status'] = df['Marital_Status'].replace({
    'Married': 'Partner', 'Together': 'Partner',
    'Single': 'Alone', 'Divorced': 'Alone', 'Widow': 'Alone',
    'Absurd': 'Alone', 'YOLO': 'Alone'
})

# Add marketing-related derived features
df['OnlinePurchaseRatio'] = df['NumWebPurchases'] / (df['NumStorePurchases'] + 1)
df['PromotionAcceptanceRate'] = df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                                    'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1) / 5
df['AvgSpendingPerChild'] = df['TotalSpending'] / (df['TotalChildren'] + 1)
df['Frequency'] = df[['NumDealsPurchases', 'NumWebPurchases',
                      'NumCatalogPurchases', 'NumStorePurchases']].sum(axis=1)
df['RecencyScore'] = (100 - df['Recency']) / 100  # Normalize recency (lower better)
df['MonetaryScore'] = df['TotalSpending'] / df['Income']
df['RFM_Score'] = (df['RecencyScore'] + (df['Frequency'] / df['Frequency'].max()) +
                   (df['MonetaryScore'] / df['MonetaryScore'].max())) / 3

# Drop unused columns
df.drop([
    'Year_Birth', 'Dt_Customer', 'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'Kidhome', 'Teenhome'
], axis=1, inplace=True)

# ======================================================
# 3. Prepare Data for Modeling
# ======================================================
print(">>> Preparing data for model...")
X = df.drop('Response', axis=1)
y = df['Response']

numerical_cols = X.select_dtypes(include=['number']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# ======================================================
# 4. Train-Test Split + Balancing (SMOTE)
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)

# ======================================================
# 5. Model Training with Tuning
# ======================================================
print(">>> Training model with hyperparameter tuning...")
base_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    base_model, param_grid, cv=3,
    scoring='roc_auc', n_jobs=-1, verbose=1
)
grid_search.fit(X_train_res, y_train_res)

best_model = grid_search.best_estimator_
print(f">>> Best Model Params: {grid_search.best_params_}")

# ======================================================
# 6. Evaluation
# ======================================================
print("\n--- Model Performance ---")
y_pred = best_model.predict(X_test_processed)
y_prob = best_model.predict_proba(X_test_processed)[:, 1]

print(classification_report(y_test, y_pred, target_names=['Did not Respond (0)', 'Responded (1)']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# ======================================================
# 7. Save Artifacts
# ======================================================
print("\n>>> Saving model artifacts...")
joblib.dump(preprocessor, 'preprocessor_pipeline.pkl')
joblib.dump(best_model, 'campaign_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

# Save feature importance
feature_names = preprocessor.get_feature_names_out()
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)
joblib.dump(feature_importance_df, 'feature_importance.pkl')

