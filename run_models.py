# Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# Load and filter dataset

df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

# Keep only diabetic and pre-diabetic cases
df = df[df['Diabetes_012'] > 0]
df.reset_index(drop=True, inplace=True)


# Feature engineering

df['patient_characteristics'] = (
    df['BMI'] * 0.3 +
    df['HighBP'] * 0.2 +
    df['HighChol'] * 0.3 +
    df['Age'] * 0.2
)

df['hypertension_highChol'] = df['HighBP'] * df['HighChol']
df['Smoking'] = df['Smoker'] * df['Age']
df['Alcohol_Consumption'] = df['HvyAlcoholConsump'] * df['Age']

df['Health_Status_Index'] = (
    df['GenHlth'] * 0.4 +
    df['MentHlth'] * 0.3 +
    df['PhysHlth'] * 0.3
)

df['Lifestyle_Score'] = (
    df['PhysActivity'] * 0.4 +
    df['Fruits'] * 0.2 +
    df['Veggies'] * 0.2 -
    df['Smoker'] * 0.2
)

df['Healthcare_Access'] = (
    df['AnyHealthcare'] * 0.5 -
    df['NoDocbcCost'] * 0.5
)

df['Age_BP_Risk'] = df['Age'] * df['HighBP']
df['Age_Chol_Risk'] = df['Age'] * df['HighChol']


# Define outcome (risk indicator)

df['Outcome'] = (
    (df['HighBP'] == 1) |
    (df['HighChol'] == 1) |
    (df['Smoking'] == 1) |
    (df['HvyAlcoholConsump'] == 1) |
    (df['BMI'] >= 30)
).astype(int)


# Prepare features and target

X = df[
    [
        'patient_characteristics',
        'hypertension_highChol',
        'Smoking',
        'Alcohol_Consumption',
        'Health_Status_Index',
        'Lifestyle_Score',
        'Healthcare_Access',
        'Age_BP_Risk',
        'Age_Chol_Risk'
    ]
]

y = df['Outcome']


# Train / test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Handle class imbalance (SMOTE)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Random Forest Model

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train_resampled, y_train_resampled)

y_pred_rfc = rfc.predict(X_test)

print("\n=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rfc))
print("ROC-AUC:", roc_auc_score(y_test, rfc.predict_proba(X_test)[:, 1]))
print(classification_report(y_test, y_pred_rfc))

cv_rfc = cross_val_score(
    rfc, X_train_resampled, y_train_resampled,
    cv=5, scoring='accuracy'
)
print("RFC CV Accuracy:", cv_rfc)


# XGBoost Model

xgb = XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train_resampled, y_train_resampled)

y_pred_xgb = xgb.predict(X_test)

print("\n=== XGBoost Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]))
print(classification_report(y_test, y_pred_xgb))

cv_xgb = cross_val_score(
    xgb, X_train_resampled, y_train_resampled,
    cv=5, scoring='accuracy'
)
print("XGB CV Accuracy:", cv_xgb)
