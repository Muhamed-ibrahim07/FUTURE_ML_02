import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc

# ------------- PHASE 1-2: Load, clean, feature engineering -------------

df = pd.read_csv("Telco-Customer-Churn.csv")
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

binary_map = {'Yes': 1, 'No': 0}
binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_columns:
    df[col] = df[col].map(binary_map)

df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

no_service_cols = [
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
for col in no_service_cols:
    df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

df = pd.get_dummies(df, columns=[
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
], drop_first=True)

df['MonthlyBucket'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 120], labels=['Low', 'Medium', 'High'])
df = pd.get_dummies(df, columns=['MonthlyBucket'], drop_first=True)

# ------------- PHASE 3: Train models -------------

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000))
])

models = {
    "Logistic Regression": logistic_pipeline,
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

model_results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train, y_train)
        X_scaled = model.named_steps['scaler'].transform(X_test)
        y_pred = model.named_steps['clf'].predict(X_scaled)
        y_prob = model.named_steps['clf'].predict_proba(X_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    model_results[name] = {'y_pred': y_pred, 'y_prob': y_prob}

    print(f"\n📌 {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# ------------- PHASE 4: Visualizations -------------

# Confusion Matrix heatmaps
plt.figure(figsize=(15, 4))
for i, (name, result) in enumerate(model_results.items()):
    cm = confusion_matrix(y_test, result['y_pred'])
    plt.subplot(1, 3, i + 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curves
plt.figure(figsize=(7, 6))
for name, result in model_results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# Feature Importance Visuals

# Logistic Regression
coef_lr = pd.Series(logistic_pipeline.named_steps['clf'].coef_[0], index=X.columns)
plt.figure(figsize=(10, 5))
coef_lr.abs().sort_values(ascending=False).head(10).plot(kind='barh', title='Logistic Regression - Top Coefficients')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Random Forest
feat_rf = pd.Series(models['Random Forest'].feature_importances_, index=X.columns)
plt.figure(figsize=(10, 5))
feat_rf.sort_values(ascending=False).head(10).plot(kind='barh', title='Random Forest - Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# XGBoost
feat_xgb = pd.Series(models['XGBoost'].feature_importances_, index=X.columns)
plt.figure(figsize=(10, 5))
feat_xgb.sort_values(ascending=False).head(10).plot(kind='barh', title='XGBoost - Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Churn Probability Segmentation
plt.figure(figsize=(15, 4))
for i, (name, result) in enumerate(model_results.items()):
    risk_df = pd.DataFrame({
        'Churn_Prob': result['y_prob'],
        'Actual_Churn': y_test.reset_index(drop=True)
    })
    risk_df['Risk_Level'] = pd.cut(risk_df['Churn_Prob'], bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])
    counts = risk_df['Risk_Level'].value_counts().sort_index()

    plt.subplot(1, 3, i + 1)
    sns.barplot(x=counts.index, y=counts.values, palette='coolwarm')
    plt.title(f"{name} - Churn Risk Levels")
    plt.xlabel("Risk Level")
    plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# ------------- PHASE 5: Export all data into one Excel file -------------

X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

X_test_scaled = logistic_pipeline.named_steps['scaler'].transform(X_test)
log_y_prob = logistic_pipeline.named_steps['clf'].predict_proba(X_test_scaled)[:, 1]
rf_y_prob = models['Random Forest'].predict_proba(X_test)[:, 1]
xgb_y_prob = models['XGBoost'].predict_proba(X_test)[:, 1]

def risk_level(prob_series):
    return pd.cut(prob_series, bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])

df_test_combined = X_test_reset.copy()
df_test_combined['ActualChurn'] = y_test_reset
df_test_combined['PredictedChurnProb_Logistic'] = log_y_prob
df_test_combined['RiskLevel_Logistic'] = risk_level(log_y_prob)
df_test_combined['PredictedChurnProb_RF'] = rf_y_prob
df_test_combined['RiskLevel_RF'] = risk_level(rf_y_prob)
df_test_combined['PredictedChurnProb_XGB'] = xgb_y_prob
df_test_combined['RiskLevel_XGB'] = risk_level(xgb_y_prob)

metrics = []
for name, result in model_results.items():
    acc = accuracy_score(y_test, result['y_pred'])
    roc = roc_auc_score(y_test, result['y_prob'])
    metrics.append([name, acc, roc])

metrics_df = pd.DataFrame(metrics, columns=['Model', 'Accuracy', 'ROC_AUC'])

coef_lr_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(logistic_pipeline.named_steps['clf'].coef_[0])
}).sort_values(by='Importance', ascending=False)

feat_rf_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': models['Random Forest'].feature_importances_
}).sort_values(by='Importance', ascending=False)

feat_xgb_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': models['XGBoost'].feature_importances_
}).sort_values(by='Importance', ascending=False)

fpr_lr, tpr_lr, _ = roc_curve(y_test, log_y_prob)
roc_lr_df = pd.DataFrame({'FPR': fpr_lr, 'TPR': tpr_lr})

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_y_prob)
roc_rf_df = pd.DataFrame({'FPR': fpr_rf, 'TPR': tpr_rf})

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_y_prob)
roc_xgb_df = pd.DataFrame({'FPR': fpr_xgb, 'TPR': tpr_xgb})

with pd.ExcelWriter('churn_analysis_all_in_one.xlsx') as writer:
    df_test_combined.to_excel(writer, sheet_name='Test_Predictions', index=False)
    metrics_df.to_excel(writer, sheet_name='Model_Metrics', index=False)
    coef_lr_df.to_excel(writer, sheet_name='FeatImp_Logistic', index=False)
    feat_rf_df.to_excel(writer, sheet_name='FeatImp_RF', index=False)
    feat_xgb_df.to_excel(writer, sheet_name='FeatImp_XGB', index=False)
    roc_lr_df.to_excel(writer, sheet_name='ROC_Logistic', index=False)
    roc_rf_df.to_excel(writer, sheet_name='ROC_RF', index=False)
    roc_xgb_df.to_excel(writer, sheet_name='ROC_XGB', index=False)

print("\n✅ Exported all data to 'churn_analysis_all_in_one.xlsx' - ready for Power BI import!")
