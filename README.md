# 📊 Churn Prediction System

A machine learning pipeline to predict customer churn using real-world telecom data. The project identifies at-risk customers and provides business insights through model evaluation, churn probability segmentation, and Power BI dashboards.

---

## 🔍 Project Overview

Churn prediction helps businesses retain customers by identifying those likely to leave. In this project, we use supervised classification models to analyze customer behavior and predict churn, then visualize results using Power BI.

---

## 🗂️ Dataset

**Source**: [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Size**: 7,043 records × 21 features  
**Target Variable**: `Churn` (Yes/No)

---

## ✅ Features

- Customer demographics (gender, senior citizen, dependents)
- Account info (tenure, contract type, payment method)
- Service usage (Internet service, streaming, online security)
- Monthly and total charges

---

## 🛠️ Technologies & Libraries

- **Python** – data processing & ML (`pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`)
- **Power BI** – dashboard and data storytelling
- **DAX** – measures for interactive insights
- **Jupyter / PyCharm** – development

---

## 📌 Phases

### Phase 1–2: Data Processing
- Cleaned missing and inconsistent values
- Encoded binary and categorical variables
- Engineered new feature: `MonthlyBucket` (Low, Medium, High)

### Phase 3: Modeling
- Trained 3 models: Logistic Regression, Random Forest, XGBoost
- Used `train_test_split`, `StandardScaler`, pipelines

### Phase 4: Evaluation & Visualization
- Accuracy, ROC-AUC, Confusion Matrix, Classification Report
- Visuals: ROC curves, heatmaps, feature importance, churn probability segmentation

### Phase 5: Dashboarding with Power BI
- Built a single-page dashboard:
  - Model selection slicer
  - Key metrics (Accuracy, ROC-AUC)
  - Churn risk counts
  - Risk-level bar charts
  - Feature importance

---

## 📁 Output Files

| File Name | Description |
|-----------|-------------|
| `final_predictions.csv` | Churn probabilities and risk levels for all models |
| `model_metrics.csv`     | Accuracy, ROC-AUC for each model |
| `ModelSelector` table   | Slicer support in Power BI |

---

## 📉 Key Insights

- Customers on **month-to-month contracts** and using **electronic check** are more likely to churn.
- Short tenure and **fiber optic internet** correlate strongly with churn.
- Most customers fall in the **Low** churn probability range, but **High-risk** customers account for ~10–15%.

---

## 📈 Dashboard Preview

> Power BI dashboard includes:
- Slicer to toggle between models
- Model performance cards
- Risk level comparison charts
- Feature importance visuals
- Confusion matrix and ROC curve

---

## 📚 Definitions of Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of total predictions correctly classified |
| **Precision** | Out of predicted churners, how many actually churned |
| **Recall** | Out of actual churners, how many were predicted |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Model’s ability to separate classes (0.5 = random, 1 = perfect) |

---

## 🚀 Future Improvements

- Add LightGBM / CatBoost for improved performance
- Build a Streamlit or Flask web app for business users
- Perform cost-based churn prioritization
- Incorporate customer sentiment features

---

## 🤝 Contact

**Author**: Muhamed Ibrahim  
Feel free to reach out for collaboration or feedback!

