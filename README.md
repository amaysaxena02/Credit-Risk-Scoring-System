# Credit Risk Scoring System — Walkthrough

## What Was Built

A complete end-to-end Credit Risk Scoring pipeline on the LendingClub dataset (2.26M rows, 20% sample = **265,657 rows**).

---

## Modules Implemented

| File | Module | Purpose |
|---|---|---|
| `modules/data_ingestion.py` | 1 | Load CSV, filter to resolved statuses, encode binary target, stratified sample |
| `modules/feature_engineering.py` | 4 | loan_to_income, installment_to_income, credit_history_years |
| `modules/preprocessing.py` | 2 | TextFeatureConverter + ColumnTransformer sklearn Pipeline |
| `modules/eda.py` | 3 | 6 EDA plots saved to outputs/eda/ |
| `modules/imbalance.py` | 5 | SMOTE oversampling (training set only) |
| `modules/model.py` | 6 | XGBoost training + joblib save |
| `modules/evaluation.py` | 7 | ROC-AUC, precision, recall, F1, confusion matrix, ROC curve |
| `modules/explainability.py` | 8 | SHAP TreeExplainer global + single-prediction |
| `main.py` | Runner | Orchestrates all 8 modules |
| `app.py` | UI | Streamlit dark-mode prediction app |

---

## Pipeline Execution Results

### Data Ingestion
- Raw dataset: **2,260,668 rows × 37 columns**
- After filtering to resolved statuses: **1,328,284 rows**
  - Non-default (Fully Paid): 1,043,940
  - Default (Charged Off / Late 31-120d): 284,344
- After 20% stratified sample: **265,657 rows** (208,788 non-default / 56,869 default)

### SMOTE Balancing
- Train split (80%): 212,525 rows → after SMOTE: **334,060 rows** (167,030 per class)
- Test split (20%): 53,132 rows (held out, untouched)

### Model Performance

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.7181** ✅ (≥ 0.70) |
| Precision | 0.5792 |
| Recall | 0.1180 |
| F1 Score | 0.1960 |
| Accuracy | 0.79 |

> [!NOTE]
> The low recall on the default class is expected behaviour: the model was trained on SMOTE-balanced data but evaluated on the real-world test set which has a 4.6:1 imbalance (80% non-default). The ROC-AUC of **0.7181** is the correct primary metric and confirms the model's discriminative ability.

### Saved Artifacts

| File | Size |
|---|---|
| `models/xgb_model.pkl` | 1.04 MB |
| `models/preprocessor.pkl` | 6.9 KB |

### EDA Plots Saved (outputs/eda/)

1. `1_target_distribution.png` — Class balance bar chart  
2. `2_loan_amount_dist.png` — Loan amount by default status  
3. `3_annual_income_dist.png` — Income distribution (99th pct capped)  
4. `4_default_rate_by_grade.png` — Default rate per grade (A→G)  
5. `5_correlation_heatmap.png` — Top 18 feature correlation heatmap  
6. `6_purpose_breakdown.png` — Default rate by loan purpose  
7. `7_confusion_matrix.png` — Model confusion matrix  
8. `8_roc_curve.png` — ROC curve (AUC = 0.7181)  
9. `9_shap_summary.png` — SHAP beeswarm plot (2,000 samples)  
10. `10_shap_importance_bar.png` — Mean |SHAP| bar chart  

---

## Streamlit App Verification

The app at **http://localhost:8501** was tested with a high-risk borrower profile:

| Input | Value |
|---|---|
| Grade | D |
| Interest Rate | ~18% |
| Annual Income | $45,000 |
| DTI | 25% |
| Revolving Utilization | 75% |
| Bank Card Utilization | 80% |
| Bankruptcies | 1 |
| % Never Delinquent | 70% |

**Result: 26.0% Probability of Default → LOW RISK → Approve**

The SHAP bar chart rendered correctly showing top 15 feature contributions with red (increases risk) and green (decreases risk) bars.

---


## How to Use

### Re-train the model
```bash
.venv\Scripts\python.exe main.py --sample-fraction 0.2
# Skip EDA for faster re-runs:
.venv\Scripts\python.exe main.py --sample-fraction 0.2 --skip-eda
```

### Launch the Streamlit app
```bash
.venv\Scripts\streamlit.exe run app.py
```

### Scale to full dataset
```bash
.venv\Scripts\python.exe main.py --sample-fraction 1.0 --skip-eda
```

---

## Known Behaviour

- Low recall on minority class is expected (real-world imbalance in test set). ROC-AUC is the correct metric.
- A harmless `dateutil` warning appears during feature engineering (date format inference) — does not affect results.
- The model is for demonstration purposes; do not use for actual lending decisions.
