<div align="center">

# 💳 Credit Risk Scoring System

### End-to-End Machine Learning Pipeline for Loan Default Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-29B5E8?style=for-the-badge)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Predicts the **Probability of Default (PD)** for loan applicants using the LendingClub dataset, with full SHAP explainability and a real-time Streamlit interface.*

</div>

---

## 📸 Preview

> Dark-mode Streamlit UI with live prediction, risk categorisation, and SHAP feature attribution chart.

```
╔══════════════════════════════════════════════════════════════╗
║  💳 CreditRisk AI            Credit Risk Scoring System      ║
║  ─────────────────     ─────────────────────────────────     ║
║  📋 Loan Details         Probability of Default: 26.0%       ║
║  📋 Borrower Profile     Risk Category: LOW RISK             ║
║  📊 Credit History       Recommendation: ✅ Approve          ║
║  [🔍 Predict Risk]       ──────────────────────────────      ║
║                          SHAP Feature Contributions chart     ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Getting Started](#-getting-started)
- [Running the Pipeline](#-running-the-pipeline)
- [Streamlit App](#-streamlit-app)
- [Model Performance](#-model-performance)
- [How It Works](#-how-it-works)
- [Key Design Decisions](#-key-design-decisions)
- [Known Behaviour](#-known-behaviour)

---

## 🎯 Overview

This project is a **production-ready Credit Risk Scoring System** built on the LendingClub peer-to-peer lending dataset (2.26 million loan records). It predicts whether a borrower will **default on their loan** and explains *why* using SHAP values.

**What it does:**
- Loads and preprocesses 37 origination-time features (no post-loan data leakage)
- Handles severe class imbalance (~21% default rate) using SMOTE
- Trains an XGBoost classifier achieving **ROC-AUC ≥ 0.72**
- Generates 10 diagnostic plots (EDA, confusion matrix, SHAP)
- Serves real-time predictions via an interactive Streamlit web app
- Explains every single prediction with a SHAP waterfall chart

---

## 🏗️ Architecture

```
loan.csv (1.1 GB)
      │
      ▼
┌─────────────────────┐
│  1. Data Ingestion  │  ← 20% stratified sample, binary target encoding
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Feature Eng.    │  ← loan_to_income, installment_to_income,
└──────────┬──────────┘     credit_history_years
           │
           ▼
┌─────────────────────┐        ┌──────────────────────┐
│  3. Preprocessing   │ ──────▶│  preprocessor.pkl    │
│  (sklearn Pipeline) │        └──────────────────────┘
└──────────┬──────────┘
           │ 80/20 stratified split
    ┌──────┴──────┐
    │             │
  Train          Test (held out)
    │
    ▼
┌─────────────────────┐
│  4. SMOTE           │  ← 45K minority → 167K (balanced)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐        ┌──────────────────────┐
│  5. XGBoost Train   │ ──────▶│  xgb_model.pkl       │
└──────────┬──────────┘        └──────────────────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────┐   ┌────────┐
│ Eval   │   │  SHAP  │
│Metrics │   │Explain │
└────────┘   └────────┘
                 │
                 ▼
        ┌────────────────┐
        │  Streamlit UI  │  ← Real-time predictions
        └────────────────┘
```

---

## 🛠️ Technology Stack

| Library | Version | Role |
|---|---|---|
| `pandas` | ≥ 1.5 | Data loading & manipulation |
| `numpy` | ≥ 1.23 | Numerical operations |
| `scikit-learn` | ≥ 1.2 | Preprocessing pipeline, metrics |
| `xgboost` | ≥ 1.7 | Primary classifier |
| `imbalanced-learn` | ≥ 0.10 | SMOTE oversampling |
| `shap` | ≥ 0.42 | Model explainability |
| `streamlit` | ≥ 1.25 | Interactive prediction UI |
| `matplotlib` | ≥ 3.6 | Visualisations |
| `seaborn` | ≥ 0.12 | Statistical plots |
| `joblib` | ≥ 1.2 | Model serialisation |

---

## 📁 Project Structure

```
Credit Risk Scoring System/
│
├── modules/                        # Modular pipeline components
│   ├── __init__.py
│   ├── data_ingestion.py           # Module 1: Load & sample dataset
│   ├── preprocessing.py            # Module 2: sklearn Pipeline
│   ├── eda.py                      # Module 3: EDA visualisations
│   ├── feature_engineering.py      # Module 4: Derived features
│   ├── imbalance.py                # Module 5: SMOTE
│   ├── model.py                    # Module 6: XGBoost training
│   ├── evaluation.py               # Module 7: Metrics & plots
│   └── explainability.py           # Module 8: SHAP analysis
│
├── data/                           # ← Place loan.csv here (not tracked)
│   └── .gitkeep
│
├── models/                         # ← Auto-created after training
│   └── .gitkeep                    #   xgb_model.pkl, preprocessor.pkl
│
├── outputs/eda/                    # ← Auto-created after training
│   └── .gitkeep                    #   10 diagnostic plots
│
├── app.py                          # Streamlit web application
├── main.py                         # Pipeline runner (CLI)
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

This project uses the **LendingClub Loan Data** dataset from Kaggle.

🔗 **Download here:** [https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv)

| Property | Value |
|---|---|
| File | `loan.csv` |
| Size | ~1.1 GB |
| Rows | 2,260,668 |
| Columns | 145 |
| Target | `loan_status` → binary (default / non-default) |

**After downloading**, place the file in the `data/` directory:
```
data/
└── loan.csv          ← place here
```

> **Note:** `loan.csv` is excluded from git (`.gitignore`) due to its size. The data dictionary (`LCDataDictionary.xlsx`) is also available at the Kaggle link above.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- ~3 GB free disk space (dataset + virtual environment)

### 1. Clone the repository

```bash
git clone https://github.com/amaysaxena02/Credit-Risk-Scoring-System.git
cd "Credit Risk Scoring System"
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `loan.csv` from the [Kaggle link above](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv) and place it in the `data/` folder:

```
data/
└── loan.csv
```

---

## ⚙️ Running the Pipeline

The `main.py` script runs all 8 pipeline modules end-to-end.

### Basic run (20% sample — recommended for first run)

```bash
python main.py
```

### All CLI options

```bash
python main.py [OPTIONS]

Options:
  --sample-fraction FLOAT   Fraction of data to use (default: 0.2)
  --data-path PATH          Path to loan.csv (default: data/loan.csv)
  --skip-eda                Skip EDA plots (faster iteration)
  --skip-shap               Skip SHAP analysis (faster iteration)
```

### Examples

```bash
# Fast re-train (skip EDA and SHAP plots)
python main.py --sample-fraction 0.2 --skip-eda --skip-shap

# Full dataset run (takes 15–30 min)
python main.py --sample-fraction 1.0 --skip-eda

# Custom data path
python main.py --data-path "C:/datasets/loan.csv"
```

### What the pipeline produces

After a successful run you will find:

```
models/
├── xgb_model.pkl         ← Trained XGBoost classifier (1 MB)
└── preprocessor.pkl      ← Fitted sklearn preprocessing pipeline (7 KB)

outputs/eda/
├── 1_target_distribution.png
├── 2_loan_amount_dist.png
├── 3_annual_income_dist.png
├── 4_default_rate_by_grade.png
├── 5_correlation_heatmap.png
├── 6_purpose_breakdown.png
├── 7_confusion_matrix.png
├── 8_roc_curve.png
├── 9_shap_summary.png        ← SHAP beeswarm (global)
└── 10_shap_importance_bar.png
```

**Expected console output:**
```
============================================================
  CREDIT RISK SCORING SYSTEM — PIPELINE START
============================================================
[1/7] Data Ingestion ...      → 265,657 rows (20% sample)
[2/7] EDA ...                 → 6 plots saved
[3/7] Feature Engineering ... → 3 new features
[4/7] Preprocessing ...       → 38 features, pipeline saved
[5/7] SMOTE ...               → 334,060 balanced samples
[6/7] Training XGBoost ...    → 300 trees, max_depth=6
[7/7] Evaluation ...          → ROC-AUC: 0.7181
[+]   SHAP Analysis ...       → 2 plots saved
============================================================
  PIPELINE COMPLETE
  ROC-AUC  : 0.7181
  Precision: 0.5792
  Recall   : 0.1180
  F1 Score : 0.1960
============================================================
```

---

## 🌐 Streamlit App

### Launch the app

```bash
# Make sure the pipeline has been run first (models/ must exist)
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### How to use it

1. **Fill in the borrower profile** in the left sidebar:
   - Loan amount, term, interest rate, grade
   - Annual income, employment length, home ownership
   - Credit history details (DTI, revolving utilization, delinquencies, etc.)

2. **Click "🔍 Predict Default Risk"**

3. **Read the results:**

| Output | Description |
|---|---|
| **Probability of Default** | 0–100% gauge with colour fill |
| **Risk Category** | 🟢 Low (<30%) · 🟡 Medium (30–60%) · 🔴 High (>60%) |
| **Recommendation** | Approve / Review / Decline |
| **SHAP Chart** | Top 15 features driving this specific prediction (red = increases risk, green = decreases risk) |
| **Input Summary** | Collapsible table of all entered values |

> The Streamlit app uses the saved `models/xgb_model.pkl` and `models/preprocessor.pkl` — the **same** preprocessing applied during training is applied at inference time, ensuring no train/serve mismatch.

---

## 📈 Model Performance

Evaluated on a held-out 20% test set (53,132 rows, real-world class distribution):

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.7181** |
| Precision | 0.5792 |
| Recall | 0.1180 |
| F1 Score | 0.1960 |
| Accuracy | 0.79 |

**Training data breakdown:**

| Stage | Rows |
|---|---|
| Raw dataset | 2,260,668 |
| After status filter (resolved only) | 1,328,284 |
| After 20% stratified sample | 265,657 |
| Train split (80%) | 212,525 |
| After SMOTE balancing | 334,060 |
| Test split (20%, untouched) | 53,132 |

> **Why is recall low?** The model is trained on SMOTE-balanced data (50/50 split) but evaluated on the real-world test set (80% non-default / 20% default). **ROC-AUC is the correct primary metric** — it measures discriminative ability independent of the decision threshold.

---

## 🔍 How It Works

### 1. Data Ingestion (`modules/data_ingestion.py`)
- Reads only 37 origination-time columns from the 145-column CSV (no post-loan leakage)
- Maps `loan_status` to binary target: `Charged Off` / `Default` / `Late 31-120d` → **1**, `Fully Paid` → **0**. Ambiguous statuses (`Current`, `In Grace Period`) are dropped
- Applies a **stratified sample** to maintain the natural class ratio

### 2. Feature Engineering (`modules/feature_engineering.py`)
Three derived features are created before preprocessing:

| Feature | Formula |
|---|---|
| `loan_to_income` | `loan_amnt / annual_inc` |
| `installment_to_income` | `installment / (annual_inc / 12)` |
| `credit_history_years` | `2024 − earliest_cr_line year` |

### 3. Preprocessing (`modules/preprocessing.py`)
A **sklearn `Pipeline`** ensures identical transformations at training and inference time:
1. **`TextFeatureConverter`** (custom transformer) — converts string columns: `term` (`" 36 months"` → `36`), `emp_length` (`"10+ years"` → `10`), `int_rate`/`revol_util` (strip `%`)
2. **`ColumnTransformer`** — `SimpleImputer(median)` for numerics, `SimpleImputer(mode) + OrdinalEncoder` for categoricals
3. Saved as `models/preprocessor.pkl` — loaded by the Streamlit app at inference

### 4. SMOTE (`modules/imbalance.py`)
Applied **only on the training split** to prevent leakage. Oversamples the minority class (default) to achieve a 1:1 ratio.

### 5. XGBoost Training (`modules/model.py`)
```python
XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1.0,
    objective='binary:logistic', eval_metric='auc'
)
```

### 6. SHAP Explainability (`modules/explainability.py`)
- Uses `shap.TreeExplainer` (exact, fast for tree models)
- **Global:** beeswarm + bar plots computed on a 2,000-row subsample
- **Local:** per-prediction bar chart in the Streamlit app (real-time, single row)

---

## 🎛️ Key Design Decisions

| Decision | Rationale |
|---|---|
| **Origination-time features only** | Prevents post-loan data leakage (`total_pymnt`, `out_prncp`, etc. are excluded) |
| **sklearn Pipeline for preprocessing** | Same `preprocessor.pkl` is used at training and inference — no mismatch |
| **SMOTE on train split only** | Prevents synthetic minority samples from bleeding into the test set |
| **ROC-AUC as primary metric** | Robust to class imbalance; measures ranking ability at all thresholds |
| **20% sample default** | Balances speed (2–5 min) with statistical robustness (~265K rows) |
| **SHAP capped at 2,000 rows** | Keeps global plots fast without compromising accuracy |
| **XGBoost over deep learning** | Interpretable, fast, no GPU required, excellent on tabular data |

---

## ⚠️ Known Behaviour

- A harmless `dateutil` warning appears during feature engineering (pandas date parsing). Does not affect results.
- The `pipeline.log` file is created on every run and overwritten each time (excluded from git).
- Models and outputs are **not committed** to git. Re-run `python main.py` after cloning to regenerate them.

---

## 📄 License

This project is licensed under the MIT License. For demonstration and educational purposes only. **Not intended for actual lending decisions.**

---

<div align="center">
Built with ❤️ using XGBoost · SHAP · Streamlit · scikit-learn
</div>
