"""
main.py — Credit Risk Scoring System Pipeline Runner

Orchestrates all 8 modules end-to-end:
  1. Data ingestion (20% stratified sample)
  2. EDA visualizations
  3. Feature engineering
  4. Preprocessing (fit + save)
  5. Train/test split
  6. SMOTE oversampling
  7. XGBoost training + save
  8. Model evaluation
  9. SHAP analysis

Usage:
    python main.py              # Full pipeline
    python main.py --skip-eda  # Skip EDA for faster iteration
"""

import argparse
import logging
import sys
import os

import numpy as np
from sklearn.model_selection import train_test_split

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from modules.data_ingestion import load_data
from modules.feature_engineering import engineer_features
from modules.preprocessing import build_preprocessor, get_feature_names, save_preprocessor
from modules.eda import run_eda
from modules.imbalance import apply_smote
from modules.model import train_model, save_model
from modules.evaluation import evaluate_model
from modules.explainability import run_shap_analysis

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log', mode='w'),
    ],
)
logger = logging.getLogger('main')


def parse_args():
    parser = argparse.ArgumentParser(description='Credit Risk Scoring Pipeline')
    parser.add_argument('--skip-eda', action='store_true', help='Skip EDA plots')
    parser.add_argument('--skip-shap', action='store_true', help='Skip SHAP analysis')
    parser.add_argument('--sample-fraction', type=float, default=0.2,
                        help='Fraction of dataset to use (default: 0.2)')
    parser.add_argument('--data-path', type=str, default='data/loan.csv',
                        help='Path to loan.csv')
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  CREDIT RISK SCORING SYSTEM — PIPELINE START")
    logger.info("=" * 60)

    # ── 1. Data Ingestion ─────────────────────────────────────────
    logger.info("\n[1/7] Data Ingestion ...")
    df = load_data(args.data_path, sample_fraction=args.sample_fraction)

    # ── 2. EDA ────────────────────────────────────────────────────
    if not args.skip_eda:
        logger.info("\n[2/7] Exploratory Data Analysis ...")
        run_eda(df, output_dir='outputs/eda')
    else:
        logger.info("\n[2/7] EDA skipped (--skip-eda flag).")

    # ── 3. Feature Engineering ────────────────────────────────────
    logger.info("\n[3/7] Feature Engineering ...")
    df = engineer_features(df)

    # ── 4. Preprocessing ──────────────────────────────────────────
    logger.info("\n[4/7] Preprocessing ...")
    X = df.drop(columns=['target'])
    y = df['target'].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"Train: {X_train_raw.shape[0]:,} | Test: {X_test_raw.shape[0]:,}")

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    feature_names = get_feature_names(preprocessor)
    logger.info(f"Preprocessed feature count: {len(feature_names)}")

    save_preprocessor(preprocessor, path='models/preprocessor.pkl')

    # ── 5. SMOTE ──────────────────────────────────────────────────
    logger.info("\n[5/7] Applying SMOTE ...")
    X_train_bal, y_train_bal = apply_smote(X_train_proc, y_train)

    # ── 6. Model Training ─────────────────────────────────────────
    logger.info("\n[6/7] Training XGBoost ...")
    model = train_model(X_train_bal, y_train_bal)
    save_model(model, path='models/xgb_model.pkl')

    # ── 7. Evaluation ─────────────────────────────────────────────
    logger.info("\n[7/7] Model Evaluation ...")
    metrics = evaluate_model(model, X_test_proc, y_test, output_dir='outputs/eda')

    # ── 8. SHAP ───────────────────────────────────────────────────
    if not args.skip_shap:
        logger.info("\n[+] SHAP Analysis ...")
        run_shap_analysis(model, X_train_proc, feature_names, output_dir='outputs/eda')
    else:
        logger.info("\n[+] SHAP skipped (--skip-shap flag).")

    # ── Summary ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall   : {metrics['recall']:.4f}")
    logger.info(f"  F1 Score : {metrics['f1']:.4f}")

    if metrics['roc_auc'] < 0.70:
        logger.warning("WARNING: ROC-AUC below 0.70 threshold! Check data and features.")
    else:
        logger.info("SUCCESS: ROC-AUC meets the >= 0.70 success criterion.")

    logger.info("\nArtifacts saved:")
    logger.info("  models/xgb_model.pkl")
    logger.info("  models/preprocessor.pkl")
    logger.info("  outputs/eda/*.png")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
