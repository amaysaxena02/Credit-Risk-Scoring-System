"""
Module 8: Explainability (SHAP)
Uses TreeExplainer for global feature importance and single-prediction explanations.
SHAP computations are capped at 2,000 rows for performance.
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

logger = logging.getLogger(__name__)

MAX_SHAP_ROWS = 2000


def run_shap_analysis(model, X_train: np.ndarray, feature_names: list,
                      output_dir: str = 'outputs/eda') -> np.ndarray:
    """
    Compute SHAP values and save global summary plot.

    Args:
        model: Trained XGBoost model
        X_train: Processed training data (numpy array)
        feature_names: List of feature names matching X_train columns
        output_dir: Directory to save SHAP plots

    Returns:
        shap_values array
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('dark_background')

    # Subsample for performance
    n = min(MAX_SHAP_ROWS, X_train.shape[0])
    idx = np.random.choice(X_train.shape[0], n, replace=False)
    X_sample = X_train[idx]

    logger.info(f"Computing SHAP values on {n} samples ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # --- Global Summary (Beeswarm) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_type='dot'
    )
    plt.title('SHAP Feature Importance (Global)', fontsize=13, pad=10)
    plt.tight_layout()
    path = os.path.join(output_dir, '9_shap_summary.png')
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    logger.info(f"SHAP summary plot saved: {path}")

    # --- Bar importance plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_type='bar'
    )
    plt.title('SHAP Mean |SHAP Value| (Feature Importance)', fontsize=13, pad=10)
    plt.tight_layout()
    path2 = os.path.join(output_dir, '10_shap_importance_bar.png')
    plt.savefig(path2, dpi=120, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    logger.info(f"SHAP bar plot saved: {path2}")

    return shap_values, explainer


def explain_single_prediction(model, X_single: np.ndarray, feature_names: list,
                               explainer=None) -> dict:
    """
    Generate SHAP explanation for a single prediction.

    Args:
        model: Trained XGBoost model
        X_single: Single-row numpy array (1, n_features)
        feature_names: Feature names
        explainer: Pre-built TreeExplainer (created if None)

    Returns:
        dict with shap_values, base_value, expected_value, feature_names
    """
    if explainer is None:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_single)
    base_value = explainer.expected_value

    return {
        'shap_values': shap_values[0] if shap_values.ndim > 1 else shap_values,
        'base_value': float(base_value),
        'feature_names': feature_names,
        'feature_values': X_single[0] if X_single.ndim > 1 else X_single,
    }
