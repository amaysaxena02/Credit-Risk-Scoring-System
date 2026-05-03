"""
Module 7: Model Evaluation
Computes ROC-AUC, precision, recall, F1, and confusion matrix.
Saves plots to outputs/eda/.
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   output_dir: str = 'outputs/eda') -> dict:
    """
    Evaluate model performance and save plots.

    Returns:
        dict with roc_auc, precision, recall, f1
    """
    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info("=" * 50)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"  ROC-AUC  : {roc_auc:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall   : {recall:.4f}")
    logger.info(f"  F1 Score : {f1:.4f}")
    logger.info("=" * 50)
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

    _plot_confusion_matrix(y_test, y_pred, output_dir)
    _plot_roc_curve(y_test, y_proba, roc_auc, output_dir)

    return {'roc_auc': roc_auc, 'precision': precision, 'recall': recall, 'f1': f1}


def _plot_confusion_matrix(y_test, y_pred, out):
    plt.style.use('dark_background')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Default', 'Default'],
                yticklabels=['Non-Default', 'Default'],
                linewidths=0.5, linecolor='#0a0e1a')
    ax.set_title('Confusion Matrix', fontsize=13)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    path = os.path.join(out, '7_confusion_matrix.png')
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved: {path}")


def _plot_roc_curve(y_test, y_proba, roc_auc, out):
    plt.style.use('dark_background')
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#4FC3F7', lw=2, label=f'ROC AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.fill_between(fpr, tpr, alpha=0.15, color='#4FC3F7')
    ax.set_title('ROC Curve', fontsize=14)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(out, '8_roc_curve.png')
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved: {path}")
