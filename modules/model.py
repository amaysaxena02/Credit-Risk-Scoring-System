"""
Module 6: Model Training
Trains an XGBoost classifier and saves it with joblib.
"""

import logging
import os
import joblib
import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)


def train_model(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> xgb.XGBClassifier:
    """
    Train an XGBoost binary classifier.

    Args:
        X_train: SMOTE-balanced training features (numpy array)
        y_train: Training labels
        random_state: Seed

    Returns:
        Fitted XGBClassifier
    """
    logger.info(f"Training XGBoost on {X_train.shape[0]:,} samples, {X_train.shape[1]} features ...")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, verbose=False)
    logger.info("XGBoost training complete.")
    return model


def save_model(model: xgb.XGBClassifier, path: str = 'models/xgb_model.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str = 'models/xgb_model.pkl') -> xgb.XGBClassifier:
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model
