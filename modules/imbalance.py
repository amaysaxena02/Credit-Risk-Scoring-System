"""
Module 5: Imbalanced Data Handling
Applies SMOTE to the training set only (after train/test split).
"""

import logging
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

logger = logging.getLogger(__name__)


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """
    Apply SMOTE to oversample the minority class.

    Args:
        X_train: Preprocessed training feature matrix (numpy array)
        y_train: Training labels
        random_state: Seed for reproducibility

    Returns:
        Tuple (X_resampled, y_resampled)
    """
    logger.info(f"Class distribution BEFORE SMOTE: {Counter(y_train)}")

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    logger.info(f"Class distribution AFTER SMOTE:  {Counter(y_res)}")
    logger.info(f"Training set size after SMOTE: {X_res.shape[0]:,} samples")

    return X_res, y_res
