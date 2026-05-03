"""
Module 4: Feature Engineering
Creates derived features from raw origination-time columns.
Must be applied consistently during both training and inference.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

CURRENT_YEAR = 2024


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the dataset.
    Operates on raw string/mixed columns (before preprocessing).

    Args:
        df: Raw DataFrame from data_ingestion

    Returns:
        DataFrame with additional engineered columns
    """
    df = df.copy()

    # --- Loan-to-Income Ratio ---
    annual_inc = pd.to_numeric(df['annual_inc'], errors='coerce').replace(0, np.nan)
    df['loan_to_income'] = pd.to_numeric(df['loan_amnt'], errors='coerce') / annual_inc

    # --- Installment-to-Monthly-Income Ratio ---
    monthly_inc = annual_inc / 12
    df['installment_to_income'] = pd.to_numeric(df['installment'], errors='coerce') / monthly_inc

    # --- Credit History Length (years) ---
    if 'earliest_cr_line' in df.columns:
        cr_year = pd.to_datetime(df['earliest_cr_line'], errors='coerce').dt.year
        df['credit_history_years'] = CURRENT_YEAR - cr_year
        df.drop(columns=['earliest_cr_line'], inplace=True)
    else:
        df['credit_history_years'] = np.nan

    # Clip extreme ratios to avoid inf values
    df['loan_to_income'] = df['loan_to_income'].clip(upper=50)
    df['installment_to_income'] = df['installment_to_income'].clip(upper=10)

    logger.info(
        "Feature engineering complete. New features: "
        "loan_to_income, installment_to_income, credit_history_years"
    )
    return df
