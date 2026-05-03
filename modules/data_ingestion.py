"""
Module 1: Data Ingestion
Loads the LendingClub dataset, filters to resolved loan statuses,
encodes the binary target, and applies a stratified sample.
"""

import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# Origination-time columns only (no post-loan data leakage)
COLS_TO_LOAD = [
    'loan_amnt', 'funded_amnt', 'term', 'int_rate', 'installment',
    'grade', 'sub_grade', 'emp_length', 'home_ownership',
    'annual_inc', 'verification_status', 'loan_status', 'purpose',
    'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line',
    'inq_last_6mths', 'mths_since_last_delinq',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
    'total_acc', 'initial_list_status',
    'collections_12_mths_ex_med', 'acc_now_delinq',
    # Additional credit bureau features (available at origination)
    'mort_acc', 'pub_rec_bankruptcies', 'pct_tl_nvr_dlq',
    'bc_util', 'avg_cur_bal', 'tot_cur_bal', 'tot_hi_cred_lim',
    'num_actv_rev_tl', 'num_bc_sats', 'total_rev_hi_lim',
]

DEFAULT_STATUSES = [
    'Charged Off',
    'Default',
    'Late (31-120 days)',
    'Does not meet the credit policy. Status:Charged Off',
]

NON_DEFAULT_STATUSES = [
    'Fully Paid',
    'Does not meet the credit policy. Status:Fully Paid',
]


def load_data(filepath: str, sample_fraction: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    """
    Load and sample the LendingClub dataset.

    Args:
        filepath: Path to loan.csv
        sample_fraction: Fraction of data to sample (default 0.2)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with binary 'target' column (1=default, 0=non-default)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    logger.info(f"Reading CSV header from {filepath} ...")
    header = pd.read_csv(filepath, nrows=0, low_memory=False)
    available = set(header.columns.tolist())
    cols_to_use = [c for c in COLS_TO_LOAD if c in available]
    missing = set(COLS_TO_LOAD) - set(cols_to_use)
    if missing:
        logger.warning(f"Columns not found in dataset (will skip): {missing}")

    logger.info(f"Loading {len(cols_to_use)} columns from dataset ...")
    df = pd.read_csv(filepath, usecols=cols_to_use, low_memory=False)
    logger.info(f"Raw shape: {df.shape}")

    # Drop rows with null loan_status
    df = df.dropna(subset=['loan_status'])

    # Encode target and filter to resolved statuses
    df = _encode_target(df)

    logger.info(f"After target encoding: {df.shape}")
    logger.info(f"Class counts:\n{df['target'].value_counts().to_string()}")

    # Stratified sample
    if sample_fraction < 1.0:
        sampled_idx = (
            df.groupby('target', group_keys=False)
            .apply(lambda x: x.sample(frac=sample_fraction, random_state=random_state))
            .index
        )
        df = df.loc[sampled_idx].reset_index(drop=True)
        logger.info(f"After {sample_fraction*100:.0f}% stratified sample: {df.shape}")
        logger.info(f"Sampled class counts:\n{df['target'].value_counts().to_string()}")

    return df


def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map loan_status to binary: 1=default, 0=non-default. Drop ambiguous rows."""
    all_valid = DEFAULT_STATUSES + NON_DEFAULT_STATUSES
    df = df[df['loan_status'].isin(all_valid)].copy()
    df['target'] = df['loan_status'].apply(lambda x: 1 if x in DEFAULT_STATUSES else 0)
    df.drop(columns=['loan_status'], inplace=True)
    return df
