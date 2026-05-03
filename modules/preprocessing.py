"""
Module 2: Preprocessing
Builds a sklearn Pipeline for consistent train/inference preprocessing.
Saves and loads the fitted preprocessor as a .pkl artifact.
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Column definitions (post-feature-engineering, pre-preprocessing)
# ------------------------------------------------------------------
NUMERIC_FEATURES = [
    'loan_amnt', 'funded_amnt', 'int_rate', 'installment',
    'term',          # converted from string by TextFeatureConverter
    'emp_length',    # converted from string by TextFeatureConverter
    'annual_inc', 'dti',
    'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
    'collections_12_mths_ex_med', 'acc_now_delinq',
    'mort_acc', 'pub_rec_bankruptcies', 'pct_tl_nvr_dlq',
    'bc_util', 'avg_cur_bal', 'tot_cur_bal', 'tot_hi_cred_lim',
    'num_actv_rev_tl', 'num_bc_sats', 'total_rev_hi_lim',
    # Engineered
    'loan_to_income', 'installment_to_income', 'credit_history_years',
]

CATEGORICAL_FEATURES = [
    'grade', 'sub_grade', 'home_ownership',
    'verification_status', 'purpose', 'initial_list_status', 'addr_state',
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ------------------------------------------------------------------
# Custom transformer: converts string columns to numeric in-place
# ------------------------------------------------------------------
class TextFeatureConverter(BaseEstimator, TransformerMixin):
    """
    Converts text-encoded columns (term, emp_length, int_rate, revol_util)
    to float. Must run before the ColumnTransformer.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # term: " 36 months" → 36
        if 'term' in X.columns:
            X['term'] = (
                X['term'].astype(str)
                .str.extract(r'(\d+)', expand=False)
                .astype(float)
            )

        # emp_length: "10+ years" → 10, "< 1 year" → 0
        if 'emp_length' in X.columns:
            X['emp_length'] = (
                X['emp_length'].astype(str)
                .str.extract(r'(\d+)', expand=False)
                .astype(float)
            )

        # int_rate: may be "10.5%" or already float
        if 'int_rate' in X.columns:
            X['int_rate'] = pd.to_numeric(
                X['int_rate'].astype(str).str.replace('%', '', regex=False),
                errors='coerce'
            )

        # revol_util: may be "45.2%" or already float
        if 'revol_util' in X.columns:
            X['revol_util'] = pd.to_numeric(
                X['revol_util'].astype(str).str.replace('%', '', regex=False),
                errors='coerce'
            )

        return X


# ------------------------------------------------------------------
# Column selector helper
# ------------------------------------------------------------------
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects a specified list of columns and returns a DataFrame."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        present = [c for c in self.columns if c in X.columns]
        return X[present]


# ------------------------------------------------------------------
# Build the full preprocessing pipeline
# ------------------------------------------------------------------
def build_preprocessor() -> Pipeline:
    """
    Build the full preprocessing pipeline:
      1. TextFeatureConverter  — string → float for term/emp_length/rates
      2. ColumnTransformer     — impute + encode numeric and categorical cols
    """
    numeric_present = NUMERIC_FEATURES
    categorical_present = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_present),
            ('cat', categorical_transformer, categorical_present),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline([
        ('text_converter', TextFeatureConverter()),
        ('col_transformer', col_transformer),
    ])

    return pipeline


def get_feature_names(preprocessor: Pipeline) -> list:
    """Return feature names in the order the preprocessor outputs them."""
    col_transformer = preprocessor.named_steps['col_transformer']
    return list(col_transformer.get_feature_names_out())


def save_preprocessor(preprocessor: Pipeline, path: str = 'models/preprocessor.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    logger.info(f"Preprocessor saved to {path}")


def load_preprocessor(path: str = 'models/preprocessor.pkl') -> Pipeline:
    preprocessor = joblib.load(path)
    logger.info(f"Preprocessor loaded from {path}")
    return preprocessor
