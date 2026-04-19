"""
Nedbank Transaction Forecasting Challenge — Local Evaluation Script
===================================================================
Use this to score your predictions locally before submitting to Zindi.

Usage:
    python evaluate.py <submission.csv> <reference.csv>

Example:
    python evaluate.py my_submission.csv PublicReference.csv

The submission CSV must have columns: UniqueID, next_3m_txn_count
"""

import sys
import numpy as np
import pandas as pd


def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error."""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    if np.any(y_pred < 0):
        raise ValueError("Predictions must be non-negative for RMSLE.")
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <submission.csv> <reference.csv>")
        sys.exit(1)

    submission_path = sys.argv[1]
    reference_path = sys.argv[2]

    sub = pd.read_csv(submission_path)
    ref = pd.read_csv(reference_path)

    # Validate columns
    required_cols = {'UniqueID', 'next_3m_txn_count'}
    if not required_cols.issubset(sub.columns):
        print(f"ERROR: Submission must have columns: {required_cols}")
        print(f"Found: {set(sub.columns)}")
        sys.exit(1)

    if not required_cols.issubset(ref.columns):
        print(f"ERROR: Reference must have columns: {required_cols}")
        sys.exit(1)

    # Merge on UniqueID
    merged = ref.merge(sub, on='UniqueID', suffixes=('_true', '_pred'))

    if len(merged) != len(ref):
        missing = set(ref['UniqueID']) - set(sub['UniqueID'])
        print(f"ERROR: {len(missing)} UniqueIDs in reference not found in submission.")
        print(f"Expected {len(ref)} rows, matched {len(merged)}.")
        sys.exit(1)

    # Check for negative predictions
    if merged['next_3m_txn_count_pred'].min() < 0:
        print("ERROR: Predictions must be non-negative (RMSLE is undefined for negative values).")
        sys.exit(1)

    # Check for NaN predictions
    if merged['next_3m_txn_count_pred'].isna().any():
        n_nan = merged['next_3m_txn_count_pred'].isna().sum()
        print(f"ERROR: {n_nan} NaN values in predictions.")
        sys.exit(1)

    score = rmsle(merged['next_3m_txn_count_true'], merged['next_3m_txn_count_pred'])
    print(f"RMSLE: {score:.6f}")
    print(f"Rows scored: {len(merged)}")


if __name__ == '__main__':
    main()
