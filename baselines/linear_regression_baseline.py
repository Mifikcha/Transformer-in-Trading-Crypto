"""
Linear (Ridge) Regression baseline: predict target_return, then convert to trading classes via cost threshold.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils import (
    load_dataset,
    get_feature_columns,
    walk_forward_split,
    encode_trading_class,
    TRADING_CLASS_COL,
    TARGET_RETURN_COL,
    compute_metrics,
    print_metrics,
    get_default_data_path,
    DEFAULT_COST_PER_TRADE,
    CLASS_ORDER,
)


def _return_to_class(pred_return: np.ndarray, cost_threshold: float) -> np.ndarray:
    """Map predicted return to class: long=2 if pred > c, short=1 if pred < -c, flat=0 else."""
    out = np.zeros(len(pred_return), dtype=int)
    out[pred_return > cost_threshold] = 2   # long
    out[pred_return < -cost_threshold] = 1   # short
    return out


def run(data_path: str | None = None, n_splits: int = 5, cost_threshold: float | None = None) -> list[dict]:
    """Train Ridge on target_return, convert predictions to classes; walk-forward; print metrics."""
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].astype(float)
    X = X.fillna(0)
    y, le = encode_trading_class(df[TRADING_CLASS_COL])
    target_return = df[TARGET_RETURN_COL].astype(float).values

    if cost_threshold is None:
        cost_threshold = DEFAULT_COST_PER_TRADE

    splits = walk_forward_split(df, n_splits=n_splits)
    metrics_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        ret_train = target_return[train_idx]
        ret_test = target_return[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_s, ret_train)
        pred_return = model.predict(X_test_s)
        y_pred = _return_to_class(pred_return, cost_threshold)

        # No probabilities for regression; build pseudo-proba from distance to threshold for ECE
        y_proba = np.column_stack([
            np.clip(-pred_return / (cost_threshold + 1e-8), 0, 1),   # flat-ish
            np.clip((-pred_return - cost_threshold) / (cost_threshold + 1e-8), 0, 1),  # short
            np.clip((pred_return - cost_threshold) / (cost_threshold + 1e-8), 0, 1),   # long
        ])
        y_proba = y_proba / (y_proba.sum(axis=1, keepdims=True) + 1e-8)

        m = compute_metrics(
            y_test, y_pred, y_proba, ret_test, cost_per_trade=cost_threshold
        )
        metrics_per_fold.append(m)

    print_metrics(metrics_per_fold, "Linear Regression (Ridge)")
    return metrics_per_fold


if __name__ == "__main__":
    run()
