"""
XGBoost baseline: multi-class classification with walk-forward validation.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
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
)


def run(data_path: str | None = None, n_splits: int = 5) -> list[dict]:
    """Train XGBoost with walk-forward validation; print metrics; return metrics per fold."""
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].astype(float)
    X = X.fillna(0)
    y, le = encode_trading_class(df[TRADING_CLASS_COL])
    target_return = df[TARGET_RETURN_COL].astype(float).values

    splits = walk_forward_split(df, n_splits=n_splits)
    metrics_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        ret_test = target_return[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            max_depth=6,
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        model.fit(X_train_s, y_train, verbose=False)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)

        m = compute_metrics(
            y_test, y_pred, y_proba, ret_test, cost_per_trade=DEFAULT_COST_PER_TRADE
        )
        metrics_per_fold.append(m)

    print_metrics(metrics_per_fold, "XGBoost")
    return metrics_per_fold


if __name__ == "__main__":
    run()
