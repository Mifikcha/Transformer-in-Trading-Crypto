"""
Feature importance (gain) from LightGBM over walk-forward folds.
"""

from __future__ import annotations

import io
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_SCRIPT_DIR)
_BASELINES = os.path.join(_BASE, "baselines")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _BASELINES not in sys.path:
    sys.path.insert(0, _BASELINES)

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from utils import (
    load_dataset,
    get_feature_columns,
    walk_forward_split,
    encode_trading_class,
    TRADING_CLASS_COL,
    TARGET_RETURN_COL,
    get_default_data_path,
    get_lightgbm_classifier_config,
)
from feature_groups import get_group_for_feature


def _out(msg: str, log_file: io.TextIOWrapper | None) -> None:
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def run(
    data_path: str | None = None,
    n_splits: int = 5,
    log_file: io.TextIOWrapper | None = None,
) -> pd.DataFrame:
    """Train LightGBM per fold, collect feature_importances_ (gain), aggregate and return ranking."""
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].astype(float)
    X = X.fillna(0)
    y, _ = encode_trading_class(df[TRADING_CLASS_COL])

    splits = walk_forward_split(df, n_splits=n_splits)
    importances_per_fold: list[dict[str, float]] = []

    for train_idx, test_idx in splits:
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = lgb.LGBMClassifier(**get_lightgbm_classifier_config())
        model.fit(X_train_s, y_train)
        imp = model.feature_importances_
        importances_per_fold.append(dict(zip(feat_cols, imp)))

    # Aggregate: mean and std per feature
    rows = []
    for col in feat_cols:
        vals = [d[col] for d in importances_per_fold]
        rows.append({
            "feature": col,
            "group": get_group_for_feature(col),
            "mean_importance": np.mean(vals),
            "std_importance": np.std(vals) if len(vals) > 1 else 0.0,
        })
    result = pd.DataFrame(rows)
    result = result.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)

    # Console and log: top-30 and sum by group
    _out("\n" + "=" * 70, log_file)
    _out("  BUILTIN IMPORTANCE (LightGBM gain)", log_file)
    _out("=" * 70, log_file)
    top = result.head(30)
    _out("\n  Top 30 features:", log_file)
    for _, r in top.iterrows():
        _out(f"    {r['rank']:3d}  {r['feature']:<30}  {r['mean_importance']:>10.2f}  (+- {r['std_importance']:.2f})  [{r['group']}]", log_file)
    group_sum = result.groupby("group", sort=False)["mean_importance"].sum().sort_values(ascending=False)
    _out("\n  Sum of importance by group:", log_file)
    for grp, s in group_sum.items():
        _out(f"    {grp:<20}  {s:>10.2f}", log_file)
    _out("=" * 70 + "\n", log_file)

    return result
