"""
Permutation importance on walk-forward validation windows (balanced_accuracy).
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer

from utils import (
    load_dataset,
    get_feature_columns,
    walk_forward_split,
    encode_trading_class,
    TRADING_CLASS_COL,
    get_default_data_path,
)
from feature_groups import get_group_for_feature


def _lgbm_config():
    return dict(
        objective="multiclass",
        num_class=3,
        max_depth=6,
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
        class_weight="balanced",
    )


def _out(msg: str, log_file: io.TextIOWrapper | None) -> None:
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def run(
    data_path: str | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    log_file: io.TextIOWrapper | None = None,
) -> pd.DataFrame:
    """Compute permutation importance per fold, aggregate mean/std, return ranking."""
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].astype(float)
    X = X.fillna(0)
    y, _ = encode_trading_class(df[TRADING_CLASS_COL])

    scorer = get_scorer("balanced_accuracy")
    splits = walk_forward_split(df, n_splits=n_splits)
    perm_importances: list[dict[str, float]] = []

    for train_idx, test_idx in splits:
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = lgb.LGBMClassifier(**_lgbm_config())
        model.fit(X_train_s, y_train)

        # importance_decreases when feature is permuted (lower score = more important)
        pi = permutation_importance(
            model,
            X_test_s,
            y_test,
            n_repeats=n_repeats,
            random_state=42,
            scoring=scorer,
        )
        perm_importances.append(dict(zip(feat_cols, pi.importances_mean)))

    rows = []
    for col in feat_cols:
        vals = [d[col] for d in perm_importances]
        rows.append({
            "feature": col,
            "group": get_group_for_feature(col),
            "mean_perm_importance": np.mean(vals),
            "std_perm_importance": np.std(vals) if len(vals) > 1 else 0.0,
        })
    result = pd.DataFrame(rows)
    result = result.sort_values("mean_perm_importance", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)

    _out("\n" + "=" * 70, log_file)
    _out("  PERMUTATION IMPORTANCE (balanced_accuracy drop)", log_file)
    _out("=" * 70, log_file)
    top = result.head(30)
    _out("\n  Top 30 features:", log_file)
    for _, r in top.iterrows():
        _out(f"    {r['rank']:3d}  {r['feature']:<30}  {r['mean_perm_importance']:>10.4f}  (+- {r['std_perm_importance']:.4f})  [{r['group']}]", log_file)
    group_sum = result.groupby("group", sort=False)["mean_perm_importance"].sum().sort_values(ascending=False)
    _out("\n  Sum of permutation importance by group:", log_file)
    for grp, s in group_sum.items():
        _out(f"    {grp:<20}  {s:>10.4f}", log_file)
    _out("=" * 70 + "\n", log_file)

    return result
