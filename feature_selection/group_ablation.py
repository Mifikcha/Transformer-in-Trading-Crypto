"""
Group ablation: remove each feature family, retrain LightGBM, report metric deltas.
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
    compute_metrics,
    get_default_data_path,
    DEFAULT_COST_PER_TRADE,
    get_lightgbm_classifier_config,
)
from feature_groups import FEATURE_GROUPS, get_group_for_feature


def _run_with_features(
    X: pd.DataFrame,
    y: np.ndarray,
    target_return: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    feat_cols: list[str],
) -> dict[str, float]:
    """Train LightGBM per fold on X[feat_cols], return mean metrics over folds."""
    metrics_per_fold: list[dict] = []
    X_sub = X[feat_cols].astype(float).fillna(0)
    for train_idx, test_idx in splits:
        X_train = X_sub.iloc[train_idx]
        X_test = X_sub.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        ret_test = target_return[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = lgb.LGBMClassifier(**get_lightgbm_classifier_config())
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)
        m = compute_metrics(
            y_test, y_pred, y_proba, ret_test, cost_per_trade=DEFAULT_COST_PER_TRADE
        )
        metrics_per_fold.append(m)

    keys = ["balanced_accuracy", "macro_f1", "coverage", "pnl_proxy"]
    return {k: float(np.mean([f[k] for f in metrics_per_fold])) for k in keys}


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
    """Baseline (all features) then ablate each group; return table of deltas."""
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].astype(float).fillna(0)
    y, _ = encode_trading_class(df[TRADING_CLASS_COL])
    target_return = df[TARGET_RETURN_COL].astype(float).values

    splits = walk_forward_split(df, n_splits=n_splits)

    baseline_metrics = _run_with_features(X, y, target_return, splits, feat_cols)

    rows = []
    for group_name in FEATURE_GROUPS:
        group_cols = set(FEATURE_GROUPS[group_name])
        ablated_cols = [c for c in feat_cols if c not in group_cols]
        if not ablated_cols:
            continue
        ablated_metrics = _run_with_features(X, y, target_return, splits, ablated_cols)
        row = {
            "group": group_name,
            "baseline_bal_acc": baseline_metrics["balanced_accuracy"],
            "ablated_bal_acc": ablated_metrics["balanced_accuracy"],
            "delta_bal_acc": baseline_metrics["balanced_accuracy"] - ablated_metrics["balanced_accuracy"],
            "baseline_macro_f1": baseline_metrics["macro_f1"],
            "ablated_macro_f1": ablated_metrics["macro_f1"],
            "delta_macro_f1": baseline_metrics["macro_f1"] - ablated_metrics["macro_f1"],
            "baseline_coverage": baseline_metrics["coverage"],
            "ablated_coverage": ablated_metrics["coverage"],
            "delta_coverage": baseline_metrics["coverage"] - ablated_metrics["coverage"],
            "baseline_pnl_proxy": baseline_metrics["pnl_proxy"],
            "ablated_pnl_proxy": ablated_metrics["pnl_proxy"],
            "delta_pnl_proxy": baseline_metrics["pnl_proxy"] - ablated_metrics["pnl_proxy"],
        }
        rows.append(row)

    result = pd.DataFrame(rows)

    _out("\n" + "=" * 90, log_file)
    _out("  GROUP ABLATION (LightGBM)", log_file)
    _out("=" * 90, log_file)
    _out(f"  Baseline (all features):  bal_acc={baseline_metrics['balanced_accuracy']:.4f}  macro_f1={baseline_metrics['macro_f1']:.4f}  coverage={baseline_metrics['coverage']:.4f}  pnl_proxy={baseline_metrics['pnl_proxy']:.4f}", log_file)
    _out("-" * 90, log_file)
    _out(f"  {'Group':<18}  {'bal_acc':>10}  {'delta_ba':>10}  {'macro_f1':>10}  {'delta_f1':>10}  {'coverage':>10}  {'delta_cov':>10}  {'pnl_proxy':>10}  {'delta_pnl':>10}", log_file)
    _out("-" * 90, log_file)
    for _, r in result.iterrows():
        _out(f"  {r['group']:<18}  {r['ablated_bal_acc']:>10.4f}  {r['delta_bal_acc']:>+10.4f}  {r['ablated_macro_f1']:>10.4f}  {r['delta_macro_f1']:>+10.4f}  {r['ablated_coverage']:>10.4f}  {r['delta_coverage']:>+10.4f}  {r['ablated_pnl_proxy']:>10.4f}  {r['delta_pnl_proxy']:>+10.4f}", log_file)
    _out("=" * 90 + "\n", log_file)

    return result
