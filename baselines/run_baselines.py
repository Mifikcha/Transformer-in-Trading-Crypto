"""
Orchestrator: run all 5 baseline models and print a summary table.
Run from project root: python baselines/run_baselines.py
"""

import os
import sys
import time

# Ensure baselines dir is on path when running as script
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

import numpy as np
from logistic_regression_baseline import run as run_logistic
from linear_regression_baseline import run as run_linear
from xgboost_baseline import run as run_xgboost
from lightgbm_baseline import run as run_lightgbm
from catboost_baseline import run as run_catboost


def _mean_metric(metrics_per_fold: list[dict], key: str) -> float:
    vals = [m[key] for m in metrics_per_fold if key in m and isinstance(m[key], (int, float)) and not (isinstance(m[key], float) and np.isnan(m[key]))]
    return sum(vals) / len(vals) if vals else float("nan")


def main(data_path: str | None = None, n_splits: int = 5) -> None:
    runners = [
        ("Logistic Regression", run_logistic),
        ("Linear Regression (Ridge)", run_linear),
        ("XGBoost", run_xgboost),
        ("LightGBM", run_lightgbm),
        ("CatBoost", run_catboost),
    ]
    results = []
    for name, run_fn in runners:
        start = time.perf_counter()
        metrics_per_fold = run_fn(data_path=data_path, n_splits=n_splits)
        elapsed = time.perf_counter() - start
        results.append({
            "model": name,
            "balanced_accuracy": _mean_metric(metrics_per_fold, "balanced_accuracy"),
            "macro_f1": _mean_metric(metrics_per_fold, "macro_f1"),
            "coverage": _mean_metric(metrics_per_fold, "coverage"),
            "pnl_proxy": _mean_metric(metrics_per_fold, "pnl_proxy"),
            "time_sec": round(elapsed, 2),
        })

    print("\n" + "=" * 80)
    print("  BASELINE SUMMARY")
    print("=" * 80)
    print(f"  {'Model':<28}  {'Bal.Acc':>8}  {'Macro F1':>8}  {'Coverage':>8}  {'PnL-proxy':>10}  {'Time(s)':>8}")
    print("-" * 80)
    for r in results:
        print(f"  {r['model']:<28}  {r['balanced_accuracy']:>8.4f}  {r['macro_f1']:>8.4f}  {r['coverage']:>8.4f}  {r['pnl_proxy']:>10.4f}  {r['time_sec']:>8.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
