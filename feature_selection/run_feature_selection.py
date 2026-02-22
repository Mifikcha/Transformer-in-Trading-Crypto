"""
Feature selection pipeline: builtin importance -> permutation importance -> group ablation -> summary.

Run from project root:
  python feature_selection/run_feature_selection.py
"""

from __future__ import annotations

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

import builtin_importance
import permutation_importance
import group_ablation
from feature_groups import FEATURE_GROUPS


def _normalized_rank_ascending(series: pd.Series) -> pd.Series:
    """Rank 1 = best (highest value). Return rank in [0,1] with 0 = best."""
    r = series.rank(ascending=False, method="average")
    n = r.dropna().count()
    if n <= 0:
        return r
    return (r - 1) / (n - 1) if n > 1 else pd.Series(0.0, index=series.index)


def main(data_path: str | None = None, n_splits: int = 5, n_repeats: int = 5) -> None:
    print("\n" + "=" * 80)
    print("  FEATURE SELECTION PIPELINE")
    print("=" * 80)

    print("\nStep 1: Builtin importance (LightGBM gain)")
    df_builtin = builtin_importance.run(data_path=data_path, n_splits=n_splits)

    print("\nStep 2: Permutation importance")
    df_perm = permutation_importance.run(
        data_path=data_path, n_splits=n_splits, n_repeats=n_repeats
    )

    print("\nStep 3: Group ablation")
    df_ablation = group_ablation.run(data_path=data_path, n_splits=n_splits)

    # --- SUMMARY ---
    # Combined feature rank: average of normalized ranks (lower = more important)
    # Builtin: higher mean_importance = better -> rank 1 = best
    # Perm: higher mean_perm_importance = better -> rank 1 = best
    by_feature = df_builtin.set_index("feature")["mean_importance"].reindex(
        df_perm["feature"]
    )
    by_feature = by_feature.fillna(0)
    rank_builtin = _normalized_rank_ascending(by_feature)
    rank_perm = _normalized_rank_ascending(df_perm.set_index("feature")["mean_perm_importance"])
    common = rank_builtin.index.intersection(rank_perm.index)
    rank_combined = (rank_builtin.reindex(common).fillna(0.5) + rank_perm.reindex(common).fillna(0.5)) / 2
    rank_combined = rank_combined.sort_values()
    rank_combined = rank_combined.dropna()
    # Final rank 1 = best (lowest combined normalized rank)
    result_rank = pd.DataFrame({
        "feature": rank_combined.index,
        "combined_rank_norm": rank_combined.values,
    })
    result_rank["rank"] = np.arange(1, len(result_rank) + 1, dtype=int)

    # Recommended compact set: top features by cumulative builtin importance until 90%
    total_imp = df_builtin["mean_importance"].sum()
    df_builtin_sorted = df_builtin.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    cumsum = df_builtin_sorted["mean_importance"].cumsum()
    threshold = 0.9 * total_imp
    n_top = int((cumsum <= threshold).sum()) + 1
    n_top = min(n_top, len(df_builtin_sorted))
    recommended = df_builtin_sorted.head(n_top)["feature"].tolist()

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print("\n  Combined feature ranking (average of normalized builtin + permutation rank):")
    print("  Top 25:")
    for _, row in result_rank.head(25).iterrows():
        print(f"    {row['rank']:3.0f}  {row['feature']:<35}  (norm_rank={row['combined_rank_norm']:.4f})")
    print("\n  Group ranking by ablation (delta balanced_accuracy, higher = more important):")
    ablation_sorted = df_ablation.sort_values("delta_bal_acc", ascending=False)
    for _, r in ablation_sorted.iterrows():
        print(f"    {r['group']:<18}  delta_bal_acc = {r['delta_bal_acc']:+.4f}")
    print(f"\n  Recommended compact feature set (cumulative builtin importance >= 90%): {n_top} features")
    print("  " + ", ".join(recommended[:15]) + ("  ..." if len(recommended) > 15 else ""))
    if len(recommended) > 15:
        print("  " + ", ".join(recommended[15:]))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
