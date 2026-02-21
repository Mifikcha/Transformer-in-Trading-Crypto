"""
Shared utilities for baseline models: data loading, feature columns,
walk-forward validation, metrics (balanced acc, macro-F1, coverage, ECE, PnL-proxy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Columns to exclude from features (targets, leakage, aux)
EXCLUDED_COLUMNS = {
    "ts",
    "target_return",
    "target_class",
    "is_valid_target",
    "future_close_30min",
    "future_close_60min",
    "future_close_120min",
    "future_close_240min",
    "delta_log_30min",
    "delta_log_60min",
    "delta_log_120min",
    "delta_log_240min",
    "base_regression",
    "base_class",
    "trading_class_optimistic",
    "trading_class_base",
    "trading_class_pessimistic",
}

TRADING_CLASS_COL = "trading_class_base"
TARGET_RETURN_COL = "target_return"
VALID_TARGET_COL = "is_valid_target"

CLASS_ORDER = ["flat", "short", "long"]  # consistent order for encoding
DEFAULT_COST_PER_TRADE = 0.001
ECE_N_BINS = 10


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV, filter is_valid_target==1, sort by ts."""
    df = pd.read_csv(path)
    df = df.loc[df[VALID_TARGET_COL].astype(int) == 1].copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of feature column names (exclude targets and leakage)."""
    return [c for c in df.columns if c not in EXCLUDED_COLUMNS and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]


def walk_forward_split(
    df: pd.DataFrame, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window walk-forward: for each fold, train on past chunks, test on next chunk.
    Returns list of (train_idx, test_idx) arrays.
    """
    n = len(df)
    if n < n_splits + 1:
        raise ValueError("Not enough rows for walk-forward splits")
    # n_splits+1 segments; fold k: train on segments 0..k, test on segment k+1
    segment_size = n // (n_splits + 1)
    splits = []
    for k in range(n_splits):
        train_end = (k + 1) * segment_size
        test_end = (k + 2) * segment_size if (k + 2) <= n_splits else n
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, min(test_end, n))
        if len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def encode_trading_class(series: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encode trading_class_base (flat/short/long) to 0, 1, 2. Returns (encoded, LabelEncoder)."""
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    encoded = le.transform(series.astype(str).str.strip().str.lower())
    return encoded, le


def decode_trading_class(encoded: np.ndarray, le: LabelEncoder) -> np.ndarray:
    """Decode 0,1,2 back to flat/short/long."""
    return le.inverse_transform(encoded.astype(int))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    target_return: np.ndarray,
    cost_per_trade: float = DEFAULT_COST_PER_TRADE,
    class_names: list[str] | None = None,
) -> dict:
    """
    Compute balanced accuracy, macro F1, precision/recall per class,
    coverage, ECE, PnL-proxy. y_true, y_pred: encoded (0=flat, 1=short, 2=long).
    """
    if class_names is None:
        class_names = CLASS_ORDER

    n = len(y_true)
    metrics = {}

    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    labels_ar = np.arange(len(class_names))
    prec_per = precision_score(y_true, y_pred, labels=labels_ar, average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred, labels=labels_ar, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        metrics[f"precision_{name}"] = prec_per[i] if i < len(prec_per) else 0.0
        metrics[f"recall_{name}"] = rec_per[i] if i < len(rec_per) else 0.0

    # Coverage: fraction where model says long or short (not flat). Flat = index 0.
    flat_idx = 0
    coverage = np.mean(y_pred != flat_idx)
    metrics["coverage"] = float(coverage)

    # ECE (Expected Calibration Error) if we have probabilities
    if y_proba is not None and y_proba.size > 0:
        metrics["ece"] = _expected_calibration_error(y_true, y_pred, y_proba, n_bins=ECE_N_BINS)
    else:
        metrics["ece"] = float("nan")

    # PnL-proxy: sign(pred) * target_return - cost when not flat
    # Encoding: 0=flat -> 0, 1=short -> -1, 2=long -> +1
    sign_pred = np.where(y_pred == 2, 1.0, np.where(y_pred == 1, -1.0, 0.0))
    cost = np.where(y_pred != flat_idx, cost_per_trade, 0.0)
    pnl_per_step = sign_pred * target_return - cost
    metrics["pnl_proxy"] = float(np.sum(pnl_per_step))

    return metrics


def _expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    """ECE for multi-class: bin by max predicted probability, compare accuracy vs confidence."""
    if y_proba.ndim == 1:
        conf = np.clip(np.abs(y_proba), 0, 1)
        acc_bin = (y_pred == y_true).astype(float)
    else:
        conf = np.max(y_proba, axis=1)
        acc_bin = (y_pred == y_true).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = 0
    for i in range(n_bins):
        in_bin = (conf >= bin_edges[i]) & (conf < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = (conf >= bin_edges[i]) & (conf <= bin_edges[i + 1])
        n_bin = in_bin.sum()
        if n_bin == 0:
            continue
        avg_confidence = conf[in_bin].mean()
        avg_accuracy = acc_bin[in_bin].mean()
        ece += n_bin * np.abs(avg_accuracy - avg_confidence)
        total += n_bin
    return float(ece / total) if total > 0 else 0.0


def print_metrics(metrics_per_fold: list[dict], model_name: str) -> None:
    """Print metrics aggregated over folds (mean +- std) to console."""
    if not metrics_per_fold:
        print(f"[{model_name}] No folds.")
        return

    keys = list(metrics_per_fold[0].keys())
    means = {}
    stds = {}
    for k in keys:
        vals = [m[k] for m in metrics_per_fold if isinstance(m.get(k), (int, float)) and not (isinstance(m.get(k), float) and np.isnan(m[k]))]
        if vals:
            means[k] = np.mean(vals)
            stds[k] = np.std(vals) if len(vals) > 1 else 0.0
        else:
            means[k] = np.nan
            stds[k] = np.nan

    print("\n" + "=" * 60)
    print(f"  {model_name}")
    print("=" * 60)
    print(f"  Balanced Accuracy:  {means.get('balanced_accuracy', np.nan):.4f}  (+- {stds.get('balanced_accuracy', 0):.4f})")
    print(f"  Macro F1:          {means.get('macro_f1', np.nan):.4f}  (+- {stds.get('macro_f1', 0):.4f})")
    print(f"  Coverage:           {means.get('coverage', np.nan):.4f}  (+- {stds.get('coverage', 0):.4f})")
    print(f"  ECE:                {means.get('ece', np.nan):.4f}  (+- {stds.get('ece', 0):.4f})")
    print(f"  PnL-proxy (total):  {means.get('pnl_proxy', np.nan):.4f}  (+- {stds.get('pnl_proxy', 0):.4f})")
    print("  --- Per-class (mean over folds) ---")
    for name in CLASS_ORDER:
        p = means.get(f"precision_{name}", np.nan)
        r = means.get(f"recall_{name}", np.nan)
        print(f"    {name:5s}: precision {p:.4f}, recall {r:.4f}")
    print("=" * 60 + "\n")


def get_default_data_path() -> str:
    """Default path to CSV relative to project root."""
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "btcusdt_5m_final_with_targets.csv")
