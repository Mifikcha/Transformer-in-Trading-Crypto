"""
Orchestrator for Step 6 strategy pipeline.

Run from project root:
  python strategy/run_strategy.py
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
BASELINES_DIR = os.path.join(BASE_DIR, "baselines")
for p in (SCRIPT_DIR, BASE_DIR, BASELINES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import lightgbm as lgb  # type: ignore

    HAS_LIGHTGBM = True
except Exception:
    lgb = None
    HAS_LIGHTGBM = False

from baselines.utils import (  # noqa: E402
    get_default_data_path,
    get_feature_columns,
    get_lightgbm_classifier_config,
    load_dataset,
    walk_forward_split,
)
from backtest_engine import BacktestConfig, add_daily_atr  # noqa: E402
import config as cfg  # noqa: E402
from optimize_rrr import MonteCarloConfig, run_rrr_optimization  # noqa: E402
from signal_layer import SignalLayer  # noqa: E402


def _build_out_dir(output_dir: Optional[str]) -> str:
    out_dir = output_dir or cfg.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def generate_lgbm_predictions(
    df: pd.DataFrame,
    n_splits: int,
) -> pd.DataFrame:
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].astype(float).fillna(0.0)
    splits = walk_forward_split(df, n_splits=n_splits)
    pred_parts: list[pd.DataFrame] = []
    model_cfg = get_lightgbm_classifier_config()

    for fold_id, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = (
            df.iloc[train_idx]["trading_class_base"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"flat": 0, "short": 1, "long": 2})
            .astype(int)
            .values
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if HAS_LIGHTGBM:
            model = lgb.LGBMClassifier(**model_cfg)
        else:
            model = LogisticRegression(
                solver="lbfgs",
                max_iter=500,
                class_weight="balanced",
                random_state=42,
            )
        model.fit(X_train_s, y_train)
        y_proba = model.predict_proba(X_test_s)

        fold_df = df.iloc[test_idx].copy()
        fold_df["fold_id"] = int(fold_id)
        fold_df["p_flat"] = y_proba[:, 0]
        fold_df["p_short"] = y_proba[:, 1]
        fold_df["p_long"] = y_proba[:, 2]
        pred_parts.append(fold_df)

    out = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return out


def main(data_path: str | None = None, n_splits: int = cfg.N_SPLITS, output_dir: str | None = None) -> None:
    out_dir = _build_out_dir(output_dir)
    path = data_path or get_default_data_path()
    print(f"Loading dataset: {path}")
    df = load_dataset(path)
    print(f"Rows after valid-target filter: {len(df):,}")

    model_name = "LightGBM" if HAS_LIGHTGBM else "LogisticRegression fallback"
    print(f"Generating walk-forward predictions ({model_name})...")
    pred_df = generate_lgbm_predictions(df=df, n_splits=n_splits)
    if pred_df.empty:
        raise RuntimeError("No walk-forward predictions were generated.")

    pred_df = add_daily_atr(pred_df)
    pred_df.to_csv(os.path.join(out_dir, "predictions_walkforward_lgbm.csv"), index=False)

    signal_layer = SignalLayer.from_dataframe(
        pred_df,
        threshold_long=cfg.ENTRY_THRESHOLD_LONG,
        threshold_short=cfg.ENTRY_THRESHOLD_SHORT,
        margin=cfg.ENTRY_MARGIN,
        use_margin=cfg.USE_MARGIN,
        atr_percentile_low=cfg.ATR_PERCENTILE_LOW,
        atr_percentile_high=cfg.ATR_PERCENTILE_HIGH,
        session_filter_enabled=cfg.SESSION_FILTER_ENABLED,
    )

    backtest_cfg = BacktestConfig(
        commission_per_side=cfg.COMMISSION_PER_SIDE,
        effective_half_spread=cfg.EFFECTIVE_HALF_SPREAD,
        slippage_per_side=cfg.SLIPPAGE_PER_SIDE,
        latency_bars=cfg.LATENCY_BARS,
        max_hold_bars=cfg.MAX_HOLD_BARS,
        min_stop_distance_pct=cfg.MIN_STOP_DISTANCE_PCT,
        max_stop_daily_atr_ratio=cfg.MAX_STOP_DAILY_ATR_RATIO,
    )
    mc_cfg = MonteCarloConfig(
        n_simulations=cfg.MC_N_SIMULATIONS,
        random_state=cfg.MC_RANDOM_STATE,
        max_prob_negative=cfg.MC_MAX_PROB_NEGATIVE,
    )

    folds = sorted(pred_df["fold_id"].dropna().astype(int).unique().tolist())
    print(f"Running optimization on folds: {folds}")
    grid_df, top_summary = run_rrr_optimization(
        df_with_predictions=pred_df,
        folds=folds,
        signal_layer=signal_layer,
        backtest_cfg=backtest_cfg,
        max_daily_loss_pct=cfg.MAX_DAILY_LOSS_PCT,
        cooldown_bars=cfg.COOLDOWN_BARS,
        max_position=cfg.MAX_POSITION,
        k_sl_grid=cfg.K_SL_GRID,
        k_tp_grid=cfg.K_TP_GRID,
        top_k_candidates=cfg.TOP_K_CANDIDATES,
        mc_cfg=mc_cfg,
        output_dir=out_dir,
    )

    robust = top_summary.loc[top_summary.get("mc_is_robust", 0) == 1].copy()
    if robust.empty:
        best_row = top_summary.iloc[0:1].copy()
    else:
        best_row = robust.sort_values(
            ["p_total_return_50", "total_return"], ascending=[False, False]
        ).head(1)
    best_row.to_csv(os.path.join(out_dir, "best_strategy.csv"), index=False)

    print("\n" + "=" * 80)
    print("STEP 6 STRATEGY PIPELINE FINISHED")
    print("=" * 80)
    print(f"Output dir: {out_dir}")
    print(f"Grid candidates: {len(grid_df)}")
    if not best_row.empty:
        r = best_row.iloc[0]
        print(f"Best candidate: {r['candidate_id']} (k_sl={r['k_sl']:.2f}, k_tp={r['k_tp']:.2f})")
        print("Best strategy metrics:")
        print(f"  total_return:         {r.get('total_return', float('nan')):.4f}")
        print(f"  max_drawdown:         {r.get('max_drawdown', float('nan')):.4f}")
        print(f"  n_trades:             {int(r.get('n_trades', 0))}")
        print(f"  win_rate:             {r.get('win_rate', float('nan')):.4f}")
        print(f"  avg_trade:            {r.get('avg_trade', float('nan')):.6f}")
        print(f"  sharpe_trade:         {r.get('sharpe_trade', float('nan')):.4f}")
        print(f"  profit_factor:        {r.get('profit_factor', float('nan')):.4f}")
        print(f"  avg_hold_bars:        {r.get('avg_hold_bars', float('nan')):.2f}")
        print(f"  turnover:             {r.get('turnover', float('nan')):.2f}")
        print("Monte Carlo robustness:")
        print(f"  p_total_return_05:    {r.get('p_total_return_05', float('nan')):.4f}")
        print(f"  p_total_return_50:    {r.get('p_total_return_50', float('nan')):.4f}")
        print(f"  p_total_return_95:    {r.get('p_total_return_95', float('nan')):.4f}")
        print(f"  p_max_drawdown_95:    {r.get('p_max_drawdown_95', float('nan')):.4f}")
        print(f"  prob_negative_return: {r.get('prob_negative_return', float('nan')):.4f}")
        print(f"  mc_is_robust:         {int(r.get('mc_is_robust', 0))}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
