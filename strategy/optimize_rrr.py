"""
Grid search and Monte Carlo robustness for (k_sl, k_tp).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest_engine import BacktestConfig, run_backtest
from metrics import compute_strategy_metrics, fold_metrics_table, trade_distribution_by_hour
from risk_manager import RiskManager
from signal_layer import SignalLayer


@dataclass(frozen=True)
class MonteCarloConfig:
    n_simulations: int
    random_state: int
    max_prob_negative: float


def _select_top_candidates(grid_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if grid_df.empty:
        return grid_df
    ranked = grid_df.copy()
    ranked["rank_return"] = ranked["total_return"].rank(ascending=False, method="average")
    ranked["rank_drawdown"] = ranked["max_drawdown"].abs().rank(ascending=True, method="average")
    ranked["rank_combined"] = ranked["rank_return"] + ranked["rank_drawdown"]
    ranked = ranked.sort_values(
        ["rank_combined", "total_return", "max_drawdown"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return ranked.head(top_k).copy()


def _mc_simulate_one(trade_returns: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(trade_returns)
    if n == 0:
        return 0.0, 0.0
    idx = rng.integers(0, n, size=n)
    sample = trade_returns[idx]
    equity = np.cumprod(1.0 + sample)
    total_ret = float(equity[-1] - 1.0)
    roll_max = np.maximum.accumulate(equity)
    dd = equity / roll_max - 1.0
    max_dd = float(dd.min())
    return total_ret, max_dd


def monte_carlo_bootstrap(
    trades: pd.DataFrame,
    candidate_id: str,
    mc_cfg: MonteCarloConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = trades["pnl_pct"].astype(float).values if not trades.empty else np.array([], dtype=float)
    rng = np.random.default_rng(mc_cfg.random_state)
    rows: list[dict] = []
    for sim_id in range(mc_cfg.n_simulations):
        total_ret, max_dd = _mc_simulate_one(returns, rng)
        rows.append(
            {
                "candidate_id": candidate_id,
                "sim_id": sim_id,
                "total_return": total_ret,
                "max_drawdown": max_dd,
            }
        )
    sim_df = pd.DataFrame(rows)
    if sim_df.empty:
        q_df = pd.DataFrame(
            {
                "candidate_id": [candidate_id],
                "p_total_return_05": [0.0],
                "p_total_return_25": [0.0],
                "p_total_return_50": [0.0],
                "p_total_return_75": [0.0],
                "p_total_return_95": [0.0],
                "p_max_drawdown_05": [0.0],
                "p_max_drawdown_25": [0.0],
                "p_max_drawdown_50": [0.0],
                "p_max_drawdown_75": [0.0],
                "p_max_drawdown_95": [0.0],
                "prob_negative_return": [0.0],
            }
        )
        return sim_df, q_df

    q = sim_df.quantile([0.05, 0.25, 0.5, 0.75, 0.95], numeric_only=True)
    q_df = pd.DataFrame(
        {
            "candidate_id": [candidate_id],
            "p_total_return_05": [float(q.loc[0.05, "total_return"])],
            "p_total_return_25": [float(q.loc[0.25, "total_return"])],
            "p_total_return_50": [float(q.loc[0.5, "total_return"])],
            "p_total_return_75": [float(q.loc[0.75, "total_return"])],
            "p_total_return_95": [float(q.loc[0.95, "total_return"])],
            "p_max_drawdown_05": [float(q.loc[0.05, "max_drawdown"])],
            "p_max_drawdown_25": [float(q.loc[0.25, "max_drawdown"])],
            "p_max_drawdown_50": [float(q.loc[0.5, "max_drawdown"])],
            "p_max_drawdown_75": [float(q.loc[0.75, "max_drawdown"])],
            "p_max_drawdown_95": [float(q.loc[0.95, "max_drawdown"])],
            "prob_negative_return": [float((sim_df["total_return"] < 0).mean())],
        }
    )
    return sim_df, q_df


def run_rrr_optimization(
    df_with_predictions: pd.DataFrame,
    folds: list[int],
    signal_layer: SignalLayer,
    backtest_cfg: BacktestConfig,
    max_daily_loss_pct: float,
    cooldown_bars: int,
    max_position: int,
    k_sl_grid: list[float],
    k_tp_grid: list[float],
    top_k_candidates: int,
    mc_cfg: MonteCarloConfig,
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)

    grid_rows: list[dict] = []
    trades_by_candidate: dict[str, pd.DataFrame] = {}

    for k_sl in k_sl_grid:
        for k_tp in k_tp_grid:
            candidate_id = f"ksl_{k_sl:.2f}_ktp_{k_tp:.2f}"
            candidate_trades: list[pd.DataFrame] = []

            for fold_id in folds:
                fold_df = df_with_predictions.loc[df_with_predictions["fold_id"] == fold_id].copy()
                if fold_df.empty:
                    continue
                risk_mgr = RiskManager(
                    max_daily_loss_pct=max_daily_loss_pct,
                    cooldown_bars=cooldown_bars,
                    max_position=max_position,
                )
                trades = run_backtest(
                    df=fold_df,
                    signal_layer=signal_layer,
                    risk_manager=risk_mgr,
                    cfg=backtest_cfg,
                    k_sl=float(k_sl),
                    k_tp=float(k_tp),
                )
                if not trades.empty:
                    trades["fold_id"] = int(fold_id)
                candidate_trades.append(trades)

            all_trades = pd.concat(candidate_trades, ignore_index=True) if candidate_trades else pd.DataFrame()
            trades_by_candidate[candidate_id] = all_trades
            m = compute_strategy_metrics(all_trades)
            m["candidate_id"] = candidate_id
            m["k_sl"] = float(k_sl)
            m["k_tp"] = float(k_tp)
            grid_rows.append(m)

    grid_df = pd.DataFrame(grid_rows).sort_values(
        ["total_return", "max_drawdown"], ascending=[False, False]
    ).reset_index(drop=True)
    grid_path = os.path.join(output_dir, "rrr_grid_results.csv")
    grid_df.to_csv(grid_path, index=False)

    top_df = _select_top_candidates(grid_df, top_k=top_k_candidates)
    top_rows: list[dict] = []

    for _, row in top_df.iterrows():
        candidate_id = str(row["candidate_id"])
        candidate_trades = trades_by_candidate.get(candidate_id, pd.DataFrame())
        trades_path = os.path.join(output_dir, f"trades_{candidate_id}.csv")
        candidate_trades.to_csv(trades_path, index=False)

        fold_table = fold_metrics_table(candidate_trades)
        fold_table.to_csv(os.path.join(output_dir, f"fold_metrics_{candidate_id}.csv"), index=False)

        hourly = trade_distribution_by_hour(candidate_trades)
        hourly.to_csv(os.path.join(output_dir, f"hourly_distribution_{candidate_id}.csv"), index=False)

        sim_df, q_df = monte_carlo_bootstrap(candidate_trades, candidate_id, mc_cfg)
        sim_df.to_csv(os.path.join(output_dir, f"mc_results_{candidate_id}.csv"), index=False)
        q_df.to_csv(os.path.join(output_dir, f"mc_quantiles_{candidate_id}.csv"), index=False)

        out_row = row.to_dict()
        out_row.update(q_df.iloc[0].to_dict())
        out_row["mc_is_robust"] = int(out_row["prob_negative_return"] <= mc_cfg.max_prob_negative)
        top_rows.append(out_row)

    top_summary = pd.DataFrame(top_rows).sort_values(
        ["mc_is_robust", "p_total_return_50", "total_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    top_summary.to_csv(os.path.join(output_dir, "top_strategies_summary.csv"), index=False)

    return grid_df, top_summary
