"""
Strategy metrics from trade logs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _equity_curve_from_returns(pnl_pct: pd.Series) -> pd.Series:
    if pnl_pct.empty:
        return pd.Series(dtype=float)
    return (1.0 + pnl_pct.astype(float)).cumprod()


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def compute_strategy_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "n_trades": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_trade": 0.0,
            "profit_factor": 0.0,
            "sharpe_trade": 0.0,
            "turnover": 0.0,
            "avg_hold_bars": 0.0,
        }

    pnl = trades["pnl_pct"].astype(float).reset_index(drop=True)
    equity = _equity_curve_from_returns(pnl)
    total_return = float(equity.iloc[-1] - 1.0)
    max_dd = _max_drawdown(equity)
    win_rate = float((pnl > 0).mean())
    avg_trade = float(pnl.mean())
    pos_sum = float(pnl[pnl > 0].sum())
    neg_sum = float(np.abs(pnl[pnl < 0].sum()))
    profit_factor = float(pos_sum / neg_sum) if neg_sum > 0 else float("inf")
    sharpe = 0.0
    if pnl.std(ddof=0) > 0:
        sharpe = float(pnl.mean() / pnl.std(ddof=0) * np.sqrt(len(pnl)))

    # Approx turnover proxy: round trip notional 2 per completed trade at fixed unit size.
    turnover = float(len(trades) * 2.0)
    avg_hold = float(trades["hold_bars"].astype(float).mean()) if "hold_bars" in trades else 0.0

    return {
        "n_trades": int(len(trades)),
        "total_return": total_return,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "profit_factor": profit_factor,
        "sharpe_trade": sharpe,
        "turnover": turnover,
        "avg_hold_bars": avg_hold,
    }


def trade_distribution_by_hour(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame({"hour": [], "n_trades": []})
    ts = pd.to_datetime(trades["entry_ts"], utc=True)
    out = (
        ts.dt.hour.value_counts()
        .sort_index()
        .rename_axis("hour")
        .reset_index(name="n_trades")
    )
    return out


def fold_metrics_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "fold_id" not in trades.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for fold_id, tdf in trades.groupby("fold_id"):
        row = compute_strategy_metrics(tdf)
        row["fold_id"] = int(fold_id)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)
