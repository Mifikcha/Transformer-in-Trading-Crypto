"""
Event-level backtest engine for one-position strategy with ATR-based SL/TP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from risk_manager import RiskManager
from signal_layer import SignalLayer


@dataclass(frozen=True)
class BacktestConfig:
    commission_per_side: float
    effective_half_spread: float
    slippage_per_side: float
    latency_bars: int
    max_hold_bars: int
    min_stop_distance_pct: float
    max_stop_daily_atr_ratio: float


@dataclass
class Position:
    side: str
    entry_bar: int
    entry_ts: pd.Timestamp
    entry_price: float
    stop_price: float
    take_price: float
    stop_distance: float
    take_distance: float
    atr_14_at_entry: float
    atr_daily_at_entry: float


def add_daily_atr(df: pd.DataFrame, daily_lookback: int = 14) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["ts"], utc=True).dt.date
    tmp = out.copy()
    tmp["date"] = pd.to_datetime(tmp["ts"], utc=True).dt.date
    daily = (
        tmp.groupby("date", as_index=False)
        .agg(high_day=("high_perp", "max"), low_day=("low_perp", "min"), close_day=("close_perp", "last"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    prev_close = daily["close_day"].shift(1)
    tr = np.maximum(
        daily["high_day"] - daily["low_day"],
        np.maximum((daily["high_day"] - prev_close).abs(), (daily["low_day"] - prev_close).abs()),
    )
    daily["atr_daily"] = tr.rolling(daily_lookback, min_periods=1).mean()
    out = out.merge(daily[["date", "atr_daily"]], on="date", how="left")
    out = out.drop(columns=["date"])
    return out


def _apply_entry_friction(side: str, mid_price: float, cfg: BacktestConfig) -> float:
    x = cfg.effective_half_spread + cfg.slippage_per_side
    if side == "long":
        return float(mid_price * (1.0 + x))
    return float(mid_price * (1.0 - x))


def _apply_exit_friction(side: str, mid_price: float, cfg: BacktestConfig) -> float:
    x = cfg.effective_half_spread + cfg.slippage_per_side
    if side == "long":
        return float(mid_price * (1.0 - x))
    return float(mid_price * (1.0 + x))


def _pnl_pct(side: str, entry_price: float, exit_price: float, cfg: BacktestConfig) -> float:
    if side == "long":
        gross = exit_price / entry_price - 1.0
    else:
        gross = entry_price / exit_price - 1.0
    # Commission is applied on both entry and exit.
    net = gross - 2.0 * cfg.commission_per_side
    return float(net)


def _resolve_bar_hit(side: str, low: float, high: float, stop_price: float, take_price: float) -> Optional[str]:
    if side == "long":
        stop_hit = low <= stop_price
        take_hit = high >= take_price
    else:
        stop_hit = high >= stop_price
        take_hit = low <= take_price
    if stop_hit and take_hit:
        return "stop"  # pessimistic ordering
    if stop_hit:
        return "stop"
    if take_hit:
        return "take"
    return None


def run_backtest(
    df: pd.DataFrame,
    signal_layer: SignalLayer,
    risk_manager: RiskManager,
    cfg: BacktestConfig,
    k_sl: float,
    k_tp: float,
) -> pd.DataFrame:
    """
    Run one backtest on a single dataframe segment and return trade log.
    Required columns:
    - ts, close_perp, high_perp, low_perp, atr_14, atr_daily, p_flat, p_short, p_long
    """
    work = df.copy().reset_index(drop=True)
    trades: list[dict] = []
    position: Position | None = None
    pending_entry: tuple[int, str] | None = None

    for i, row in work.iterrows():
        ts = pd.to_datetime(row["ts"], utc=True)
        low = float(row["low_perp"])
        high = float(row["high_perp"])
        close = float(row["close_perp"])
        atr_14 = float(row.get("atr_14", 0.0))
        atr_daily = float(row.get("atr_daily", 0.0))

        # Manage open position first.
        if position is not None:
            hit = _resolve_bar_hit(position.side, low, high, position.stop_price, position.take_price)
            hold_bars = i - position.entry_bar
            if hit is not None:
                raw_exit = position.stop_price if hit == "stop" else position.take_price
                exit_price = _apply_exit_friction(position.side, raw_exit, cfg)
                pnl_pct = _pnl_pct(position.side, position.entry_price, exit_price, cfg)
                trades.append(
                    {
                        "entry_ts": position.entry_ts,
                        "exit_ts": ts,
                        "entry_bar": position.entry_bar,
                        "exit_bar": i,
                        "side": position.side,
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "stop_price": position.stop_price,
                        "take_price": position.take_price,
                        "stop_distance": position.stop_distance,
                        "take_distance": position.take_distance,
                        "atr_14_at_entry": position.atr_14_at_entry,
                        "atr_daily_at_entry": position.atr_daily_at_entry,
                        "hold_bars": hold_bars,
                        "exit_reason": hit,
                        "pnl_pct": pnl_pct,
                    }
                )
                risk_manager.on_trade_close(ts=ts, pnl_pct=pnl_pct, bar_idx=i)
                position = None
                pending_entry = None
            elif hold_bars >= cfg.max_hold_bars:
                exit_price = _apply_exit_friction(position.side, close, cfg)
                pnl_pct = _pnl_pct(position.side, position.entry_price, exit_price, cfg)
                trades.append(
                    {
                        "entry_ts": position.entry_ts,
                        "exit_ts": ts,
                        "entry_bar": position.entry_bar,
                        "exit_bar": i,
                        "side": position.side,
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "stop_price": position.stop_price,
                        "take_price": position.take_price,
                        "stop_distance": position.stop_distance,
                        "take_distance": position.take_distance,
                        "atr_14_at_entry": position.atr_14_at_entry,
                        "atr_daily_at_entry": position.atr_daily_at_entry,
                        "hold_bars": hold_bars,
                        "exit_reason": "timeout",
                        "pnl_pct": pnl_pct,
                    }
                )
                risk_manager.on_trade_close(ts=ts, pnl_pct=pnl_pct, bar_idx=i)
                position = None
                pending_entry = None

        # Execute pending signal after latency.
        if position is None and pending_entry is not None:
            exec_idx, side = pending_entry
            if i >= exec_idx:
                if risk_manager.can_open_trade(ts=ts, bar_idx=i, current_positions=0):
                    if atr_14 > 0 and atr_daily > 0:
                        base_stop = k_sl * atr_14
                        stop_cap = cfg.max_stop_daily_atr_ratio * atr_daily
                        stop_distance = min(base_stop, stop_cap)
                        min_stop = max(cfg.min_stop_distance_pct * close, 0.0)
                        if stop_distance >= min_stop:
                            take_distance = max(k_tp * atr_14, min_stop)
                            entry_price = _apply_entry_friction(side, close, cfg)
                            if side == "long":
                                stop_price = entry_price - stop_distance
                                take_price = entry_price + take_distance
                            else:
                                stop_price = entry_price + stop_distance
                                take_price = entry_price - take_distance
                            position = Position(
                                side=side,
                                entry_bar=i,
                                entry_ts=ts,
                                entry_price=entry_price,
                                stop_price=float(stop_price),
                                take_price=float(take_price),
                                stop_distance=float(stop_distance),
                                take_distance=float(take_distance),
                                atr_14_at_entry=float(atr_14),
                                atr_daily_at_entry=float(atr_daily),
                            )
                pending_entry = None

        # Generate next signal only when flat and no pending order.
        if position is None and pending_entry is None:
            if risk_manager.can_open_trade(ts=ts, bar_idx=i, current_positions=0):
                signal = signal_layer.get_signal(row)
                if signal in ("long", "short"):
                    pending_entry = (i + cfg.latency_bars, signal)

    return pd.DataFrame(trades)
