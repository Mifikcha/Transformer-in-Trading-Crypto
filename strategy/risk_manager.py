"""
Risk constraints for the backtest engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd


@dataclass
class RiskManager:
    max_daily_loss_pct: float
    cooldown_bars: int
    max_position: int
    current_day: Optional[date] = None
    day_realized_pnl_pct: float = 0.0
    daily_stop_active: bool = False
    cooldown_until_bar: int = -1

    def _roll_day(self, ts: pd.Timestamp) -> None:
        d = ts.date()
        if self.current_day != d:
            self.current_day = d
            self.day_realized_pnl_pct = 0.0
            self.daily_stop_active = False

    def can_open_trade(self, ts: pd.Timestamp, bar_idx: int, current_positions: int) -> bool:
        self._roll_day(ts)
        if current_positions >= self.max_position:
            return False
        if self.daily_stop_active:
            return False
        if bar_idx <= self.cooldown_until_bar:
            return False
        return True

    def on_trade_close(self, ts: pd.Timestamp, pnl_pct: float, bar_idx: int) -> None:
        self._roll_day(ts)
        self.day_realized_pnl_pct += float(pnl_pct)
        if self.day_realized_pnl_pct <= -abs(self.max_daily_loss_pct):
            self.daily_stop_active = True
        self.cooldown_until_bar = bar_idx + self.cooldown_bars
