"""
Signal layer that maps model probabilities to deterministic trade intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class SignalConfig:
    threshold_long: float
    threshold_short: float
    margin: float
    use_margin: bool
    atr_min: float
    atr_max: float
    session_filter_enabled: bool = False


class SignalLayer:
    def __init__(self, config: SignalConfig) -> None:
        self.config = config

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        threshold_long: float,
        threshold_short: float,
        margin: float,
        use_margin: bool,
        atr_percentile_low: float,
        atr_percentile_high: float,
        session_filter_enabled: bool = False,
    ) -> "SignalLayer":
        atr_series = df["atr_14"].astype(float)
        atr_min = float(atr_series.quantile(atr_percentile_low / 100.0))
        atr_max = float(atr_series.quantile(atr_percentile_high / 100.0))
        cfg = SignalConfig(
            threshold_long=threshold_long,
            threshold_short=threshold_short,
            margin=margin,
            use_margin=use_margin,
            atr_min=atr_min,
            atr_max=atr_max,
            session_filter_enabled=session_filter_enabled,
        )
        return SignalLayer(cfg)

    def _session_ok(self, row: pd.Series) -> bool:
        if not self.config.session_filter_enabled:
            return True
        return bool(row.get("is_eu_session", 0) == 1 or row.get("is_ny_session", 0) == 1)

    def get_signal(self, row: pd.Series) -> Optional[str]:
        atr = float(row.get("atr_14", 0.0))
        if atr <= 0:
            return None
        if atr < self.config.atr_min or atr > self.config.atr_max:
            return None
        if not self._session_ok(row):
            return None

        p_flat = float(row.get("p_flat", 0.0))
        p_short = float(row.get("p_short", 0.0))
        p_long = float(row.get("p_long", 0.0))

        long_ok = p_long > self.config.threshold_long
        short_ok = p_short > self.config.threshold_short

        if self.config.use_margin:
            long_ok = long_ok and (p_long - p_flat > self.config.margin)
            short_ok = short_ok and (p_short - p_flat > self.config.margin)

        if long_ok and short_ok:
            # Resolve ambiguity by larger edge vs flat.
            long_edge = p_long - p_flat
            short_edge = p_short - p_flat
            return "long" if long_edge >= short_edge else "short"
        if long_ok:
            return "long"
        if short_ok:
            return "short"
        return None
