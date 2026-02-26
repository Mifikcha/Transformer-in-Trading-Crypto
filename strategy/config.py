"""
Configuration for Step 6 trading engine.
"""

from __future__ import annotations

import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# Dataset / validation
N_SPLITS = 5


# Costs
# Existing assumptions from 2. Target/form_target/count_fees.py:
# - commissions_roundtrip: 0.0009
# - effective_half_spread: 0.0003
# - slippage_proxy: 0.0007
COMMISSION_ROUNDTRIP = 0.0009
COMMISSION_PER_SIDE = COMMISSION_ROUNDTRIP / 2.0
EFFECTIVE_HALF_SPREAD = 0.0003
SLIPPAGE_PER_SIDE = 0.0007


# Signal layer
ENTRY_THRESHOLD_LONG = 0.45
ENTRY_THRESHOLD_SHORT = 0.45
ENTRY_MARGIN = 0.10
USE_MARGIN = True
ATR_PERCENTILE_LOW = 5.0
ATR_PERCENTILE_HIGH = 95.0
SESSION_FILTER_ENABLED = False


# Execution assumptions
LATENCY_BARS = 1
MAX_HOLD_BARS = 60  # 5 hours on 5-minute bars


# Position / risk
MAX_POSITION = 1
COOLDOWN_BARS = 3
MAX_DAILY_LOSS_PCT = 0.02


# Stop / take-profit
MIN_STOP_DISTANCE_PCT = 0.0002
MAX_STOP_DAILY_ATR_RATIO = 0.20  # stop cannot exceed 20% of daily ATR


# RRR optimization
K_SL_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
K_TP_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
TOP_K_CANDIDATES = 5


# Monte Carlo robustness
MC_N_SIMULATIONS = 5000
MC_RANDOM_STATE = 42
MC_MAX_PROB_NEGATIVE = 0.45
