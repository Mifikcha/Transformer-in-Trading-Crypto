"""
Full data pipeline: raw OHLCV (spot + perp) -> validated -> merged -> features -> targets -> final cleaned -> CSV + EDA.

Run from project root (parent of get_data):
  python get_data/_get_data.py

Or from get_data:
  python _get_data.py
"""
import os
import sys

# Ensure parent directory is on path so "from get_data.xxx" works when run from get_data/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# ─── Constants (CLI defaults for get_OHLCV / validate / parquet_to_csv) ─────
SYMBOL = "BTCUSDT"
CATEGORY_SPOT = "spot"
CATEGORY_PERP = "linear"
INTERVAL = "5"
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"

BASE_DIR = os.path.join(_SCRIPT_DIR, "output", "_main")
RAW_DIR = os.path.join(BASE_DIR, "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "clean")
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "intermediate")
FINAL_DIR = os.path.join(BASE_DIR, "_final")
EDA_DIR_PATH = os.path.join(FINAL_DIR, "EDA_dir")

# File paths for each step
RAW_SPOT = os.path.join(RAW_DIR, "btcusdt_5m_spot_API.parquet")
RAW_PERP = os.path.join(RAW_DIR, "btcusdt_5m_perp_API.parquet")
CLEAN_SPOT = os.path.join(CLEAN_DIR, "btcusdt_5m_spot_API_clean.parquet")
CLEAN_PERP = os.path.join(CLEAN_DIR, "btcusdt_5m_perp_API_clean.parquet")
COMBINED = os.path.join(INTERMEDIATE_DIR, "btcusdt_5m_combined_2020-2026.parquet")
WITH_TIME = os.path.join(INTERMEDIATE_DIR, "btcusdt_5m_combined_with_time_features.parquet")
WITH_LOG = os.path.join(INTERMEDIATE_DIR, "btcusdt_5m_with_price_derivatives.parquet")
WITH_VOL = os.path.join(INTERMEDIATE_DIR, "btcusdt_5m_with_volatility.parquet")
WITH_VOLUME_STATS = os.path.join(INTERMEDIATE_DIR, "btcusdt_5m_with_volume_stats.parquet")
WITH_DERIVATIVES = os.path.join(INTERMEDIATE_DIR, "btcusdt_5m_with_derivatives.parquet")
WITH_TARGETS = os.path.join(FINAL_DIR, "btcusdt_5m_final_with_targets.parquet")
FINAL_CLEANED = os.path.join(FINAL_DIR, "btcusdt_5m_final_cleaned.parquet")
FINAL_CSV = os.path.join(FINAL_DIR, "btcusdt_5m_final_cleaned.csv")


def _ensure_dirs() -> None:
    for d in (RAW_DIR, CLEAN_DIR, INTERMEDIATE_DIR, FINAL_DIR, EDA_DIR_PATH):
        os.makedirs(d, exist_ok=True)


def run_pipeline() -> None:
    _ensure_dirs()

    from get_data.get_OHLCV import fetch_ohlcv
    from get_data.validate_OHLCV import validate_and_clean
    from get_data.merge_dataset import run as merge_run
    from get_data.add_time import run as add_time_run
    from get_data.add_log import run as add_log_run
    from get_data.add_volatility import run as add_volatility_run
    from get_data.add_volume_statistics import run as add_volume_stats_run
    from get_data.add_funding import run as add_funding_run
    from get_data.add_target import run as add_target_run
    from get_data.EDA import clean_final_dataset, run_eda
    from get_data.parquet_to_csv import convert as parquet_to_csv_convert

    print("Step 1: get_OHLCV (spot)")
    fetch_ohlcv(
        symbol=SYMBOL,
        category=CATEGORY_SPOT,
        interval=INTERVAL,
        start=START_DATE,
        end=END_DATE,
        out_path=RAW_SPOT,
    )

    print("\nStep 2: get_OHLCV (perp)")
    fetch_ohlcv(
        symbol=SYMBOL,
        category=CATEGORY_PERP,
        interval=INTERVAL,
        start=START_DATE,
        end=END_DATE,
        out_path=RAW_PERP,
    )

    print("\nStep 3: validate_OHLCV (spot)")
    validate_and_clean(RAW_SPOT, CLEAN_SPOT)

    print("\nStep 4: validate_OHLCV (perp)")
    validate_and_clean(RAW_PERP, CLEAN_PERP)

    print("\nStep 5: merge_dataset")
    merge_run(perp_path=CLEAN_PERP, spot_path=CLEAN_SPOT, output_path=COMBINED)

    print("\nStep 6: add_time")
    add_time_run(COMBINED, WITH_TIME)

    print("\nStep 7: add_log")
    add_log_run(WITH_TIME, WITH_LOG)

    print("\nStep 8: add_volatility")
    add_volatility_run(WITH_LOG, WITH_VOL)

    print("\nStep 9: add_volume_statistics")
    add_volume_stats_run(WITH_VOL, WITH_VOLUME_STATS)

    print("\nStep 10: add_funding")
    add_funding_run(WITH_VOLUME_STATS, WITH_DERIVATIVES, symbol=SYMBOL, category=CATEGORY_PERP)

    print("\nStep 11: add_target")
    add_target_run(WITH_DERIVATIVES, WITH_TARGETS)

    print("\nStep 12: validate (final) + clean_final_dataset")
    validate_and_clean(WITH_TARGETS, FINAL_CLEANED, required_cols=None)
    clean_final_dataset(FINAL_CLEANED, FINAL_CLEANED)

    print("\nStep 13: parquet_to_csv")
    parquet_to_csv_convert(FINAL_CLEANED, FINAL_CSV)

    print("\nStep 14: EDA (analysis only)")
    run_eda(FINAL_CLEANED, EDA_DIR_PATH)

    print("\n=== Pipeline finished ===")
    print(f"Final dataset: {FINAL_CLEANED}")
    print(f"CSV: {FINAL_CSV}")
    print(f"EDA plots: {EDA_DIR_PATH}")


if __name__ == "__main__":
    run_pipeline()
