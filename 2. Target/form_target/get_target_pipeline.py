"""
Target pipeline: check_dataset -> fix_time -> count_fees -> form_target.

Reads btcusdt_5m_final_cleaned.csv, produces btcusdt_5m_final_with_targets.csv.

Run from project root:
  python "2. Target/form_target/get_target_pipeline.py"

Or from form_target:
  python get_target_pipeline.py
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import check_dataset
import fix_time
import count_fees
import form_target

INPUT_CSV = os.path.join(_SCRIPT_DIR, "..", "..", "1. Dataset", "get_data", "output", "_main", "_final", "btcusdt_5m_final_cleaned.csv")
HORIZONS_CSV = os.path.join(_SCRIPT_DIR, "btcusdt_5m_with_horizons.csv")
FINAL_CSV = os.path.join(_SCRIPT_DIR, "btcusdt_5m_final_with_targets.csv")


def run_pipeline() -> None:
    print("Step 1: check_dataset")
    check_dataset.run(INPUT_CSV)

    print("\nStep 2: fix_time")
    fix_time.run(INPUT_CSV, HORIZONS_CSV)

    print("\nStep 3: count_fees")
    fees_config = count_fees.run(HORIZONS_CSV)

    print("\nStep 4: form_target")
    form_target.run(HORIZONS_CSV, FINAL_CSV, fees_config["c_scenarios"])

    print("\n=== Pipeline finished ===")
    print(f"Final dataset: {FINAL_CSV}")


if __name__ == "__main__":
    # Use paths relative to cwd (typically form_target when run from there)
    run_pipeline()
