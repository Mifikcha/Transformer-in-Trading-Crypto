# validate_OHLCV.py
"""
Validate and clean OHLCV dataset.

Checks:
- required columns
- dtypes (ts must be UTC)
- sorting
- duplicate timestamps (removed)
- time step consistency (reported)
- OHLC logical consistency

Writes cleaned dataset to new file.

python get_data/validate_OHLCV.py get_data/output/main/btcusdt_5m_spot.parquet get_data/output/main/btcusdt_1m_spot_clean.parquet
"""

import sys
import pandas as pd

REQUIRED_COLS = ["ts", "open", "high", "low", "close", "volume", "turnover"]


def validate_and_clean(
    in_path: str,
    out_path: str,
    freq: str = "5min",
) -> None:

    # --- load ---
    if in_path.endswith(".parquet"):
        df = pd.read_parquet(in_path)
    elif in_path.endswith(".csv"):
        df = pd.read_csv(in_path, parse_dates=["ts"])
    else:
        raise ValueError("Unsupported file format")

    print(f"Loaded: {df.shape[0]:,} rows")

    # --- columns ---
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # --- ts dtype ---
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if df["ts"].dt.tz is None:
        raise ValueError("ts must be timezone-aware (UTC)")

    # --- sort ---
    df = df.sort_values("ts").reset_index(drop=True)

    # --- duplicates ---
    dup_count = df["ts"].duplicated().sum()
    if dup_count > 0:
        print(f"Removing duplicates: {dup_count}")
        df = df.drop_duplicates(subset="ts", keep="first")

    # --- time step check ---
    dt = df["ts"].diff().dropna()
    expected = pd.Timedelta(freq)

    bad_steps = (dt != expected).sum()
    if bad_steps > 0:
        print(f"WARNING: irregular time steps found: {bad_steps}")
        print("Sample:")
        print(df.loc[dt.ne(expected).to_numpy().nonzero()[0][:5] + 1, "ts"])

    # --- OHLC sanity ---
    bad_ohlc = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1)) |
        (df["low"]  > df[["open", "close", "high"]].min(axis=1))
    ).sum()

    if bad_ohlc > 0:
        raise ValueError(f"OHLC consistency failed on {bad_ohlc} rows")

    # --- save ---
    if out_path.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"Saved cleaned dataset: {df.shape[0]:,} rows")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate_OHLCV.py <input_file> <output_file>")
        sys.exit(1)

    validate_and_clean(sys.argv[1], sys.argv[2])
