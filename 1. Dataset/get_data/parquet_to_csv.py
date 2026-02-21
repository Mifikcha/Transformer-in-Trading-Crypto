import pandas as pd
import sys


def convert(in_path: str, out_path: str) -> None:
    """Convert parquet file to CSV."""
    df = pd.read_parquet(in_path)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parquet_to_csv.py <input.parquet> <output.csv>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
