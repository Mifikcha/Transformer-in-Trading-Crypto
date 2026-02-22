import pandas as pd
import numpy as np


def run(input_path: str) -> None:
    df = pd.read_csv(input_path, parse_dates=['ts'], index_col='ts')
    df.sort_index(inplace=True)
    print(f"Δt: {df.index.to_series().diff().median()}")
    print(f"Период: {df.index.min()} to {df.index.max()}")
    print(f"Средняя волатильность: {df['log_return_1min'].std()}")
    print(df.describe())


if __name__ == "__main__":
    run('btcusdt_5m_final_cleaned.csv')
