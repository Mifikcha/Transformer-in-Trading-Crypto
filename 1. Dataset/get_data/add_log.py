import pandas as pd
import numpy as np


def run(input_path: str, output_path: str) -> None:
    """Add price derivatives (log returns, cum returns, delta price)."""
    print("=== Добавление производных от цены ===")
    df = pd.read_parquet(input_path)
    print(f"Загружено строк: {len(df):,}")
    print(f"Период: {df['ts'].min()} → {df['ts'].max()}")

    c = df['close_perp']
    lags_minutes = [1, 3, 5, 15, 60]
    lags_steps = [max(1, round(m / 5)) for m in lags_minutes]
    for lag_min, lag_step in zip(lags_minutes, lags_steps):
        df[f'log_return_{lag_min}min'] = np.log(c / c.shift(lag_step))

    cum_windows_min = [15, 60, 240]
    cum_windows_steps = [round(m / 5) for m in cum_windows_min]
    for win_min, win_step in zip(cum_windows_min, cum_windows_steps):
        df[f'cum_log_return_{win_min}min'] = np.log(c / c.shift(win_step))

    df['delta_price_5min'] = c.diff()

    print("\nПроверка новых признаков (последние 5 строк):")
    print(df.filter(regex='log_return|cum_log_return|delta_price').tail(5))
    print("\nПропуски в новых признаках:")
    print(df.filter(regex='log_return|cum_log_return|delta_price').isna().sum())

    df.to_parquet(output_path, index=False)
    print(f"\nСохранено: {output_path}")
    print("Производные от цены добавлены. Готово к волатильности/объёмам.")


if __name__ == "__main__":
    INPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_combined_with_time_features.parquet"
    OUTPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_price_derivatives.parquet"
    run(INPUT_PATH, OUTPUT_PATH)
