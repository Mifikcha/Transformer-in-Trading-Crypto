import pandas as pd
import numpy as np


def run(input_path: str, output_path: str) -> None:
    """Add volatility features (SMA, delta SMA, ATR, realized vol)."""
    print("=== Добавление оценок волатильности ===")
    df = pd.read_parquet(input_path)
    print(f"Загружено строк: {len(df):,}")
    print(f"Период: {df['ts'].min()} → {df['ts'].max()}")

    c = df['close_perp']
    h = df['high_perp']
    l = df['low_perp']
    windows_min = [15, 60, 240]
    windows_steps = [round(m / 5) for m in windows_min]

    print("Расчёт SMA и их дельты...")
    for win_min, win_step in zip(windows_min, windows_steps):
        sma = c.rolling(win_step).mean()
        df[f'sma_{win_min}min'] = sma
        df[f'delta_sma_{win_min}min'] = sma.diff()

    print("Расчёт ATR...")
    tr = np.maximum(
        h - l,
        np.maximum(abs(h - c.shift()), abs(l - c.shift()))
    )
    df['atr_14'] = tr.rolling(14).mean()

    log_ret_5 = df['log_return_5min']
    print("Расчёт realized volatility...")
    for win_min, win_step in zip(windows_min, windows_steps):
        rv = np.sqrt((log_ret_5**2).rolling(win_step).sum())
        df[f'realized_vol_{win_min}min'] = rv

    vol_cols = [col for col in df.columns if 'sma' in col or 'delta_sma' in col or 'atr' in col or 'realized_vol' in col]
    print("\nПроверка новых признаков (последние 5 строк):")
    print(df[vol_cols].tail(5))
    print("\nПропуски в новых признаках:")
    print(df[vol_cols].isna().sum())

    df.to_parquet(output_path, index=False)
    print(f"\nСохранено: {output_path}")
    print("Оценки волатильности добавлены. Готово к объёмным статистикам или funding/OI.")


if __name__ == "__main__":
    INPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_price_derivatives.parquet"
    OUTPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_volatility.parquet"
    run(INPUT_PATH, OUTPUT_PATH)
