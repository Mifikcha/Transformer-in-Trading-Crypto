import pandas as pd
import numpy as np


def run(input_path: str, output_path: str) -> None:
    """Add volume statistics (rolling mean, z-score, anomalous ratio)."""
    print("=== Добавление объёмных статистик ===")
    df = pd.read_parquet(input_path)
    print(f"Загружено строк: {len(df):,}")
    print(f"Период: {df['ts'].min()} → {df['ts'].max()}")

    v = df['volume_perp']
    windows_min = [60, 240]
    windows_steps = [round(m / 5) for m in windows_min]

    print("Расчёт rolling volume mean...")
    for win_min, win_step in zip(windows_min, windows_steps):
        df[f'rolling_vol_mean_{win_min}min'] = v.rolling(win_step).mean()

    print("Расчёт z-score...")
    for win_min, win_step in zip(windows_min, windows_steps):
        mean_col = f'rolling_vol_mean_{win_min}min'
        std_col = f'rolling_vol_std_{win_min}min'
        df[std_col] = v.rolling(win_step).std()
        df[f'z_score_vol_{win_min}min'] = (v - df[mean_col]) / df[std_col].replace(0, np.nan)

    print("Расчёт доли аномального объёма...")
    for win_min, win_step in zip(windows_min, windows_steps):
        mean_col = f'rolling_vol_mean_{win_min}min'
        std_col = f'rolling_vol_std_{win_min}min'
        df[f'anomalous_vol_{win_min}min'] = (v > (df[mean_col] + 2 * df[std_col])).astype(int)
        df[f'anomalous_ratio_{win_min}min'] = df[f'anomalous_vol_{win_min}min'].rolling(win_step).mean()

    vol_cols = [c for c in df.columns if 'rolling_vol' in c or 'z_score_vol' in c or 'anomalous' in c]
    print("\nПроверка новых признаков (последние 5 строк):")
    print(df[vol_cols].tail(5))
    print("\nПропуски в новых признаках:")
    print(df[vol_cols].isna().sum())

    df.to_parquet(output_path, index=False)
    print(f"\nСохранено: {output_path}")
    print("Объёмные статистики добавлены. Готово к funding/OI или таргетам.")


if __name__ == "__main__":
    INPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_volatility.parquet"
    OUTPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_volume_stats.parquet"
    run(INPUT_PATH, OUTPUT_PATH)
