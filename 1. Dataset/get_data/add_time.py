import pandas as pd
import numpy as np


def _minutes_to_session_open(ts):
    """Время до ближайшего открытия сессии в минутах (0–1440)."""
    hour = ts.hour
    if hour < 0:
        next_asia = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        next_asia = (ts + pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    if hour < 8:
        next_eu = ts.replace(hour=8, minute=0, second=0, microsecond=0)
    else:
        next_eu = (ts + pd.Timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
    if hour < 14:
        next_ny = ts.replace(hour=14, minute=0, second=0, microsecond=0)
    else:
        next_ny = (ts + pd.Timedelta(days=1)).replace(hour=14, minute=0, second=0, microsecond=0)
    minutes_asia = (next_asia - ts).total_seconds() / 60
    minutes_eu = (next_eu - ts).total_seconds() / 60
    minutes_ny = (next_ny - ts).total_seconds() / 60
    return minutes_asia, minutes_eu, minutes_ny


def run(input_path: str, output_path: str) -> None:
    """Add time-based features (hour, weekday, sin/cos, session indicators)."""
    print("=== Добавление временных признаков ===")
    df = pd.read_parquet(input_path)
    print(f"Загружено строк: {len(df):,}")
    print(f"Период: {df['ts'].min()} → {df['ts'].max()}")

    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df['hour'] = df['ts'].dt.hour
    df['weekday'] = df['ts'].dt.weekday
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_day'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['cos_day'] = np.cos(2 * np.pi * df['weekday'] / 7)

    print("Расчёт времени до сессий...")
    df[['min_to_asia_open', 'min_to_eu_open', 'min_to_ny_open']] = df['ts'].apply(
        lambda ts: pd.Series(_minutes_to_session_open(ts))
    )
    df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
    df['is_eu_session'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
    df['is_ny_session'] = ((df['hour'] >= 14) & (df['hour'] < 21)).astype(int)

    print("\nПроверка добавленных признаков:")
    print(df[['ts', 'hour', 'weekday', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day',
              'min_to_ny_open', 'is_ny_session']].tail(8))
    print(f"\nПропуски в новых признаках: {df.filter(regex='sin|cos|min_to|is_').isna().sum().sum()}")

    df.to_parquet(output_path, index=False)
    print(f"\nСохранено: {output_path}")
    print("Временные признаки добавлены. Готово к следующему блоку фич.")


if __name__ == "__main__":
    INPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_combined_2020-2026.parquet"
    OUTPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_combined_with_time_features.parquet"
    run(INPUT_PATH, OUTPUT_PATH)
