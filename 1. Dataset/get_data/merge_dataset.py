import pandas as pd
import numpy as np


def basic_checks(df: pd.DataFrame, name: str) -> None:
    print(f"\nПроверка {name}:")
    print(f"  Строк: {len(df):,}")
    print(f"  Период: {df['ts'].min()} → {df['ts'].max()}")
    print(f"  Пропуски в close: {df['close'].isna().sum():,}")
    print(f"  Монотонность ts: {'OK' if df['ts'].is_monotonic_increasing else 'НАРУШЕНА!'}")
    print(f"  Дубли ts: {df['ts'].duplicated().sum():,}")
    print(f"  Цены разумные: {df['close'].between(100, 500000).all()}")


def run(perp_path: str, spot_path: str, output_path: str) -> None:
    """Merge perp and spot OHLCV datasets (merge_asof backward), add basis column."""
    print("=== Объединение perp и spot (5m) ===")

    print("Загрузка perp...")
    df_perp = pd.read_parquet(perp_path)
    print("Загрузка spot...")
    df_spot = pd.read_parquet(spot_path)

    basic_checks(df_perp, "perp")
    basic_checks(df_spot, "spot")

    df_perp = df_perp.rename(columns={
        'open': 'open_perp', 'high': 'high_perp', 'low': 'low_perp',
        'close': 'close_perp', 'volume': 'volume_perp', 'turnover': 'turnover_perp'
    })
    df_spot = df_spot.rename(columns={
        'open': 'open_spot', 'high': 'high_spot', 'low': 'low_spot',
        'close': 'close_spot', 'volume': 'volume_spot', 'turnover': 'turnover_spot'
    })

    print("\nСинхронизация через merge_asof (backward)...")
    df_combined = pd.merge_asof(
        df_perp.sort_values('ts'),
        df_spot.sort_values('ts'),
        on='ts',
        direction='backward'
    )
    df_combined = df_combined.loc[:, ~df_combined.columns.str.endswith('_drop')]

    print("Заполнение пропусков (ffill)...")
    df_combined = df_combined.ffill()

    initial_rows = len(df_combined)
    df_combined = df_combined.dropna(subset=['close_spot'])
    print(f"Удалено строк без spot: {initial_rows - len(df_combined):,}")
    print(f"Осталось строк: {len(df_combined):,}")

    critical_na = df_combined[['close_perp', 'close_spot', 'volume_perp']].isna().sum()
    if critical_na.sum() > 0:
        print("ВНИМАНИЕ! Остались критические пропуски:\n", critical_na)
    else:
        print("Критические поля без пропусков — OK")

    df_combined['basis'] = (df_combined['close_perp'] - df_combined['close_spot']) / df_combined['close_spot'].replace(0, np.nan)

    print("\nФинальные характеристики объединённого датасета:")
    print(f"  Строк: {len(df_combined):,}")
    print(f"  Период: {df_combined['ts'].min()} → {df_combined['ts'].max()}")
    print(f"  Пропуски в close_perp: {df_combined['close_perp'].isna().sum():,}")
    print(f"  Пропуски в basis: {df_combined['basis'].isna().sum():,}")
    print(f"  Монотонность ts: {'OK' if df_combined['ts'].is_monotonic_increasing else 'НАРУШЕНА!'}")

    df_combined.to_parquet(output_path, index=False)
    print(f"\nСохранено: {output_path}")
    print("Готово к добавлению признаков!")


if __name__ == "__main__":
    PERP_PATH = "get_data/output/main/btcusdt_5m_perp_API_clean.parquet"
    SPOT_PATH = "get_data/output/main/btcusdt_5m_spot_API_clean.parquet"
    OUTPUT_PATH = "get_data/output/main/btcusdt_5m_combined_2020-2026.parquet"
    run(PERP_PATH, SPOT_PATH, OUTPUT_PATH)
