import pandas as pd
import numpy as np

# Пути (подставь свои)
INPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_price_derivatives.parquet"
OUTPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_volatility.parquet"

print("=== Добавление оценок волатильности ===")

# 1. Загрузка
df = pd.read_parquet(INPUT_PATH)
print(f"Загружено строк: {len(df):,}")
print(f"Период: {df['ts'].min()} → {df['ts'].max()}")

# Основные колонки для расчётов
c = df['close_perp']
h = df['high_perp']
l = df['low_perp']

# 2. Окна в минутах (как в плане и твоём описании)
windows_min = [15, 60, 240]
windows_steps = [round(m / 5) for m in windows_min]  # [3, 12, 48]

# 3. Дельта средних скользящих (SMA.diff())
print("Расчёт SMA и их дельты...")
for win_min, win_step in zip(windows_min, windows_steps):
    sma = c.rolling(win_step).mean()
    df[f'sma_{win_min}min'] = sma
    df[f'delta_sma_{win_min}min'] = sma.diff()

# 4. ATR (14-периодный, стандартный)
print("Расчёт ATR...")
tr = np.maximum(
    h - l,
    np.maximum(
        abs(h - c.shift()),
        abs(l - c.shift())
    )
)
df['atr_14'] = tr.rolling(14).mean()

# 5. Реализованная волатильность
# Используем log_return_5min (уже есть из предыдущего шага)
log_ret_5 = df['log_return_5min']

print("Расчёт realized volatility...")
for win_min, win_step in zip(windows_min, windows_steps):
    # sqrt( sum( log_return^2 ) за окно ) → аннуализировать не будем (пока), оставим как есть
    rv = np.sqrt((log_ret_5**2).rolling(win_step).sum())
    df[f'realized_vol_{win_min}min'] = rv

# 6. Проверки
print("\nПроверка новых признаков (последние 5 строк):")
vol_cols = [c for c in df.columns if 'sma' in c or 'delta_sma' in c or 'atr' in c or 'realized_vol' in c]
print(df[vol_cols].tail(5))

print("\nПропуски в новых признаках:")
print(df[vol_cols].isna().sum())

# 7. Сохранение
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nСохранено: {OUTPUT_PATH}")
print("Оценки волатильности добавлены. Готово к объёмным статистикам или funding/OI.")