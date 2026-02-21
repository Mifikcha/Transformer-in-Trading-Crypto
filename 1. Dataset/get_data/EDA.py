import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Пути (подставь свои)
INPUT_PATH = "get_data/output/_main/_final/btcusdt_5m_final_with_targets.parquet"
OUTPUT_PATH = "get_data/output/_main/_final/btcusdt_5m_final_cleaned.parquet"
EDA_DIR = "get_data/output/_main/_final/EDA_dir"

os.makedirs(EDA_DIR, exist_ok=True)

print("=== Полный EDA анализ датасета ===")

# Загрузка
df = pd.read_parquet(INPUT_PATH)
print("\n1. Общая информация:")
print(df.info())
print(f"\nShape: {df.shape}")
print("\nПервые 5 строк:")
print(df.head())
print("\nПоследние 5 строк:")
print(df.tail())

# Статистики
print("\n2. Описательные статистики:")
print(df.describe())

# Пропуски
print("\n3. Пропуски по колонкам:")
print(df.isna().sum())

# Дистрибуции ключевых колонок
key_cols = ['close_perp', 'volume_perp', 'fundingRate', 'openInterest', 'basis', 'target_return']
print("\n4. Дистрибуции (сохраняем графики в {EDA_DIR})")
for col in key_cols:
    if col in df.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Дистрибуция {col}")
        plt.savefig(os.path.join(EDA_DIR, f"hist_{col}.png"))
        plt.close()
        print(f"Сохранён: hist_{col}.png")

# Корреляции
print("\n5. Корреляции (heatmap)")
corr = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title("Корреляционная матрица")
plt.savefig(os.path.join(EDA_DIR, "corr_heatmap.png"))
plt.close()
print("Сохранён: corr_heatmap.png")

# Временные паттерны
print("\n6. Временные паттерны")
plt.figure(figsize=(12, 6))
df.set_index('ts')['close_perp'].plot()
plt.title("Цена close_perp во времени")
plt.savefig(os.path.join(EDA_DIR, "price_time.png"))
plt.close()
print("Сохранён: price_time.png")

plt.figure(figsize=(12, 6))
df.set_index('ts')['volume_perp'].plot()
plt.title("Объём volume_perp во времени")
plt.savefig(os.path.join(EDA_DIR, "volume_time.png"))
plt.close()
print("Сохранён: volume_time.png")

# Распределение классов
if 'target_class' in df.columns:
    print("\n7. Распределение target_class:")
    print(df['target_class'].value_counts(normalize=True) * 100)
    sns.countplot(x=df['target_class'])
    plt.title("Распределение классов таргета")
    plt.savefig(os.path.join(EDA_DIR, "target_class.png"))
    plt.close()
    print("Сохранён: target_class.png")

# Выбросы (boxplots)
print("\n8. Выбросы (boxplots)")
for col in key_cols:
    if col in df.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot {col} (выбросы)")
        plt.savefig(os.path.join(EDA_DIR, f"box_{col}.png"))
        plt.close()
        print(f"Сохранён: box_{col}.png")

print("\nEDA завершён.")

# === Очистка датасета ===
print("\n=== Очистка датасета ===")

initial_rows = len(df)

# 1. Клиппинг объёмов вместо удаления строк
print("Клиппинг экстремальных объёмов...")
vol_q995 = df['volume_perp'].quantile(0.995)  # 99.5% — мягче, чем 99.9%
print(f"99.5% квантиль объёма: {vol_q995:,.2f}")

df['volume_perp_clipped'] = df['volume_perp'].clip(upper=vol_q995)

# 2. Пересчёт объёмных статистик на клипнутых объёмах
print("Пересчёт объёмных статистик на клипнутых данных...")
v_clipped = df['volume_perp_clipped']

# Определяем окна (как в скрипте добавления объёмных статистик)
windows_min = [60, 240]
windows_steps = [round(m / 5) for m in windows_min]  # [12, 48]

for win_min, win_step in zip(windows_min, windows_steps):
    mean_col = f'rolling_vol_mean_{win_min}min'
    std_col = f'rolling_vol_std_{win_min}min'
    
    df[mean_col] = v_clipped.rolling(win_step).mean()
    df[std_col] = v_clipped.rolling(win_step).std()
    
    df[f'z_score_vol_{win_min}min'] = (v_clipped - df[mean_col]) / df[std_col].replace(0, np.nan)
    
    df[f'anomalous_vol_{win_min}min'] = (v_clipped > (df[mean_col] + 2 * df[std_col])).astype(int)
    df[f'anomalous_ratio_{win_min}min'] = df[f'anomalous_vol_{win_min}min'].rolling(win_step).mean()

# 3. Дроп первых 100 строк (оставляем)
print("Дроп первых 100 строк...")
df = df.iloc[100:].reset_index(drop=True)

# удаляем только строки без валидного таргета
df = df[df['is_valid_target'] == 1].reset_index(drop=True)


print(f"Удалено только первых строк: 100")
print(f"Осталось строк: {len(df):,}")
print("Пропуски после очистки:")
print(df.isna().sum())

# 5. Сохранение очищенного датасета
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nСохранено очищенный датасет: {OUTPUT_PATH}")
print("Очистка завершена.")