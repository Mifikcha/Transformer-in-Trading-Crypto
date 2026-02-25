import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def clean_final_dataset(input_path: str, output_path: str) -> None:
    """
    Clean final dataset: clip extreme volumes, recalc volume stats,
    drop first 100 rows, keep only rows with valid target.
    """
    print("=== Очистка датасета ===")
    df = pd.read_parquet(input_path)
    initial_rows = len(df)

    print("Клиппинг экстремальных объёмов...")
    vol_q995 = df["volume_perp"].quantile(0.995)
    print(f"99.5% квантиль объёма: {vol_q995:,.2f}")
    df["volume_perp_clipped"] = df["volume_perp"].clip(upper=vol_q995)

    print("Пересчёт объёмных статистик на клипнутых данных...")
    v_clipped = df["volume_perp_clipped"]
    windows_min = [60, 240]
    windows_steps = [round(m / 5) for m in windows_min]
    for win_min, win_step in zip(windows_min, windows_steps):
        mean_col = f"rolling_vol_mean_{win_min}min"
        std_col = f"rolling_vol_std_{win_min}min"
        df[mean_col] = v_clipped.rolling(win_step).mean()
        df[std_col] = v_clipped.rolling(win_step).std()
        df[f"z_score_vol_{win_min}min"] = (v_clipped - df[mean_col]) / df[std_col].replace(0, np.nan)
        df[f"anomalous_vol_{win_min}min"] = (v_clipped > (df[mean_col] + 2 * df[std_col])).astype(int)
        df[f"anomalous_ratio_{win_min}min"] = df[f"anomalous_vol_{win_min}min"].rolling(win_step).mean()

    print("Дроп первых 100 строк...")
    df = df.iloc[100:].reset_index(drop=True)
    df = df[df["is_valid_target"] == 1].reset_index(drop=True)

    print(f"Удалено первых строк: 100")
    print(f"Осталось строк: {len(df):,}")
    print("Пропуски после очистки:")
    print(df.isna().sum())
    df.to_parquet(output_path, index=False)
    print(f"\nСохранено очищенный датасет: {output_path}")
    print("Очистка завершена.")


def run_eda(input_path: str, eda_dir: str) -> None:
    """Run exploratory data analysis: stats, histograms, correlations, time series, boxplots. Saves plots to eda_dir."""
    os.makedirs(eda_dir, exist_ok=True)
    print("=== Полный EDA анализ датасета ===")
    df = pd.read_parquet(input_path)

    print("\n1. Общая информация:")
    print(df.info())
    print(f"\nShape: {df.shape}")
    print("\nПервые 5 строк:")
    print(df.head())
    print("\nПоследние 5 строк:")
    print(df.tail())
    print("\n2. Описательные статистики:")
    print(df.describe())
    print("\n3. Пропуски по колонкам:")
    print(df.isna().sum())

    key_cols = ["close_perp", "volume_perp", "fundingRate", "openInterest", "basis", "target_return"]
    print(f"\n4. Дистрибуции (сохраняем графики в {eda_dir})")
    for col in key_cols:
        if col in df.columns:
            plt.figure(figsize=(10, 5))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Дистрибуция {col}")
            plt.savefig(os.path.join(eda_dir, f"hist_{col}.png"))
            plt.close()
            print(f"Сохранён: hist_{col}.png")

    print("\n5. Корреляции (heatmap)")
    corr = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Корреляционная матрица")
    plt.savefig(os.path.join(eda_dir, "corr_heatmap.png"))
    plt.close()
    print("Сохранён: corr_heatmap.png")

    print("\n6. Временные паттерны")
    plt.figure(figsize=(12, 6))
    df.set_index("ts")["close_perp"].plot()
    plt.title("Цена close_perp во времени")
    plt.savefig(os.path.join(eda_dir, "price_time.png"))
    plt.close()
    print("Сохранён: price_time.png")
    plt.figure(figsize=(12, 6))
    df.set_index("ts")["volume_perp"].plot()
    plt.title("Объём volume_perp во времени")
    plt.savefig(os.path.join(eda_dir, "volume_time.png"))
    plt.close()
    print("Сохранён: volume_time.png")

    if "target_class" in df.columns:
        print("\n7. Распределение target_class:")
        print(df["target_class"].value_counts(normalize=True) * 100)
        plt.figure(figsize=(8, 5))
        sns.countplot(x=df["target_class"])
        plt.title("Распределение классов таргета")
        plt.savefig(os.path.join(eda_dir, "target_class.png"))
        plt.close()
        print("Сохранён: target_class.png")

    print("\n8. Выбросы (boxplots)")
    for col in key_cols:
        if col in df.columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=df[col].dropna())
            plt.title(f"Boxplot {col} (выбросы)")
            plt.savefig(os.path.join(eda_dir, f"box_{col}.png"))
            plt.close()
            print(f"Сохранён: box_{col}.png")
    print("\nEDA завершён.")


if __name__ == "__main__":
    INPUT_PATH = "get_data/output/_main/_final/btcusdt_5m_final_with_targets.parquet"
    OUTPUT_PATH = "get_data/output/_main/_final/btcusdt_5m_final_cleaned.parquet"
    EDA_DIR = "get_data/output/_main/_final/EDA_dir"
    clean_final_dataset(INPUT_PATH, OUTPUT_PATH)
    run_eda(OUTPUT_PATH, EDA_DIR)
