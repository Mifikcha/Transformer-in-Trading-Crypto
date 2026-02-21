import pandas as pd
import numpy as np


def run(
    input_path: str,
    output_path: str,
    h_minutes: int = 60,
    delta_t: int = 5,
    cost: float = 0.0005,
) -> None:
    """Add target_return, target_class, is_valid_target."""
    H_STEPS = h_minutes // delta_t

    print("=== Добавление таргетов ===")
    df = pd.read_parquet(input_path)
    print(f"Загружено строк: {len(df):,}")
    print(f"Период: {df['ts'].min()} → {df['ts'].max()}")

    c = df["close_perp"]
    df["target_return"] = np.log(c.shift(-H_STEPS) / c)
    df["target_class"] = np.where(
        df["target_return"] > cost,
        "long",
        np.where(df["target_return"] < -cost, "short", "flat"),
    )
    df["is_valid_target"] = df["target_return"].notna().astype(int)

    target_cols = ["target_return", "target_class", "is_valid_target"]
    print("\nПроверка новых таргетов (последние 5 строк, где valid):")
    print(df[df["is_valid_target"] == 1][target_cols].tail(5))
    print("\nРаспределение классов:")
    print(df["target_class"].value_counts(normalize=True) * 100)
    print("\nПропуски в таргетах:")
    print(df[target_cols].isna().sum())

    df.to_parquet(output_path, index=False)
    print(f"\nСохранено: {output_path}")
    print("Таргеты добавлены. Датасет готов к моделям!")


if __name__ == "__main__":
    INPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_derivatives.parquet"
    OUTPUT_PATH = "get_data/output/_main/_final/btcusdt_5m_final_with_targets.parquet"
    run(INPUT_PATH, OUTPUT_PATH, h_minutes=60, delta_t=5, cost=0.0005)
