import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report  # для быстрой проверки


def run(input_path: str, output_path: str, c_scenarios: dict) -> None:
    # Загрузка обновлённого датасета из шага 2
    df = pd.read_csv(input_path, parse_dates=['ts'], index_col='ts')
    df.sort_index(inplace=True)

    print("Датасет загружен. Размер:", df.shape)
    print("Доступные колонки с доходностями:", [col for col in df.columns if 'delta_log' in col])

    # Основной горизонт
    h = 60
    ret_col = f'delta_log_{h}min'  # уже есть из шага 2
    future_col = f'future_close_{h}min'

    # Проверка: сколько NaN в конце (из-за shift -)
    print(f"NaN в {ret_col}: {df[ret_col].isna().sum()} (ожидаемо ~{int(60/5)} последних баров)")

    # === Базовый таргет: РЕГРЕССИЯ ===
    # Просто используем log-return как есть (непрерывный)
    df['base_regression'] = df[ret_col]  # уже готово, переименовываем для ясности

    # Статистики регрессионного таргета
    print("\nБазовый регрессионный таргет (log-return на 60 мин):")
    print(df['base_regression'].describe())

    # === Базовый таргет: КЛАССИФИКАЦИЯ ===
    # Порог для flat (маленький, чтобы учесть шум ~0.1–0.2%)
    threshold_flat = 0.001  # 0.1% — типичный шум на 60 мин для BTC

    df['base_class'] = 'flat'
    df.loc[df['base_regression'] > threshold_flat, 'base_class'] = 'up'
    df.loc[df['base_regression'] < -threshold_flat, 'base_class'] = 'down'

    # Распределение классов
    print("\nРаспределение базовых классов (up/down/flat):")
    class_dist = df['base_class'].value_counts(normalize=True) * 100
    print(class_dist.round(2).astype(str) + '%')

    # Визуализация распределения таргета (регрессия + классы)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Гистограмма log-return
    df['base_regression'].dropna().hist(bins=150, ax=ax1, color='teal', alpha=0.7)
    ax1.set_title(f'Базовый регрессионный таргет (log-return 60 мин)\nmean: {df["base_regression"].mean():.6f}, std: {df["base_regression"].std():.6f}')
    ax1.set_xlabel('Log-return')
    ax1.set_ylabel('Частота')
    ax1.grid(True, alpha=0.3)

    # Pie-chart классов
    class_dist.plot.pie(ax=ax2, autopct='%1.1f%%', colors=['lightgray', 'green', 'red'])
    ax2.set_title('Распределение базовых классов (up/down/flat)')
    ax2.set_ylabel('')

    plt.tight_layout()
    plt.show()

    # Быстрая проверка (опционально): если бы мы использовали случайный baseline
    print("\nПример classification_report для случайного предсказания (benchmark):")
    # Симулируем случайные предсказания для иллюстрации
    np.random.seed(42)
    random_preds = np.random.choice(['up', 'down', 'flat'], size=len(df.dropna(subset=['base_class'])), p=class_dist/100)
    print(classification_report(df['base_class'].dropna(), random_preds))

    # === Торгуемый таргет (на том же df, без промежуточного сохранения) ===
    # Торгуемый таргет для каждого сценария
    for name, c in c_scenarios.items():
        class_col = f'trading_class_{name}'
        df[class_col] = 'flat'
        df.loc[df['base_regression'] > c, class_col] = 'long'
        df.loc[df['base_regression'] < -c, class_col] = 'short'

        print(f"\n[{name} c = {c*100:.2f}%] Распределение торговых классов:")
        dist = df[class_col].value_counts(normalize=True) * 100
        print(dist.round(2).astype(str) + '%')
        print(f"Coverage (long + short): {dist.get('long', 0) + dist.get('short', 0):.2f}%")

    # Визуализация покрытия по сценариям
    coverage_data = {
        name: (df[f'trading_class_{name}'].value_counts(normalize=True).get('long', 0) +
               df[f'trading_class_{name}'].value_counts(normalize=True).get('short', 0)) * 100
        for name in c_scenarios
    }

    plt.figure(figsize=(8, 5))
    plt.bar(coverage_data.keys(), coverage_data.values(), color=['green', 'orange', 'red'])
    plt.title('Coverage (доля long/short) в зависимости от сценария издержек')
    plt.ylabel('Coverage (%)')
    plt.ylim(0, 50)
    for i, v in enumerate(coverage_data.values()):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # Сохранение финального датасета с базовыми и торговыми таргетами (один раз)
    df.to_csv(output_path, index=True)
    print(f"\nФинальный датасет с базовыми и торговыми таргетами сохранён как '{output_path}'")


if __name__ == "__main__":
    _c_scenarios = {
        'optimistic': 0.0010,
        'base':       0.0019,
        'pessimistic':0.0020
    }
    run('btcusdt_5m_with_horizons.csv', 'btcusdt_5m_final_with_targets.csv', _c_scenarios)
