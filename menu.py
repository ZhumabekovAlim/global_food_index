# menu.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from analysis import (
    compute_basic_stats,
    compute_correlation_matrix,
    compute_inflation,
    detect_high_correlations,
    get_numeric_columns,
)
from data_io import save_dataframe_to_csv, save_settings, load_settings
from ml_models import TSLinearModelResult, train_and_evaluate_ts_linear_model
from visualization import (
    plot_correlation_heatmap,
    plot_forecast,
    plot_multiple_indices,
    plot_single_index,
)


def print_main_menu() -> None:
    """Вывод основного меню на экран."""
    print("\n=== Global Food Price Index Analysis ===")
    print("1 - Показать список доступных индексов")
    print("2 - Посчитать базовую статистику по индексу")
    print("3 - Посчитать помесячную и годовую инфляцию")
    print("4 - Посчитать матрицу корреляций между индексами")
    print("5 - Построить график одного индекса")
    print("6 - Построить график нескольких индексов")
    print("7 - Сохранить последний результат в CSV")
    print("8 - ML: обучить простую модель прогноза и показать прогноз")
    print("0 - Выход")


def choose_index_column(numeric_cols: List[str]) -> Optional[str]:
    """Диалог для выбора одной числовой колонки."""
    if not numeric_cols:
        print("Числовые индексы не найдены.")
        return None

    print("\nДоступные индексы:")
    for i, col in enumerate(numeric_cols, start=1):
        print(f"{i}. {col}")

    while True:
        raw = input("Введите номер индекса (или пусто для отмены): ").strip()
        if raw == "":
            return None

        try:
            idx = int(raw)
            if 1 <= idx <= len(numeric_cols):
                return numeric_cols[idx - 1]
            else:
                print("Номер вне диапазона, попробуйте ещё раз.")
        except ValueError:
            print("Нужно ввести целое число.")


def choose_multiple_indices(numeric_cols: List[str]) -> List[str]:
    """
    Диалог выбора нескольких индексов.
    Пример ввода: 1,3,4
    """
    if not numeric_cols:
        print("Числовые индексы не найдены.")
        return []

    print("\nДоступные индексы:")
    for i, col in enumerate(numeric_cols, start=1):
        print(f"{i}. {col}")

    raw = input("Введите номера индексов через запятую (или пусто для отмены): ").strip()
    if raw == "":
        return []

    result: List[str] = []
    parts = [part.strip() for part in raw.split(",") if part.strip()]

    for part in parts:
        try:
            idx = int(part)
            if 1 <= idx <= len(numeric_cols):
                result.append(numeric_cols[idx - 1])
            else:
                print(f"Номер {idx} пропущен: вне диапазона.")
        except ValueError:
            print(f"'{part}' не является числом и будет проигнорирован.")

    # удаляем дубликаты, сохраняем порядок
    unique: List[str] = []
    for col in result:
        if col not in unique:
            unique.append(col)

    return unique


class AppState:
    """
    Простое состояние приложения:
    - df: исходный DataFrame;
    - last_result: последний результат анализа/ML (для сохранения);
    - settings: настройки (например, последний выбранный индекс).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.last_result: Optional[pd.DataFrame] = None
        self.settings: Dict[str, Any] = load_settings()


def handle_show_indices(state: AppState) -> None:
    cols = get_numeric_columns(state.df)
    if not cols:
        print("Числовых индексов не найдено.")
        return
    print("\nЧисловые индексы в датасете:")
    for col in cols:
        print("-", col)


def handle_basic_stats(state: AppState) -> None:
    cols = get_numeric_columns(state.df)
    col = choose_index_column(cols)
    if col is None:
        return

    stats_df = compute_basic_stats(state.df, col)
    state.last_result = stats_df
    print("\nБазовая статистика:")
    print(stats_df.to_string(index=False))


def handle_inflation(state: AppState) -> None:
    cols = get_numeric_columns(state.df)
    col = choose_index_column(cols)
    if col is None:
        return

    infl_df = compute_inflation(state.df, col)
    state.last_result = infl_df

    print("\nПервые 12 строк с инфляцией:")
    cols_to_show = [c for c in infl_df.columns if col in c]
    print(infl_df[[*infl_df.columns[:1], *cols_to_show]].head(12).to_string(index=False))

    state.settings["last_index_column"] = col
    save_settings(state.settings)


def handle_correlations(state: AppState) -> None:
    cols = get_numeric_columns(state.df)
    chosen = choose_multiple_indices(cols)
    if len(chosen) < 2:
        print("Нужно выбрать хотя бы два индекса.")
        return

    corr = compute_correlation_matrix(state.df, chosen)
    state.last_result = corr

    print("\nМатрица корреляций:")
    print(corr.to_string())

    messages = detect_high_correlations(corr)
    if messages:
        print("\nПары с высокой корреляцией (|r| >= 0.7):")
        for msg in messages:
            print("-", msg)
    else:
        print("\nСильных корреляций не обнаружено.")

    plot_correlation_heatmap(corr)


def handle_plot_single(state: AppState) -> None:
    cols = get_numeric_columns(state.df)
    col = choose_index_column(cols)
    if col is None:
        return

    plot_single_index(state.df, col)


def handle_plot_multiple(state: AppState) -> None:
    cols = get_numeric_columns(state.df)
    chosen = choose_multiple_indices(cols)
    if len(chosen) < 2:
        print("Нужно выбрать хотя бы два индекса.")
        return

    plot_multiple_indices(state.df, chosen)


def handle_save_last_result(state: AppState) -> None:
    if state.last_result is None:
        print("Нет последнего результата для сохранения.")
        return

    filename = input(
        "Введите имя файла для сохранения (например, result.csv или tables/stats.csv): "
    ).strip()
    if not filename:
        print("Имя файла не может быть пустым.")
        return

    save_dataframe_to_csv(state.last_result, filename)


def handle_ts_forecast(state: AppState) -> None:
    """
    ML-блок: обучаем простую линейную авторегрессионную модель
    и строим прогноз на несколько месяцев вперёд.
    """
    cols = get_numeric_columns(state.df)
    col = choose_index_column(cols)
    if col is None:
        return

    series = state.df[col]
    series.name = col  # для красивого названия на графике

    try:
        result, test_df, forecast_df = train_and_evaluate_ts_linear_model(series)
    except ValueError as e:
        print("Не удалось обучить модель:", e)
        return
    except Exception as e:
        print("Во время обучения модели произошла ошибка:")
        print(e)
        return

    # сохраняем прогноз как последний результат (можно сохранить в CSV)
    state.last_result = forecast_df

    print("\n=== Результаты ML-модели (AR-линейная регрессия) ===")
    print(f"Индекс: {col}")
    print(f"Использовано лагов: {result.n_lags}")
    print(f"Train size: {result.train_size}")
    print(f"Test size:  {result.test_size}")
    print(f"MAE train: {result.train_mae:.3f}")
    print(f"MAE test:  {result.test_mae:.3f}")
    print(f"MAPE train: {result.train_mape:.2f}%")
    print(f"MAPE test:  {result.test_mape:.2f}%")

    print("\nПримеры предсказаний (конец test-выборки):")
    print(test_df.tail(10).to_string())

    print("\nПрогноз на следующие месяцы:")
    print(forecast_df.to_string())

    # График: история + прогноз
    plot_forecast(series, forecast_df)


def run_menu_loop(state: AppState) -> None:
    """Основной цикл меню."""
    actions: Dict[str, Callable[[AppState], None]] = {
        "1": handle_show_indices,
        "2": handle_basic_stats,
        "3": handle_inflation,
        "4": handle_correlations,
        "5": handle_plot_single,
        "6": handle_plot_multiple,
        "7": handle_save_last_result,
        "8": handle_ts_forecast,
    }

    while True:
        print_main_menu()
        choice = input("Выберите пункт меню: ").strip()

        if choice == "0":
            print("Выход из программы. Пока!")
            break

        action = actions.get(choice)
        if action is None:
            print("Неизвестный пункт меню. Попробуйте ещё раз.")
            continue

        try:
            action(state)
        except Exception as e:
            # демонстрация обработки ошибок (требование курса)
            print("Во время выполнения действия произошла ошибка:")
            print(e)
