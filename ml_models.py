# ml_models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas import DateOffset


@dataclass
class TSLinearModelResult:
    """
    Результат обучения простой линейной авторегрессионной модели.
    """
    coef_: np.ndarray         # веса при лагах
    intercept_: float         # свободный член
    n_lags: int               # количество лагов
    train_mae: float
    test_mae: float
    train_mape: float
    test_mape: float
    train_size: int
    test_size: int


@dataclass
class TSForecastModelResult:
    """
    Результат простой прогностической модели без обучения сложных весов.
    Используется для сравнения бейзлайнов (наивный, скользящее среднее и т.д.).
    """

    model_name: str
    train_mae: float
    test_mae: float
    train_mape: float
    test_mape: float
    train_size: int
    test_size: int


def _create_lagged_dataset(
    series: pd.Series,
    n_lags: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Формирует матрицу признаков X и целевой вектор y для авторегрессии.
    X_t = [y_{t-1}, ..., y_{t-n_lags}], y_t = y_t.

    Возвращает:
    - X: (N, n_lags)
    - y: (N,)
    - dates: index длины N, соответствующий элементам y.
    """
    series_clean = series.dropna()
    values = series_clean.values.astype(float)
    dates = series_clean.index

    X_list = []
    y_list = []
    y_dates = []

    for t in range(n_lags, len(values)):
        X_list.append(values[t - n_lags : t])
        y_list.append(values[t])
        y_dates.append(dates[t])

    if not X_list:
        raise ValueError("Недостаточно данных для построения лаговой выборки.")

    X = np.array(X_list)
    y = np.array(y_list)
    date_index = pd.DatetimeIndex(y_dates)

    return X, y, date_index


def _fit_linear_regression(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Обучает линейную регрессию через МНК:
    добавляем столбец единиц и считаем решение задачи least squares.
    """
    # добавляем столбец единиц для интерсепта
    X_ext = np.c_[X, np.ones(X.shape[0])]
    beta, *_ = np.linalg.lstsq(X_ext, y, rcond=None)
    coef_ = beta[:-1]
    intercept_ = float(beta[-1])
    return coef_, intercept_


def _predict_linear_regression(
    X: np.ndarray,
    coef_: np.ndarray,
    intercept_: float,
) -> np.ndarray:
    """Предсказания линейной модели."""
    return X @ coef_ + intercept_


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _split_train_test(
    y_true: np.ndarray, y_pred: np.ndarray, train_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает train/test части предсказаний и фактов по временной метке."""

    n_samples = len(y_true)
    split_idx = int(n_samples * train_ratio)

    if split_idx <= 0:
        split_idx = 1
    elif split_idx >= n_samples:
        split_idx = n_samples - 1

    return (
        y_true[:split_idx],
        y_pred[:split_idx],
        y_true[split_idx:],
        y_pred[split_idx:],
    )


def train_and_evaluate_ts_linear_model(
    series: pd.Series,
    n_lags: int = 6,
    train_ratio: float = 0.8,
    n_forecast_steps: int = 6,
) -> Tuple[TSLinearModelResult, pd.DataFrame, pd.DataFrame]:
    """
    Обучает авторегрессионную линейную модель на одном индексе.

    Параметры:
    - series: временной ряд (индекс должен быть DatetimeIndex).
    - n_lags: сколько последних значений использовать как признаки.
    - train_ratio: доля выборки в train.
    - n_forecast_steps: сколько месяцев вперёд прогнозировать.

    Возвращает:
    - result: TSLinearModelResult (веса, метрики, размеры выборок);
    - test_df: DataFrame с фактическими и предсказанными значениями на тесте;
    - forecast_df: DataFrame с прогнозом на n_forecast_steps вперёд.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Ожидается DatetimeIndex для серии (нужен временной ряд).")

    if series.dropna().shape[0] < n_lags + 10:
        raise ValueError("Слишком мало данных для обучения модели.")

    X, y, dates = _create_lagged_dataset(series, n_lags=n_lags)

    # разбиение на train/test по времени
    n_samples = len(y)
    split_idx = int(n_samples * train_ratio)

    if split_idx <= 0 or split_idx >= n_samples:
        # на всякий случай, если train_ratio задан некорректно
        split_idx = max(1, n_samples - 1)

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    dates_train = dates[:split_idx]

    X_test = X[split_idx:]
    y_test = y[split_idx:]
    dates_test = dates[split_idx:]

    coef_, intercept_ = _fit_linear_regression(X_train, y_train)

    y_train_pred = _predict_linear_regression(X_train, coef_, intercept_)
    y_test_pred = _predict_linear_regression(X_test, coef_, intercept_)

    train_mae = _mae(y_train, y_train_pred)
    test_mae = _mae(y_test, y_test_pred)
    train_mape = _mape(y_train, y_train_pred)
    test_mape = _mape(y_test, y_test_pred)

    result = TSLinearModelResult(
        coef_=coef_,
        intercept_=intercept_,
        n_lags=n_lags,
        train_mae=train_mae,
        test_mae=test_mae,
        train_mape=train_mape,
        test_mape=test_mape,
        train_size=len(y_train),
        test_size=len(y_test),
    )

    test_df = pd.DataFrame(
        {
            "actual": y_test,
            "predicted": y_test_pred,
        },
        index=dates_test,
    )

    # --- Прогноз вперёд ---
    series_clean = series.dropna()
    last_values = series_clean.values.astype(float).tolist()

    if len(last_values) < n_lags:
        raise ValueError("Недостаточно значений ряда для прогноза.")

    last_date = series_clean.index[-1]

    future_dates = [
        last_date + DateOffset(months=i + 1) for i in range(n_forecast_steps)
    ]

    future_preds = []
    for _ in range(n_forecast_steps):
        x_input = np.array(last_values[-n_lags:])
        y_hat = float(x_input @ coef_ + intercept_)
        future_preds.append(y_hat)
        last_values.append(y_hat)

    forecast_df = pd.DataFrame(
        {"forecast": future_preds},
        index=pd.DatetimeIndex(future_dates),
    )

    return result, test_df, forecast_df


def compare_forecast_baselines(
    series: pd.Series,
    train_ratio: float = 0.8,
    n_forecast_steps: int = 6,
    ma_window: int = 3,
    season_length: int = 12,
    ses_alpha: float = 0.3,
) -> Tuple[List[TSForecastModelResult], pd.DataFrame]:
    """
    Сравнение нескольких простых временных моделей:
    - Наивный прогноз (копирование последнего значения);
    - Скользящее среднее с окном ma_window;
    - Сезонный наивный прогноз (значение год назад);
    - Простое экспоненциальное сглаживание (SES).

    Возвращает список результатов и DataFrame с прогнозами на n_forecast_steps.
    """

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Ожидается DatetimeIndex для серии (нужен временной ряд).")

    series_clean = series.dropna().astype(float)
    values = series_clean.values
    if len(values) < max(ma_window, season_length) + 5:
        raise ValueError("Недостаточно данных для сравнения моделей.")

    # индекс, с которого можем строить предсказания (чтобы хватило истории)
    start_idx = max(1, ma_window, season_length)
    y_true = values[start_idx:]

    predictions: Dict[str, np.ndarray] = {}

    # 1) Наивный прогноз: копируем предыдущее значение
    naive_pred = values[start_idx - 1 : -1]
    predictions["Наивный (last value)"] = naive_pred

    # 2) Скользящее среднее
    ma_preds: List[float] = []
    for t in range(start_idx, len(values)):
        ma_preds.append(float(np.mean(values[t - ma_window : t])))
    predictions[f"Скользящее среднее {ma_window}м"] = np.array(ma_preds)

    # 3) Сезонный наивный (берём значение год назад)
    seasonal_preds: List[float] = []
    for t in range(start_idx, len(values)):
        seasonal_preds.append(float(values[t - season_length]))
    predictions[f"Сезонный наивный L={season_length}"] = np.array(seasonal_preds)

    # 4) Простое экспоненциальное сглаживание (SES)
    ses_levels = [values[start_idx - 1]]
    for t in range(start_idx, len(values)):
        level = ses_alpha * values[t - 1] + (1 - ses_alpha) * ses_levels[-1]
        ses_levels.append(level)
    ses_pred = np.array(ses_levels[1:])  # пропускаем уровень, соответствующий старту
    predictions[f"SES α={ses_alpha}"] = ses_pred

    results: List[TSForecastModelResult] = []

    for model_name, y_pred in predictions.items():
        y_train, y_pred_train, y_test, y_pred_test = _split_train_test(
            y_true, y_pred, train_ratio
        )

        result = TSForecastModelResult(
            model_name=model_name,
            train_mae=_mae(y_train, y_pred_train),
            test_mae=_mae(y_test, y_pred_test),
            train_mape=_mape(y_train, y_pred_train),
            test_mape=_mape(y_test, y_pred_test),
            train_size=len(y_train),
            test_size=len(y_test),
        )
        results.append(result)

    # --- Прогноз на будущее для каждой модели ---
    last_values = values.tolist()
    forecasts: Dict[str, List[float]] = {name: [] for name in predictions}

    for step in range(n_forecast_steps):
        # Наивный
        forecasts["Наивный (last value)"].append(float(last_values[-1]))

        # Скользящее среднее
        forecasts[f"Скользящее среднее {ma_window}м"].append(
            float(np.mean(last_values[-ma_window:]))
        )

        # Сезонный наивный
        if len(last_values) >= season_length:
            forecasts[f"Сезонный наивный L={season_length}"].append(
                float(last_values[-season_length])
            )
        else:
            forecasts[f"Сезонный наивный L={season_length}"].append(float(last_values[-1]))

        # SES: используем последний уровень как прогноз
        last_level = ses_alpha * last_values[-1] + (1 - ses_alpha) * ses_levels[-1]
        ses_levels.append(last_level)
        forecasts[f"SES α={ses_alpha}"].append(float(last_level))

        # добавляем одно из прогнозных значений в историю, чтобы последующие шаги
        # зависели от предыдущего прогноза (выбираем наивный как нейтральный вариант)
        last_values.append(forecasts["Наивный (last value)"][-1])

    future_dates = [
        series_clean.index[-1] + DateOffset(months=i + 1)
        for i in range(n_forecast_steps)
    ]

    forecast_df = pd.DataFrame(forecasts, index=pd.DatetimeIndex(future_dates))
    return results, forecast_df
