# analysis.py
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def get_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Возвращает имя колонки с датой, если она приведена к datetime.
    Ищем по ключевым словам: date/month/time/period.
    """
    for col in df.columns:
        name = col.lower()
        if any(key in name for key in ("date", "month", "time", "period")):
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
    return None


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Возвращает список числовых колонок (индексы цен)."""
    return df.select_dtypes(include="number").columns.tolist()


def compute_basic_stats(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Описательная статистика по одному индексу:
    count, mean, median, min, max, std.
    """
    series = df[column].dropna()
    data = {
        "Показатель": [
            "Количество наблюдений",
            "Среднее",
            "Медиана",
            "Минимум",
            "Максимум",
            "Стандартное отклонение",
        ],
        "Значение": [
            int(series.count()),
            float(series.mean()),
            float(series.median()),
            float(series.min()),
            float(series.max()),
            float(series.std(ddof=1)),
        ],
    }
    return pd.DataFrame(data)


def compute_inflation(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Рассчитывает:
    - помесячное изменение (% к предыдущему месяцу),
    - годовое изменение (% к тому же месяцу прошлого года).

    Возвращает копию DataFrame с двумя новыми колонками:
    column_mom_pct, column_yoy_pct.
    """
    result = df.copy()
    result[f"{column}_mom_pct"] = result[column].pct_change(periods=1) * 100
    result[f"{column}_yoy_pct"] = result[column].pct_change(periods=12) * 100
    return result


def compute_correlation_matrix(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Строит матрицу корреляций Пирсона между выбранными индексами.
    """
    return df[columns].corr(method="pearson")


def detect_high_correlations(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
) -> List[str]:
    """
    Возвращает список строк с парами индексов,
    у которых |корреляция| >= threshold.
    """
    messages: List[str] = []
    cols = list(corr_matrix.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_i = cols[i]
            col_j = cols[j]
            value = corr_matrix.loc[col_i, col_j]
            if np.abs(value) >= threshold:
                messages.append(f"{col_i} и {col_j}: корреляция {value:.2f}")

    return messages
