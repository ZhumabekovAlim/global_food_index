# visualization.py
from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis import get_date_column


def plot_single_index(df: pd.DataFrame, column: str) -> None:
    """Линейный график одного индекса во времени."""
    date_col: Optional[str] = get_date_column(df)

    if date_col is None:
        print("Нельзя построить график: колонка с датой не найдена.")
        return

    fig = px.line(
        df,
        x=date_col,
        y=column,
        title=f"Динамика индекса: {column}",
        labels={date_col: "Дата", column: "Значение индекса"},
    )
    fig.show()


def plot_multiple_indices(df: pd.DataFrame, columns: List[str]) -> None:
    """График нескольких индексов на одном графике."""
    date_col: Optional[str] = get_date_column(df)

    if date_col is None:
        print("Нельзя построить график: колонка с датой не найдена.")
        return

    fig = go.Figure()

    for col in columns:
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[col],
                mode="lines",
                name=col,
            )
        )

    fig.update_layout(
        title="Сравнение динамики ценовых индексов",
        xaxis_title="Дата",
        yaxis_title="Индекс (условные единицы)",
    )

    fig.show()


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> None:
    """Тепловая карта корреляций между индексами."""
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        title="Корреляции между индексами",
        labels=dict(color="Корреляция"),
    )
    fig.update_layout(xaxis_title="Индекс", yaxis_title="Индекс")
    fig.show()


def plot_forecast(series: pd.Series, forecast_df: pd.DataFrame) -> None:
    """
    График временного ряда + прогноз (ML-модель).
    series: исторические значения с DatetimeIndex.
    forecast_df: DataFrame с одной колонкой 'forecast' и DatetimeIndex.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        print("Нельзя построить график прогноза: серия не имеет DatetimeIndex.")
        return

    series_clean = series.dropna()

    if "forecast" not in forecast_df.columns:
        print("В forecast_df не найдена колонка 'forecast'.")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=series_clean.index,
            y=series_clean.values,
            mode="lines",
            name="Фактические значения",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["forecast"].values,
            mode="lines+markers",
            name="Прогноз (ML-модель)",
        )
    )

    fig.update_layout(
        title=f"История и прогноз для ряда: {series.name or 'index'}",
        xaxis_title="Дата",
        yaxis_title="Значение индекса",
    )

    fig.show()
