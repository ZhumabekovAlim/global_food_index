# data_io.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import pandas as pd

# Базовые пути
BASE_DATA_DIR = "data"
RESULTS_DIR = "results"
DEFAULT_DATA_PATH = os.path.join(BASE_DATA_DIR, "fao_food_price_index.csv")
SETTINGS_PATH = "settings.json"


def ensure_directories() -> None:
    """
    Гарантирует наличие служебных директорий:
    - data/  (для входных данных)
    - results/ (для результатов анализа и ML).
    """
    for directory in (BASE_DATA_DIR, RESULTS_DIR):
        os.makedirs(directory, exist_ok=True)


def load_settings() -> Dict[str, Any]:
    """
    Загрузка настроек приложения (например, последний выбранный индекс)
    из JSON-файла. Если файл не найден или повреждён — возвращает пустой dict.
    """
    if not os.path.exists(SETTINGS_PATH):
        return {}

    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_settings(settings: Dict[str, Any]) -> None:
    """Сохранение настроек в JSON-файл."""
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print("Не удалось сохранить настройки:", e)


def load_fao_data(path: str = DEFAULT_DATA_PATH) -> Optional[pd.DataFrame]:
    """
    Загрузка CSV с индексами FAO.
    Делает:
    - проверку наличия файла;
    - чтение CSV;
    - нормализацию имён колонок (в нижний регистр, вместо пробелов — '_');
    - попытку найти колонку с датой;
    - сортировку по времени;
    - выделение только числовых колонок + даты.
    """
    ensure_directories()

    if not os.path.exists(path):
        print(f"Файл с данными не найден: {path}")
        print("Пожалуйста, скачайте CSV с сайта FAO и сохраните его по этому пути.")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("Ошибка при чтении CSV:", e)
        return None

    # нормализуем имена колонок
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # выбираем колонку с датой
    date_col: Optional[str] = None
    for candidate in ("date", "month", "time", "period"):
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.sort_values(by=date_col)
    else:
        print(
            "Предупреждение: колонка с датой не найдена. "
            "Графики и временные модели могут работать некорректно."
        )

    # только числовые столбцы
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if date_col:
        cols = [date_col] + numeric_cols
        df = df[cols]
    else:
        df = df[numeric_cols]

    return df


def get_results_path(filename: str) -> str:
    """
    Если пользователь ввёл только имя файла (без пути),
    сохраняем его в директорию results/.
    Если указан путь с директорией — используем его как есть.
    """
    if os.path.dirname(filename):
        return filename
    return os.path.join(RESULTS_DIR, filename)


def save_dataframe_to_csv(df: pd.DataFrame, filename: str) -> None:
    """
    Сохранение DataFrame в CSV.
    Файл кладётся в results/, если не указана другая директория.
    """
    ensure_directories()
    path = get_results_path(filename)

    try:
        df.to_csv(path, index=False)
        print(f"Файл успешно сохранён: {path}")
    except Exception as e:
        print("Ошибка при сохранении CSV:", e)
