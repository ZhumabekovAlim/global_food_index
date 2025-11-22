# main.py
from __future__ import annotations

from data_io import ensure_directories, load_fao_data
from menu import AppState, run_menu_loop


def main() -> None:
    ensure_directories()

    df = load_fao_data()
    if df is None:
        print("Данные не загружены. Завершение программы.")
        return

    state = AppState(df)
    run_menu_loop(state)


if __name__ == "__main__":
    main()

