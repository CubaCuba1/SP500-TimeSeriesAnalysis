"""
download_data.py — загрузка и сохранение данных S&P 500 в формате Nixtla.

Запуск:
    python data/download_data.py

Результат:
    data/sp500_daily.csv     — исходные дневные цены закрытия
    data/sp500_prepared.csv  — подготовленный ряд (unique_id, ds, y)
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent
START    = "2018-01-01"
END      = "2023-12-31"
TICKER   = "^GSPC"


def download() -> pd.DataFrame:
    print(f"Загрузка {TICKER} с {START} по {END} ...")
    raw = yf.download(TICKER, start=START, end=END, progress=True)
    prices = raw["Close"].reset_index()
    prices.columns = ["ds", "y"]
    prices["ds"] = pd.to_datetime(prices["ds"]).dt.tz_localize(None)
    prices = prices.sort_values("ds").reset_index(drop=True)
    return prices


def main() -> None:
    df = download()

    # Сырые данные
    raw_path = DATA_DIR / "sp500_daily.csv"
    df.to_csv(raw_path, index=False)
    print(f"Сохранено: {raw_path}  ({len(df)} строк)")

    # Формат Nixtla (unique_id, ds, y)
    df_prepared = df.copy()
    df_prepared["unique_id"] = "sp500"
    df_prepared = df_prepared[["unique_id", "ds", "y"]]
    prep_path = DATA_DIR / "sp500_prepared.csv"
    df_prepared.to_csv(prep_path, index=False)
    print(f"Сохранено: {prep_path}  ({len(df_prepared)} строк)")

    print("\nСтатистики ряда:")
    print(df["y"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
