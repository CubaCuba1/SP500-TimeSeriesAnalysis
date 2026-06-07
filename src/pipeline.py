"""
Финальный пайплайн прогнозирования цен S&P 500.

Загружает очищенный временной ряд, обучает выбранную модель на всей истории
и сохраняет прогноз на заданный горизонт в CSV. Дополнительно замеряет
время обучения и инференса.

Доступные модели:
    - snaive    : SeasonalNaive (period=5) — недельный бейзлайн, очень быстрый
    - autoarima : AutoARIMA (d=1, season=5) — автоматический подбор ARIMA
    - autotheta : AutoTheta (season=5) — победитель M3/M4, робастен к выбросам
    - lgbm      : LightGBM через mlforecast с лагами и скользящими статистиками

Запуск из корня проекта:
    python src/pipeline.py --model lgbm --horizon 10

Результат — файл outputs/forecast.csv со столбцами unique_id, ds, y_hat.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT  = PROJECT_ROOT / "data" / "sp500_prepared.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "forecast.csv"

FREQ   = "B"   # рабочие дни
SEASON = 5     # рабочая неделя


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка данных
# ─────────────────────────────────────────────────────────────────────────────

def load_series(path: Path) -> pd.DataFrame:
    """Читает ряд в формате Nixtla (unique_id, ds, y)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл {path}.\n"
            f"Сначала загрузите данные: python data/download_data.py"
        )
    df = pd.read_csv(path, parse_dates=["ds"])
    required = {"unique_id", "ds", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В файле отсутствуют столбцы: {missing}")
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Модели
# ─────────────────────────────────────────────────────────────────────────────

def fit_predict_snaive(
    df: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, float, float]:
    """SeasonalNaive: повторяет последнюю рабочую неделю."""
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive

    sf = StatsForecast(
        models=[SeasonalNaive(season_length=SEASON, alias="SNaive")],
        freq=FREQ,
        n_jobs=1,
    )
    t0 = time.perf_counter()
    sf.fit(df=df)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = sf.predict(h=horizon)
    pred_time = time.perf_counter() - t0

    pred = pred.rename(columns={"SNaive": "y_hat"})[["unique_id", "ds", "y_hat"]]
    return pred, fit_time, pred_time


def fit_predict_autoarima(
    df: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, float, float]:
    """AutoARIMA: автоматический подбор p, q при d=1."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    sf = StatsForecast(
        models=[AutoARIMA(season_length=SEASON, d=1, max_p=3, max_q=3,
                          information_criterion="aicc", alias="AutoARIMA")],
        freq=FREQ,
        n_jobs=1,
    )
    t0 = time.perf_counter()
    sf.fit(df=df)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = sf.predict(h=horizon, level=[80, 95])
    pred_time = time.perf_counter() - t0

    pred = pred.rename(columns={"AutoARIMA": "y_hat"})[["unique_id", "ds", "y_hat"]]
    return pred, fit_time, pred_time


def fit_predict_autotheta(
    df: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, float, float]:
    """AutoTheta: лидер M3/M4, робастен к выбросам."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoTheta

    sf = StatsForecast(
        models=[AutoTheta(season_length=SEASON, alias="AutoTheta")],
        freq=FREQ,
        n_jobs=1,
    )
    t0 = time.perf_counter()
    sf.fit(df=df)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = sf.predict(h=horizon)
    pred_time = time.perf_counter() - t0

    pred = pred.rename(columns={"AutoTheta": "y_hat"})[["unique_id", "ds", "y_hat"]]
    return pred, fit_time, pred_time


def fit_predict_lgbm(
    df: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, float, float]:
    """LightGBM через mlforecast с лагами и скользящими статистиками."""
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.target_transforms import Differences

    mlf = MLForecast(
        models={
            "LGBM": lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.03,
                num_leaves=31, min_child_samples=20,
                random_state=42, verbose=-1,
            ),
        },
        freq=FREQ,
        lags=[1, 2, 3, 4, 5, 10, 20],
        lag_transforms={
            5:  [("rolling_mean", 5),  ("rolling_std", 5)],
            10: [("rolling_mean", 10), ("rolling_std", 10)],
            20: [("rolling_mean", 20), ("rolling_std", 20)],
        },
        date_features=["dayofweek", "month", "quarter"],
        target_transforms=[Differences([1])],
    )

    t0 = time.perf_counter()
    mlf.fit(df=df)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = mlf.predict(h=horizon)
    pred_time = time.perf_counter() - t0

    pred = pred.rename(columns={"LGBM": "y_hat"})[["unique_id", "ds", "y_hat"]]
    return pred, fit_time, pred_time


MODEL_REGISTRY: dict[str, callable] = {
    "snaive":    fit_predict_snaive,
    "autoarima": fit_predict_autoarima,
    "autotheta": fit_predict_autotheta,
    "lgbm":      fit_predict_lgbm,
}


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────────────────────────────────────────

def run(
    input_path: Path,
    output_path: Path,
    model: str,
    horizon: int,
) -> pd.DataFrame:
    """Полный прогон пайплайна: загрузка → обучение → прогноз → сохранение."""
    if model not in MODEL_REGISTRY:
        raise ValueError(
            f"Неизвестная модель: {model!r}. "
            f"Доступные: {sorted(MODEL_REGISTRY)}"
        )

    # 1. Загрузка
    print(f"[1/4] Загрузка данных из {input_path}")
    df = load_series(input_path)
    print(
        f"      Строк: {len(df):,}  |  "
        f"Период: {df['ds'].min().date()} — {df['ds'].max().date()}"
    )

    # 2. Обучение
    print(f"[2/4] Обучение модели: {model}")
    fit_fn = MODEL_REGISTRY[model]
    pred, fit_time, pred_time = fit_fn(df, horizon)

    # 3. Прогноз
    print(f"[3/4] Прогноз на {horizon} торговых дней")
    print(f"      Время обучения:  {fit_time:.2f} с")
    print(f"      Время инференса: {pred_time:.3f} с")
    print(
        f"      Прогноз [{pred['ds'].min().date()} — {pred['ds'].max().date()}]: "
        f"среднее = {pred['y_hat'].mean():.1f}"
    )

    # 4. Сохранение
    print(f"[4/4] Сохранение прогноза → {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output_path, index=False)
    print("      Готово.")

    return pred


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S&P 500 forecasting pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="lgbm",
        choices=sorted(MODEL_REGISTRY),
        help="Модель для прогноза",
    )
    parser.add_argument(
        "--horizon", type=int, default=10,
        help="Горизонт прогноза (торговых дней)",
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Путь к входному CSV (unique_id, ds, y)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Путь к выходному CSV",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.input, args.output, args.model, args.horizon)
    return 0


if __name__ == "__main__":
    sys.exit(main())
