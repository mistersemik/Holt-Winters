import numpy as np
import pandas as pd
import logging
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from core.preprocessing import prepare_data
from utils.visualization import plot_results
from core.models import (
    HW_ARMIMA,
    HW_LSTM,
    hw_prophet_ensemble,
    build_hw_tcn_model,
    hw_xgboost_ensemble,
    hw_bayesian_ensemble,
    clustered_hw,
    wavelet_hw,
    naive_forecast,
    hw_garch,
    hw_sarima_ks,
)
from core.calculations import calculate_metrics
from utils.config_loader import load_config

# Настройка логгирования
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").disabled = True
logging.getLogger("pystan").disabled = True

# Настройка предупреждений
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", ConvergenceWarning)

config = load_config()


def print_results(forecast_add, forecast_mul, actual, month_names):
    """
    Формирует и выводит детализированный отчёт, включающий:
    - Табличное сравнение фактических значений с прогнозами двух моделей
    - Метрики качества прогноза (MAE, RMSE, MAPE) для каждой модели
    - Автоматическое форматирование чисел для удобства чтения

    Параметры:
    ----------
    forecast_add : array-like
        Прогнозные значения аддитивной модели
        (np.array, pd.Series или список).
        Должен совпадать по длине с actual.
    forecast_mul : array-like
        Прогнозные значения мультипликативной модели.
        Должен совпадать по длине с actual.
    actual : pd.Series
        Фактические значения временного ряда с корректным индексом.
    month_names : list
        Список названий месяцев/периодов для отображения
        (длина >= len(actual)).

    Возвращает:
    ----------
    pd.DataFrame
        Таблица с отформатированными результатами сравнения.

    Пример использования:
    ---------------------
    >>> actual = pd.Series([1500, 2000, 1800])
    >>> forecast_add = [1600, 1900, 1700]
    >>> forecast_mul = [1550, 1950, 1750]
    >>> month_names = ['Январь', 'Февраль', 'Март']
    >>> print_results(forecast_add, forecast_mul, actual, month_names)
    """

    # Таблица сравнения
    results = pd.DataFrame(
        {
            "Месяц": month_names[: len(actual)],
            "Факт": [f"{x:,}".replace(",", " ") for x in actual.values],
            "Аддитивная модель": [
                f"{x:,}".replace(",", " ") for x in forecast_add[: len(actual)]
            ],
            "Мультипликативная модель": [
                f"{x:,}".replace(",", " ") for x in forecast_mul[: len(actual)]
            ],
        }
    )

    print("\nСравнение прогноза и фактических значений:")
    print(results.to_string(index=False))

    # Метрики ошибок
    mae_add, rmse_add, mape_add = calculate_metrics(forecast_add, actual)
    mae_mul, rmse_mul, mape_mul = calculate_metrics(forecast_mul, actual)

    print("\nМетрики ошибок для аддитивной модели:")

    print(
        (
            f"MAE: {mae_add:,.0f}".replace(",", " ")
            + " (Средняя абсолютная ошибка)"
        )
    )
    print(
        (
            f"RMSE: {rmse_add:,.0f}".replace(",", " ")
            + "(Среднеквадратичная ошибка)"
        )
    )

    print(f"MAPE: {mape_add:.1f}%")

    print("\nМетрики ошибок для мультипликативной модели:")

    print(
        (
            f"MAE: {mae_mul:,.0f}".replace(",", " ")
            + " (Средняя абсолютная ошибка)"
        )
    )

    print(
        (
            f"RMSE: {rmse_mul:,.0f}".replace(",", " ")
            + "(Среднеквадратичная ошибка)"
        )
    )

    print(
        (f"MAPE: {mape_mul:.1f}%" + " (Средняя абсолютная процентная ошибка)")
    )


def main():
    """Основной поток выполнения"""

    ts, actual_series, year_labels = prepare_data(
        historical_data=config["data"]["historical"],
        actual_data=config["data"]["actual"],
        year=config["data"]["year"],
    )

    # Получаем метки годов для подписей
    f1 = year_labels["f1"]
    f2 = year_labels["f2"]
    f3 = year_labels["f3"]

    month_names = [
        "Янв",
        "Фев",
        "Мар",
        "Апр",
        "Май",
        "Июн",
        "Июл",
        "Авг",
        "Сен",
        "Окт",
        "Ноя",
        "Дек",
    ]

    model_add = ExponentialSmoothing(
        ts,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="heuristic",
        damped_trend=False,
    ).fit()

    model_mul = ExponentialSmoothing(
        ts,
        trend="mul",
        seasonal="mul",
        seasonal_periods=12,
        initialization_method="heuristic",
        damped_trend=False,
    ).fit()

    forecast_add = np.round(model_add.forecast(12)).astype(int)
    forecast_mul = np.round(model_mul.forecast(12)).astype(int)

    print("\nРезультат прогнозирования Хольта-Винтерса")

    plot_results(
        ts,
        forecast_add,
        forecast_mul,
        actual_series,
        model_type="HW",
        f1=f1,
        f2=f2,
        f3=f3,
    )

    print_results(forecast_add, forecast_mul, actual_series, month_names)

    print("\nРезультат прогнозирования Хольта-Винтерса & ARIMA")
    arima_forecast_add = HW_ARMIMA(ts, model_add)
    arima_forecast_mul = HW_ARMIMA(ts, model_mul)

    plot_results(
        ts,
        arima_forecast_add,
        arima_forecast_mul,
        actual_series,
        model_type="HW_ARIMA",
        f1=f1,
        f2=f2,
        f3=f3,
    )

    print_results(
        arima_forecast_add, arima_forecast_mul, actual_series, month_names
    )

    # Добавлен вызов HW_LSTM
    print("\nРезультат прогнозирования Хольта-Винтерса & LSTM")
    lstm_forecast_add = HW_LSTM(ts, model_add)
    lstm_forecast_mul = HW_LSTM(ts, model_mul)

    plot_results(
        ts,
        lstm_forecast_add,
        lstm_forecast_mul,
        actual_series,
        model_type="HW_LSTM",
        f1=f1,
        f2=f2,
        f3=f3,
    )

    print_results(
        lstm_forecast_add, lstm_forecast_mul, actual_series, month_names
    )

    # Загрузка календаря праздников (добавить после prepare_data)
    try:
        holidays_df = pd.read_csv("data/holidays.csv")
        holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])
        # Prophet требует datetime
        # print("Успешно загружено праздников:", len(holidays_df))
        # print(holidays_df.head())
    except FileNotFoundError:
        print(
            "Предупреждение: файл holidays.csv не найден."
            "Prophet будет использован без учета праздников"
        )
        holidays_df = None

    print("\nРезультат прогнозирования Хольта-Винтерса + Prophet")

    prophet_forecast_add = hw_prophet_ensemble(ts, model_add, holidays_df)

    prophet_forecast_mul = hw_prophet_ensemble(
        ts, hw_model=model_mul, holidays_df=holidays_df
    )

    plot_results(
        ts,
        prophet_forecast_add,
        prophet_forecast_mul,
        actual_series,
        model_type="HW_Prophet",
        f1=year_labels["f1"],
        f2=year_labels["f2"],
        f3=year_labels["f3"],
    )

    print_results(
        prophet_forecast_add, prophet_forecast_mul, actual_series, month_names
    )

    combined_add, hw_forecast_add, tcn_forecast_add, dates_add = (
        build_hw_tcn_model(model_add, ts, n_steps=24, forecast_steps=12)
    )

    combined_mul, hw_forecast_mul, tcn_forecast_mul, dates_mul = (
        build_hw_tcn_model(model_mul, ts, n_steps=24, forecast_steps=12)
    )

    forecast_series_add = pd.Series(combined_add, index=dates_add)
    forecast_series_mul = pd.Series(combined_mul, index=dates_mul)

    print("\nРезультат прогнозирования Хольта-Винтерса + TCN")

    plot_results(
        ts,
        forecast_series_add,
        forecast_series_mul,
        actual_series,
        model_type="HW_TCN",
        f1=year_labels["f1"],
        f2=year_labels["f2"],
        f3=year_labels["f3"],
    )

    print_results(
        forecast_series_add, forecast_series_mul, actual_series, month_names
    )

    # Прогнозирование с XGBoost
    try:
        # Загрузка внешних признаков (если есть)
        try:
            exog_data = pd.read_csv(
                "data/exog_features.csv", index_col=0, parse_dates=True
            )

            # Выравнивание по индексу временного ряда
            exog_data = exog_data.reindex(ts.index)
            print("Найдены внешние признаки для XGBoost")
        except FileNotFoundError:
            print(
                "Файл с внешними признаками не найден,"
                "будут использованы лаговые признаки"
            )
            exog_data = None

        print("\nРезультат прогнозирования Хольта-Винтерса + XGBoost")
        xgb_forecast_add = hw_xgboost_ensemble(ts, model_add, exog_data)
        xgb_forecast_mul = hw_xgboost_ensemble(ts, model_mul, exog_data)

        plot_results(
            ts,
            xgb_forecast_add,
            xgb_forecast_mul,
            actual_series,
            model_type="HW_XGBoost",
            f1=f1,
            f2=f2,
            f3=f3,
        )

        print_results(
            xgb_forecast_add, xgb_forecast_mul, actual_series, month_names
        )

    except Exception as e:
        print(f"Ошибка в XGBoost ансамбле: {str(e)}")
        naive_unit = naive_forecast(ts, periods=12)

        plot_results(
            ts,
            naive_unit,
            naive_unit,
            actual_series,
            model_type="Naive_Fallback",
            f1=f1,
            f2=f2,
            f3=f3,
        )

        print_results(naive_unit, naive_unit, actual_series, month_names)

    # Байесовский ансамбль
    try:
        print("Запуск байесовского ансамбля...")
        bayesian_forecast_add, trace = hw_bayesian_ensemble(ts, model_add)
        bayesian_forecast_mul, trace = hw_bayesian_ensemble(ts, model_mul)

        plot_results(
            ts,
            bayesian_forecast_add,
            bayesian_forecast_mul,
            actual_series,
            model_type="HW_Bayesian",
            f1=f1,
            f2=f2,
            f3=f3,
        )

        print_results(
            bayesian_forecast_add,
            bayesian_forecast_mul,
            actual_series,
            month_names,
        )

    except Exception as e:
        print(f"Ошибка в байесовском ансамбле: {str(e)}")
        naive_unit = naive_forecast(ts, periods=12)

        plot_results(
            ts,
            naive_unit,
            naive_unit,
            actual_series,
            model_type="Naive_Fallback",
            f1=f1,
            f2=f2,
            f3=f3,
        )

        print_results(naive_unit, naive_unit, actual_series, month_names)

    # Прогнозирование с кластеризацией
    try:
        cluster_forecast_add, weights_add = clustered_hw(
            ts, hw_model=model_add
        )
        cluster_forecast_mul, weights_mul = clustered_hw(
            ts, hw_model=model_mul
        )

        print("\nВеса кластеров (аддитивная модель):")
        print(weights_add)

        print("\nВеса кластеров (мультипликативная модель):")
        print(weights_mul)

        plot_results(
            ts,
            cluster_forecast_add,
            cluster_forecast_mul,
            actual_series,
            model_type="Clustered_HW",
            f1=f1,
            f2=f2,
            f3=f3,
        )

        print_results(
            cluster_forecast_add,
            cluster_forecast_mul,
            actual_series,
            month_names,
        )

    except Exception as e:
        print(f"Ошибка при прогнозировании: {str(e)}")
        naive_unit = naive_forecast(ts, periods=12)
        plot_results(
            ts,
            naive_unit,
            naive_unit,
            actual_series,
            model_type="Naive_Fallback",
            f1=f1,
            f2=f2,
            f3=f3,
        )
        print_results(naive_unit, naive_unit, actual_series, month_names)

    try:
        # Получаем вейвлет-прогноз
        wavelet_forecast_add = wavelet_hw(ts, hw_model=model_add)
        wavelet_forecast_mul = wavelet_hw(ts, hw_model=model_mul)

        plot_results(
            ts,
            forecast_add,
            forecast_mul,
            actual_series,
            model_type="HW_Wavelet",
            f1=f1,
            f2=f2,
            f3=f3,
        )

        print_results(
            wavelet_forecast_add,
            wavelet_forecast_mul,
            actual_series,
            month_names,
        )

    except Exception as e:
        print(f"Ошибка при прогнозировании: {str(e)}")
        naive_unit = naive_forecast(ts, periods=12)

        plot_results(
            ts,
            naive_unit,
            naive_unit,
            actual_series,
            model_type="Naive_Fallback",
            f1=f1,
            f2=f2,
            f3=f3,
        )

        print_results(naive_unit, naive_unit, actual_series, month_names)

    print("\nРезультат прогнозирования HW + GARCH")
    garch_forecast_add = hw_garch(ts, model_add)
    garch_forecast_mul = hw_garch(ts, model_mul)

    plot_results(
        ts,
        garch_forecast_add,
        garch_forecast_mul,
        actual_series,
        model_type="HW_GARCH",
        f1=f1,
        f2=f2,
        f3=f3,
    )
    print_results(
        garch_forecast_add, garch_forecast_mul, actual_series, month_names
    )

    # Гибрид SARIMA + Kernel Smoothing
    print("\nРезультат прогнозирования HW + SARIMA_KS")
    sarimaks_forecast_add = hw_sarima_ks(ts, model_add)
    sarimaks_forecast_mul = hw_sarima_ks(ts, model_mul)

    plot_results(
        ts,
        sarimaks_forecast_add,
        sarimaks_forecast_mul,
        actual_series,
        model_type="HW_SARIMA_KS",
        f1=f1,
        f2=f2,
        f3=f3,
    )
    print_results(
        sarimaks_forecast_add,
        sarimaks_forecast_mul,
        actual_series,
        month_names,
    )


if __name__ == "__main__":
    main()
