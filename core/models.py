import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima


def build_model(data, seasonal_type='add', seasonal_periods=12):
    """
    Строит и обучает модель Хольта-Винтерса для временных рядов.

    Параметры:
    - data: pandas.Series - временной ряд для обучения
    - seasonal_type: str - тип сезонности ('add' - аддитивная, 'mul' - мультипликативная)
    - seasonal_periods: int - количество периодов в сезонном цикле (по умолчанию 12 для месячных данных)

    Возвращает:
    - Обученную модель ExponentialSmoothing
    """
    model = ExponentialSmoothing(
        data,
        trend='add',
        seasonal=seasonal_type,
        seasonal_periods=seasonal_periods,
        initialization_method='heuristic',
        damped_trend=False
    ).fit()
    return model


def HW_ARMIMA(ts, hw_model):
    """
    Комбинированная модель Хольта-Винтерса и ARIMA для остатков.
    Использует преимущества обеих моделей: HW для основной компоненты и ARIMA для остатков.

    Параметры:
    - ts: pandas.Series - исходный временной ряд
    - hw_model: обученная модель Хольта-Винтерса

    Возвращает:
    - pandas.Series - комбинированный прогноз на 12 периодов с датами в индексе

    Алгоритм:
    1. Получает прогноз HW на 12 периодов
    2. Вычисляет остатки (факт - прогноз HW)
    3. Строит ARIMA модель на остатках
    4. Суммирует прогнозы HW и ARIMA

    """
    hw_forecast = hw_model.forecast(12) # Получаем прогноз Хольта-Винтерса
    residuals = ts - hw_model.fittedvalues # Вычисляем остатки (фактические значения - fitted values модели)

    arima_model = auto_arima( # Строим модель на остатках
        residuals.dropna(),
        seasonal=False,
        suppress_warnings=True,
        error_action='ignore'
    )

    arima_forecast = arima_model.predict(n_periods=12) # Прогнозируем остатки на 12 периодов
    forecast_dates = pd.date_range(
        start=ts.index[-1] + pd.DateOffset(months=1),
        periods=12,
        freq='MS'
    )

    return pd.Series(hw_forecast.values + arima_forecast, index=forecast_dates) # Комбинируем прогнозы