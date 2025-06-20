import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima


def build_model(data, seasonal_type='add', seasonal_periods=12):
    """Построение модели Хольта-Винтерса"""
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
    """Комбинированная модель Хольта-Винтерса + ARIMA"""
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