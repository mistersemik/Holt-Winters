import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(forecast, actual):
    """Расчет метрик ошибок"""
    forecast = forecast[:len(actual)]
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return mae, rmse, mape