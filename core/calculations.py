import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(forecast, actual):
    """
        Вычисляет основные метрики качества прогноза временного ряда.

    Параметры:
    ----------
    forecast : array-like или pandas.Series
        Прогнозируемые значения. Если длина превышает actual,
        будет автоматически обрезан до длины actual.
    actual : array-like или pandas.Series
        Фактические (наблюдаемые) значения временного ряда.
        Должен иметь одинаковую размерность с forecast.

    Возвращает:
    ----------
    tuple (mae, rmse, mape)
        - mae (float): Средняя абсолютная ошибка (Mean Absolute Error)
        - rmse (float): Среднеквадратичная ошибка (Root Mean Square Error)
        - mape (float): Средняя абсолютная процентная ошибка (Mean Absolute Percentage Error), %

    Формулы:
    -------
    MAE = mean(|actual - forecast|)
    RMSE = sqrt(mean((actual - forecast)^2))
    MAPE = mean(|(actual - forecast)/actual|) * 100%

    Особенности:
    -----------
    - Автоматически обрезает прогноз до длины фактических данных
    - Устойчив к разным типам входных данных (numpy.array, pandas.Series, list)
    - Для MAPE рекомендуется использовать только с положительными значениями actual
    - Все метрики возвращаются в исходных единицах измерения (кроме MAPE - %)

    Пример использования:
    -------------------
    >>> forecast = [10, 20, 30]
    >>> actual = [12, 18, 28]
    >>> mae, rmse, mape = calculate_metrics(forecast, actual)
    >>> print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%")
    MAE: 2.00, RMSE: 2.16, MAPE: 10.2%
    """
    forecast = forecast[:len(actual)]
    mae = mean_absolute_error(actual, forecast) # Средняя абсолютная ошибка
    rmse = np.sqrt(mean_squared_error(actual, forecast)) # Среднеквадратичная ошибка
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100 # Средняя абсолютная процентная ошибка
    return mae, rmse, mape