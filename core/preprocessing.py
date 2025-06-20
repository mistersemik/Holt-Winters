import numpy as np
import pandas as pd


def prepare_data(historical_data, actual_data, year):
    """Подготовка временных рядов"""
    if len(historical_data) % 12 != 0:
        raise ValueError("historical_data должен содержать данные ровно за целые годы (кратно 12)")
    if len(actual_data) != 12:
        raise ValueError("actual_data должен содержать ровно 12 значений (данные за 1 год)")

    historical_dates = pd.date_range(start=year, periods=len(historical_data), freq='MS')
    ts = pd.Series(np.array(historical_data), index=historical_dates)

    num_years = len(historical_data) // 12
    last_year_start = pd.to_datetime(year) + pd.DateOffset(years=num_years - 1)

    actual_dates = pd.date_range(
        start=last_year_start + pd.DateOffset(years=1),
        periods=12,
        freq='MS'
    )
    actual_series = pd.Series(np.array(actual_data), index=actual_dates)

    return ts, actual_series, {
        'f1': year[:4],
        'f2': str(last_year_start.year),
        'f3': str(int(str(last_year_start.year)) + 1)
    }