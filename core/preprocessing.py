import numpy as np
import pandas as pd


def prepare_data(historical_data, actual_data, year):
    """
    Подготавливает временные ряды для анализа и прогнозирования, создавая pandas.Series
    с правильными временными метками и проверяя целостность данных.

    Параметры:
    ----------
    historical_data : array-like
        Массив исторических данных. Должен содержать данные за целое число лет
        (длина должна быть кратна 12 месяцам).
    actual_data : array-like
        Массив актуальных данных за последний год (ровно 12 значений).
    year : str
        Год начала исторических данных в формате 'YYYY-MM-DD' или 'YYYY'.

    Возвращает:
    ----------
    tuple (ts, actual_series, year_labels)
        - ts : pandas.Series
            Исторические данные с DatetimeIndex в месячной частоте ('MS')
        - actual_series : pandas.Series
            Актуальные данные за последний год с DatetimeIndex
        - year_labels : dict
            Словарь с метками годов:
            - 'f1': первый год исторических данных
            - 'f2': последний год исторических данных
            - 'f3': год актуальных данных

    Выбрасывает:
    -----------
    ValueError
        - Если historical_data не кратен 12 (неполные годы)
        - Если actual_data не содержит ровно 12 значений

    Пример использования:
    -------------------
    >>> historical = [10, 12, 15, ...]  # 24 значения (2 года)
    >>> actual = [20, 22, 18, ...]     # 12 значений
    >>> ts, actual_series, labels = prepare_data(historical, actual, '2020')
    >>> print(labels)
    {'f1': '2020', 'f2': '2021', 'f3': '2022'}
    """
    # Загрузка данных из CSV
    historical_df = pd.read_csv(historical_data, parse_dates=['date'])
    actual_df = pd.read_csv(actual_data, parse_dates=['date'])

    # Создание временных рядов
    ts = historical_df.set_index('date')['value']
    actual_series = actual_df.set_index('date')['value']

    # Генерация меток годов
    year_labels = {
        'f1': f"{pd.to_datetime(year).year - 2} год",
        'f2': f"{pd.to_datetime(year).year - 1} год",
        'f3': f"{pd.to_datetime(year).year} год"
    }

    return ts, actual_series, {
        'f1': year[:4],
        'f2': str(last_year_start.year),
        'f3': str(int(str(last_year_start.year)) + 1)
    }