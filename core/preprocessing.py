import pandas as pd


def prepare_data(historical_data: str, actual_data: str, year: str):
    """
    Подготавливает данные для моделирования из CSV-файлов

    Параметры:
        historical_data: путь к CSV с историческими данными
        actual_data: путь к CSV с актуальными данными
        year: базовый год в формате 'YYYY-MM'

    Возвращает:
        tuple: (ts, actual_series, year_labels)
    """
    # Загрузка данных из CSV
    historical_df = pd.read_csv(historical_data, parse_dates=["date"])
    actual_df = pd.read_csv(actual_data, parse_dates=["date"])

    # Создание временных рядов
    ts = historical_df.set_index("date")["value"]
    actual_series = actual_df.set_index("date")["value"]

    # Генерация меток годов
    year_labels = {
        "f1": f"{pd.to_datetime(year).year - 2} год",
        "f2": f"{pd.to_datetime(year).year - 1} год",
        "f3": f"{pd.to_datetime(year).year} год",
    }

    return ts, actual_series, year_labels
