import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.ticker import FuncFormatter, MaxNLocator

from config import *

def prepare_data():
    """Подготовка данных с автоматическим определением последнего года"""
    # Проверка что historical_data содержит целое число лет (кратно 12)
    if len(historical_data) % 12 != 0:
        raise ValueError("historical_data должен содержать данные ровно за целые годы (кратно 12)")

    # Преобразуем в numpy array
    historical_array = np.array(historical_data)
    actual_array = np.array(actual_data)

    # 1. Создаём основной временной ряд (historical)
    historical_dates = pd.date_range(
        start=year,
        periods=len(historical_array),
        freq='MS'
    )
    ts = pd.Series(historical_array, index=historical_dates)

    # 2. Определяем последний год для сравнения (actual)
    num_years = len(historical_data) // 12
    last_year_start = pd.to_datetime(year) + pd.DateOffset(years=num_years - 1)


    global f1, f2, f3
    f1 = year[:4] #Год относительно которого отсчёт
    f2 = str(last_year_start.year) #Финальный для исторических данных год
    f3 = str(int(f2)+1) # Финальный фактический год

    # Проверяем что actual_data содержит ровно 12 значений
    if len(actual_array) != 12:
        raise ValueError("actual_data должен содержать ровно 12 значений (данные за 1 год)")

    # Создаём даты для actual (последний год historical + 1 год)
    actual_dates = pd.date_range(
        start=last_year_start + pd.DateOffset(years=1),
        periods=12,
        freq='MS'
    )
    actual_series = pd.Series(actual_array, index=actual_dates)

    return ts, actual_series


def build_model(data, seasonal_type='add', seasonal_periods=12):
    """Построение модели Хольта-Винтерса"""
    model = ExponentialSmoothing(
        data,
        trend='add',
        seasonal=seasonal_type,
        seasonal_periods=seasonal_periods,
        initialization_method='heuristic', # estimated
        damped_trend = False  # Отключаем затухание
    ).fit()
    return model


def calculate_metrics(forecast, actual):
    """Расчет метрик ошибок"""

    mae = mean_absolute_error(actual, forecast[:len(actual)]) # Средняя абсолютная ошибка
    rmse = np.sqrt(mean_squared_error(actual, forecast[:len(actual)])) # Среднеквадратичная ошибка
    mape = np.mean(np.abs((actual - forecast[:len(actual)]) / actual)) * 100 # Средняя абсолютная процентная ошибка
    return mae, rmse, mape


def plot_results(historical, forecast_add, forecast_mul, actual):
    """Визуализация результатов"""
    plt.figure(figsize=(16, 8))
    ax = plt.gca()

    # Форматирование оси Y
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " ")))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Построение графиков
    plt.plot(historical.index, historical, 'b-', label=f"Исторические данные ({f1}-{f2})", marker='o', linewidth=2)
    plt.plot(forecast_add.index, forecast_add, 'g--', label='Прогноз (аддитивная модель)', marker='s', linewidth=2)
    plt.plot(forecast_mul.index, forecast_mul, 'r--', label='Прогноз (мультипликативная модель)', marker='^',
             linewidth=2)
    plt.plot(actual.index, actual, 'k-', label=f"Фактические данные {f3}", marker='D', linewidth=2, markersize=8)

    # # Добавление подписей
    # for date, value in zip(actual.index, actual.values):
    #     plt.text(date, value, f"{value:,}".replace(",", " "),
    #              ha='center', va='bottom', fontsize=10,
    #              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Настройка отображения
    plt.title(f'Прогноз и фактические значения количества атак ({f1}-{f3})', fontsize=16, pad=20)
    plt.xlabel('Месяц', fontsize=14)
    plt.ylabel('Количество атак', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    plt.show()


def print_results(forecast_add, forecast_mul, actual, month_names):
    """Вывод табличных результатов и метрик"""
    # Таблица сравнения
    results = pd.DataFrame({
        'Месяц': month_names[:len(actual)],
        'Факт': [f"{x:,}".replace(",", " ") for x in actual.values],
        'Аддитивная модель': [f"{x:,}".replace(",", " ") for x in forecast_add[:len(actual)]],
        'Мультипликативная модель': [f"{x:,}".replace(",", " ") for x in forecast_mul[:len(actual)]]
    })

    print("\nСравнение прогноза и фактических значений:")
    print(results.to_string(index=False))

    # Метрики ошибок
    mae_add, rmse_add, mape_add = calculate_metrics(forecast_add, actual)
    mae_mul, rmse_mul, mape_mul = calculate_metrics(forecast_mul, actual)

    print("\nМетрики ошибок для аддитивной модели:")
    print(f"MAE: {mae_add:,.0f}".replace(",", " ") + ' (Средняя абсолютная ошибка)')
    print(f"RMSE: {rmse_add:,.0f}".replace(",", " ") + '(Среднеквадратичная ошибка)')
    print(f"MAPE: {mape_add:.1f}%")

    print("\nМетрики ошибок для мультипликативной модели:")
    print(f"MAE: {mae_mul:,.0f}".replace(",", " ")+ ' (Средняя абсолютная ошибка)')
    print(f"RMSE: {rmse_mul:,.0f}".replace(",", " ") + '(Среднеквадратичная ошибка)')
    print(f"MAPE: {mape_mul:.1f}%" + ' (Средняя абсолютная процентная ошибка)')


def main():
    """Основной поток выполнения"""
    # Подготовка данных
    ts, actual_data = prepare_data()
    month_names = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]

    # Построение моделей
    model_add = build_model(ts, seasonal_type='add')
    model_mul = build_model(ts, seasonal_type='mul')

    # Прогнозирование
    forecast_add = np.round(model_add.forecast(12)).astype(int)
    forecast_mul = np.round(model_mul.forecast(12)).astype(int)

    # Визуализация и вывод результатов
    plot_results(ts, forecast_add, forecast_mul, actual_data)
    print_results(forecast_add, forecast_mul, actual_data, month_names)


if __name__ == "__main__":
    main()