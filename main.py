import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.ticker import FuncFormatter, MaxNLocator

def prepare_data():
    """Подготовка и загрузка данных"""
    historical_data = np.array([
        21894, 32310, 173759, 131524, 121917, 98039, 146247, 151564,
        111169, 109041, 86699, 71410, 124160, 128950, 131759, 111093,
        176348, 149677, 178979, 162920, 162662, 270580, 339502, 331568
    ])

    actual_2024 = {
        'Январь': 204276, 'Февраль': 120337, 'Март': 172254, 'Апрель': 157239,
        'Май': 256708, 'Июнь': 298654, 'Июль': 310759,'Август': 150086,
        'Сентябрь': 150904, 'Октябрь': 219280, 'Ноябрь': 197700, 'Декабрь': 226484
    }

    dates = pd.date_range(start='2022-01', periods=len(historical_data), freq='ME')
    ts = pd.Series(historical_data, index=dates)
    actual_dates = pd.date_range(start='2024-01', periods=len(actual_2024), freq='ME')
    actual_series = pd.Series(actual_2024.values(), index=actual_dates)

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
    plt.plot(historical.index, historical, 'b-', label='Исторические данные (2022-2023)', marker='o', linewidth=2)
    plt.plot(forecast_add.index, forecast_add, 'g--', label='Прогноз (аддитивная модель)', marker='s', linewidth=2)
    plt.plot(forecast_mul.index, forecast_mul, 'r--', label='Прогноз (мультипликативная модель)', marker='^',
             linewidth=2)
    plt.plot(actual.index, actual, 'k-', label='Фактические данные 2024', marker='D', linewidth=2, markersize=8)

    # # Добавление подписей
    # for date, value in zip(actual.index, actual.values):
    #     plt.text(date, value, f"{value:,}".replace(",", " "),
    #              ha='center', va='bottom', fontsize=10,
    #              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Настройка отображения
    plt.title('Прогноз и фактические значения количества атак (2022-2024)', fontsize=16, pad=20)
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
    ts, actual_2024 = prepare_data()
    month_names = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]

    # Построение моделей
    model_add = build_model(ts, seasonal_type='add')
    model_mul = build_model(ts, seasonal_type='mul')

    # Прогнозирование
    forecast_add = np.round(model_add.forecast(12)).astype(int)
    forecast_mul = np.round(model_mul.forecast(12)).astype(int)

    # Визуализация и вывод результатов
    plot_results(ts, forecast_add, forecast_mul, actual_2024)
    print_results(forecast_add, forecast_mul, actual_2024, month_names)


if __name__ == "__main__":
    main()