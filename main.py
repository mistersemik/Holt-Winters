import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pmdarima import auto_arima

from config import *

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

def HW_ARMIMA(ts, hw_model):
    """Комбинированная модель Хольта-Винтерса + ARIMA на остатках"""
    # 1. Получаем прогноз Хольта-Винтерса
    hw_forecast = hw_model.forecast(12)

    # 2. Вычисляем остатки (фактические значения - fitted values модели)
    hw_fitted = hw_model.fittedvalues
    residuals = ts - hw_fitted

    # 3. Строим ARIMA модель на остатках
    arima_model = auto_arima(residuals.dropna(),
                             seasonal=False,
                             suppress_warnings=True,
                             error_action='ignore')

    # 4. Прогнозируем остатки на 12 периодов
    arima_forecast = arima_model.predict(n_periods=12)

    # 5. Создаем временные метки для прогноза
    last_date = ts.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=12,
        freq='MS'
    )

    # 6. Комбинируем прогнозы
    combined_forecast = pd.Series(
        hw_forecast.values + arima_forecast,
        index=forecast_dates
    )

    return combined_forecast

def plot_results(historical, forecast_add, forecast_mul, actual, model_type='HW'):
    """Визуализация с подсветкой аномалий и легендой"""
    plt.figure(figsize=(16, 8))
    ax = plt.gca()

    # Форматирование оси Y
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " ")))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Стили для аномалий
    HIST_ANOMALY_COLOR = 'gold'
    ACTUAL_ANOMALY_COLOR = 'salmon'
    ALPHA = 0.3

    # Основные подписи
    labels = {
        'historical': f"Исторические данные ({f1}-{f2})",
        'add': 'Прогноз (аддитивная модель)' if model_type == 'HW' else 'Прогноз (аддитивная HW+ARIMA)',
        'mul': 'Прогноз (мультипликативная модель)' if model_type == 'HW' else 'Прогноз (мультипликативная HW+ARIMA)',
        'actual': f"Фактические данные {f3}"
    }

    # Основные графики
    hist_line = plt.plot(historical.index, historical, 'b-', label=labels['historical'],
                        marker='o', linewidth=2)[0]
    add_line = plt.plot(forecast_add.index, forecast_add, 'g--', label=labels['add'],
                       marker='s', linewidth=2)[0]
    mul_line = plt.plot(forecast_mul.index, forecast_mul, 'r--', label=labels['mul'],
                       marker='^', linewidth=2)[0]
    actual_line = plt.plot(actual.index, actual, 'k-', label=labels['actual'],
                          marker='D', linewidth=2, markersize=8)[0]

    # Детекция аномалий
    def detect_anomalies(series, window=12):
        rolling = series.rolling(window)
        mad = 1.4826 * rolling.apply(lambda x: np.median(np.abs(x - np.median(x))))
        return series[np.abs(series - rolling.median()) > 3 * mad]

    # Подсветка исторических аномалий
    hist_anomalies = detect_anomalies(historical)
    for date in hist_anomalies.index:
        plt.axvspan(date - pd.Timedelta(days=15), date + pd.Timedelta(days=15),
                   color=HIST_ANOMALY_COLOR, alpha=ALPHA)

    # Подсветка неожиданных фактических значений
    forecast_avg = (forecast_add + forecast_mul) / 2
    deviations = np.abs(actual - forecast_avg[:len(actual)])
    actual_anomalies = actual[deviations > deviations.quantile(0.9) * 2]
    for date in actual_anomalies.index:
        plt.axvspan(date - pd.Timedelta(days=15), date + pd.Timedelta(days=15),
                   color=ACTUAL_ANOMALY_COLOR, alpha=ALPHA)

    # Создаем элементы для легенды
    from matplotlib.patches import Patch
    legend_elements = [
        hist_line,
        add_line,
        mul_line,
        actual_line,
        Patch(facecolor=HIST_ANOMALY_COLOR, alpha=ALPHA, label='Исторические аномалии'),
        Patch(facecolor=ACTUAL_ANOMALY_COLOR, alpha=ALPHA, label='Неожиданные значения')
    ]

    # Настройка отображения
    title_suffix = 'модели Хольта-Винтерса' if model_type == 'HW' else 'комбинированные модели'
    plt.title(f'Прогноз и фактические значения ({f1}-{f3}), {title_suffix}',
              fontsize=16, pad=20)
    plt.xlabel('Месяц', fontsize=14)
    plt.ylabel('Количество атак', fontsize=14)
    plt.legend(handles=legend_elements, fontsize=12, loc='upper left')
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
    plot_results(ts, forecast_add, forecast_mul, actual_data, model_type='HW')
    print_results(forecast_add, forecast_mul, actual_data, month_names)

    # Построение модели ARIMA+HW
    arima_forecast_add = HW_ARMIMA(ts, model_add)
    arima_forecast_mul = HW_ARMIMA(ts, model_mul)

    plot_results(ts, arima_forecast_add, arima_forecast_mul, actual_data, model_type='HW_ARIMA')
    print_results(arima_forecast_add, arima_forecast_mul, actual_data, month_names)


if __name__ == "__main__":
    main()