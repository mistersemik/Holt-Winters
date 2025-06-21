import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.patches import Patch

def plot_results(historical, forecast_add, forecast_mul, actual, model_type='HW', f1=None, f2=None, f3=None):
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
        'add': 'Прогноз (аддитивная модель)' if model_type == 'HW'
        else 'Прогноз (аддитивная HW+ARIMA)' if model_type == 'HW_ARIMA'
        else 'Прогноз (аддитивная HW+LSTM)',
        'mul': 'Прогноз (мультипликативная модель)' if model_type == 'HW'
        else 'Прогноз (мультипликативная HW+ARIMA)' if model_type == 'HW_ARIMA'
        else 'Прогноз (мультипликативная HW+LSTM)',
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