import numpy as np
import pandas as pd

from config import historical_data, actual_data, year
from core.preprocessing import prepare_data
from utils.visualization import plot_results
from core.models import build_model, HW_ARMIMA, HW_LSTM, hw_prophet_ensemble#, hw_xgboost_ensemble, \
    #hw_bayesian_ensemble, clustered_hw, wavelet_hw
from core.calculations import calculate_metrics

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


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
    ts, actual_series, year_labels = prepare_data(
        historical_data=historical_data,
        actual_data=actual_data,
        year=year
    )

    # Получаем метки годов для подписей
    f1 = year_labels['f1']
    f2 = year_labels['f2']
    f3 = year_labels['f3']

    month_names = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]

    # Остальной код остается прежним, но используем actual_series вместо actual_data
    model_add = build_model(ts, seasonal_type='add')
    model_mul = build_model(ts, seasonal_type='mul')

    forecast_add = np.round(model_add.forecast(12)).astype(int)
    forecast_mul = np.round(model_mul.forecast(12)).astype(int)

    print('\nРезультат прогнозирования Хольта-Винтерса')
    plot_results(ts, forecast_add, forecast_mul, actual_series, model_type='HW', f1=f1, f2=f2, f3=f3)
    print_results(forecast_add, forecast_mul, actual_series, month_names)

    print('\nРезультат прогнозирования Хольта-Винтерса & ARIMA')
    arima_forecast_add = HW_ARMIMA(ts, model_add)
    arima_forecast_mul = HW_ARMIMA(ts, model_mul)

    plot_results(ts, arima_forecast_add, arima_forecast_mul, actual_series, model_type='HW_ARIMA', f1=f1, f2=f2, f3=f3)
    print_results(arima_forecast_add, arima_forecast_mul, actual_series, month_names)

    # Добавлен вызов HW_LSTMM
    print('\nРезультат прогнозирования Хольта-Винтерса & LSTM')
    lstm_forecast_add = HW_LSTM(ts, model_add)
    lstm_forecast_mul = HW_LSTM(ts, model_mul)

    plot_results(ts, lstm_forecast_add, lstm_forecast_mul, actual_series, model_type='HW_LSTM', f1=f1, f2=f2, f3=f3)
    print_results(lstm_forecast_add, lstm_forecast_mul, actual_series, month_names)


    # Загрузка календаря праздников (добавить после prepare_data)
    try:
        holidays_df = pd.read_csv('data/holidays.csv')
        holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])  # Prophet требует datetime
    except FileNotFoundError:
        print("Предупреждение: файл holidays.csv не найден. Prophet будет использован без учета праздников")
        holidays_df = None

    print('\nРезультат прогнозирования Хольта-Винтерса + Prophet')
    prophet_forecast_add = hw_prophet_ensemble(ts,model_add,holidays_df)
    prophet_forecast_mul = hw_prophet_ensemble(ts,hw_model=model_mul,holidays_df=holidays_df)

    plot_results(
        ts,
        prophet_forecast_add,
        prophet_forecast_mul,
        actual_series,
        model_type='HW_Prophet',
        f1=year_labels['f1'],
        f2=year_labels['f2'],
        f3=year_labels['f3']
    )

    print_results(prophet_forecast_add, prophet_forecast_mul, actual_series, month_names)

if __name__ == "__main__":
    main()