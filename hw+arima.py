import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Генерация данных с внезапным изменением паттерна
np.random.seed(666)  # Для воспроизводимости

# Исторические данные (2021-2023) - стабильный рост и сезонность
historical = np.array([
    120, 110, 150, 140, 180, 170,
    200, 190, 230, 220, 260, 250,  # 2021
    280, 270, 310, 300, 340, 330,
    360, 350, 390, 380, 420, 410,  # 2022
    440, 430, 470, 460, 500, 490,
    520, 510, 550, 540, 580, 570   # 2023
])

# Данные 2024 - неожиданный спад в июне
actual_2024 = np.array([
    600, 590, 630, 620, 660,  # Январь-Май (продолжаем тренд)
    300,  # Июнь - внезапный спад (DDoS атака прекратилась)
    650, 640  # Июль-Август - восстановление
])

# 2. Создание временных рядов
dates = pd.date_range(start='2021-01', periods=36, freq='ME')
ts = pd.Series(historical, index=dates)

future_dates = pd.date_range(start='2024-01', periods=8, freq='ME')
actual_series = pd.Series(actual_2024, index=future_dates)

# 3. Прогноз Хольта-Винтерса (обычный метод)
hw_model = ExponentialSmoothing(
    ts,
    trend='add', # тренд аддтивной модели
    seasonal='mul', # сезонность аддитивной модели
    seasonal_periods=12
).fit()
hw_forecast = hw_model.forecast(8)

# 4. Комбинированный метод (HW + ARIMA)
residuals = ts - hw_model.fittedvalues
arima_model = auto_arima(residuals, seasonal=False, suppress_warnings=True)
arima_forecast = arima_model.predict(n_periods=8)
combined_forecast = hw_forecast + arima_forecast

# 5. Визуализация с акцентом на различия
plt.figure(figsize=(14, 7))
plt.plot(ts, 'b-', label='Исторические данные (2021-2023)')
plt.plot(actual_series, 'ko-', label='Факт 2024', markersize=8, linewidth=2)

# Прогнозы
plt.plot(hw_forecast, 'g--', label=f'HW (MAE: {mean_absolute_error(actual_series, hw_forecast):.1f})')
plt.plot(combined_forecast, 'r-',
         label=f'HW+ARIMA (MAE: {mean_absolute_error(actual_series, combined_forecast):.1f})')

# Выделяем аномалию
plt.axvspan(pd.to_datetime('2024-06-01'), pd.to_datetime('2024-06-30'),
            color='red', alpha=0.1, label='Аномалия (июнь)')

plt.title('Четкое сравнение методов: HW vs HW+ARIMA', fontsize=16)
plt.xlabel('Дата', fontsize=14)
plt.ylabel('Количество атак', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 6. Расчет метрик ошибок
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

mae_hw, rmse_hw, mape_hw = calculate_metrics(actual_2024, hw_forecast)
mae_comb, rmse_comb, mape_comb = calculate_metrics(actual_2024, combined_forecast)

# 7. Таблица сравнения
print("\nДетальное сравнение прогнозов:")
comparison = pd.DataFrame({
    'Месяц': ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июнь*', 'Июл', 'Авг'],
    'Факт': actual_2024,
    'HW': hw_forecast.round().astype(int),
    'HW+ARIMA': combined_forecast.round().astype(int),
    'Ошибка HW': (hw_forecast - actual_2024).round().astype(int),
    'Ошибка Comb': (combined_forecast - actual_2024).round().astype(int)
})
print(comparison.to_string(index=False))

# 8. Сравнение метрик ошибок
print("\nСравнение метрик ошибок:")
metrics_df = pd.DataFrame({
    'Метрика': ['MAE', 'RMSE', 'MAPE (%)'],
    'HW': [mae_hw, rmse_hw, mape_hw],
    'HW+ARIMA': [mae_comb, rmse_comb, mape_comb],
    'Разница': [mae_hw - mae_comb, rmse_hw - rmse_comb, mape_hw - mape_comb]
})
metrics_df['HW'] = metrics_df['HW'].round(1)
metrics_df['HW+ARIMA'] = metrics_df['HW+ARIMA'].round(1)
metrics_df['Разница'] = metrics_df['Разница'].round(1)
print(metrics_df.to_string(index=False))