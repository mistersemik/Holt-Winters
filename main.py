import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Ваши данные
sales_data = np.array([
    21894, 32310, 173759, 131524, 121917, 98039, 146247, 151564,
    111169, 109041, 86699, 71410, 124160, 128950, 131759, 111093,
    176348, 149677, 178979, 162920, 162662, 270580, 339502, 331568
])

# Создаем временной ряд
dates = pd.date_range(start='2022-01', periods=len(sales_data), freq='M')
ts = pd.Series(sales_data, index=dates)
month_names = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн",
               "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]

# Функция для построения модели
def fit_model(series, seasonal_type):
    return ExponentialSmoothing(
        series,
        trend='add',
        seasonal=seasonal_type,
        seasonal_periods=12,
        initialization_method='estimated'
    ).fit()

model_add = fit_model(ts, 'add')
model_mul = fit_model(ts, 'mul')

# Прогноз на 12 месяцев (округление до целых чисел)
forecast_add = np.round(model_add.forecast(12)).astype(int)
forecast_mul = np.round(model_mul.forecast(12)).astype(int)

# Настройка графика
plt.figure(figsize=(14, 7))
ax = plt.gca()

# Форматирование оси Y (разделители тысяч)
def format_y_axis(x, pos):
    return f"{int(x):,}".replace(",", " ")

ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Построение графиков
plt.plot(ts.index, ts, 'b-', label='Фактические данные', marker='o', linewidth=2)
plt.plot(forecast_add.index, forecast_add, 'g--', label='Аддитивная модель', marker='s', linewidth=2)
plt.plot(forecast_mul.index, forecast_mul, 'r--', label='Мультипликативная модель', marker='^', linewidth=2)

# Настройка отображения
plt.title('Прогноз количества атак на 2024 год', fontsize=16, pad=20)
plt.xlabel('Месяц', fontsize=14)
plt.ylabel('Количество атак', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()

# Функция для добавления подписей значений
def add_value_labels(ax, values, fmt_func):
    for x, y in zip(values.index, values):
        ax.text(x, y, fmt_func(y),
                ha='center', va='bottom',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Добавляем подписи ко всем графикам (черным цветом)
#add_value_labels(ax, ts, lambda x: f"{int(x):,}".replace(",", " "))
#add_value_labels(ax, pd.Series(forecast_add, index=forecast_add.index),
#               lambda x: f"{int(x):,}".replace(",", " "))
#add_value_labels(ax, pd.Series(forecast_mul, index=forecast_mul.index),
#               lambda x: f"{int(x):,}".replace(",", " "))

plt.show()

# Вывод прогнозов в терминал
results = pd.DataFrame({
    'Месяц': month_names,
    'Аддитивная модель': [f"{x:,}".replace(",", " ") for x in forecast_add],
    'Мультипликативная модель': [f"{x:,}".replace(",", " ") for x in forecast_mul]
})

print("\nПрогноз количества атак на 2024 год:")
print(results.to_string(index=False))
