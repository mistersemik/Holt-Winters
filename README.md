![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14.1-8B0000?logo=mathworks&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?logo=tensorflow)
![PyMC](https://img.shields.io/badge/PyMC-5.12.0-FFD43B?logo=python&logoColor=blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-green?logo=xgboost)
![Keras](https://img.shields.io/badge/Keras-3.10.0-D00000?logo=keras)
![Prophet 1.1.7](https://img.shields.io/badge/Prophet-1.1.7-black?logo=facebook)
![pmdarima 2.0.4](https://img.shields.io/badge/pmdarima-2.0.4-blueviolet)


Статус последнего обновления:<br>
<img src="https://github.com/mistersemik/Holt-Winters/workflows/Python-Security-Scan/badge.svg?brach=master"><br>

# Гибридные модели прогнозирования временных рядов (Holt-Winters)
### 1. Остаточное обучение (Residual Forecasting)  
- HW + ARIMA
- HW + TCN
- HW + Prophet
### 2. Ансамбли
- Кластеризация остатков
- Байесовская модель
### 3. Гибридные методы
- Вейвлет-разложение ряда + прогноз HW для аппроксимационной компоненты
### 4. Наивный метод

![alt text](images/pict1.png)

## Научный контекст

Проект реализует **сравнительный анализ модификаций** классической модели Хольта-Винтерса, направленный на:

- Исследование эффективности комбинированных подходов
- Анализ адаптации модели к различным типам сезонности
- Разработку методов коррекции остатков

```diff
+ Основная гипотеза: Ансамблевые модификации модели Хольта-Винтерса 
+ демонстрируют бо́льшую точность по сравнению с базовой версией 
+ при работе с нестационарными сезонными рядами.
```


## Действующая структура проекта
```
/Holt-Winters
│   main.py              # Основной CLI-скрипт
│   config.yaml          # Основной конфиг (заменяет config.py)
│   README.md            # Документация
│   CHANGELOG.md         # Записи обновлений
│   requirements.txt     # Зависимости
│
├───core/
│   │   models.py        # Модели (HW_LSTM, HW_ARMIMA и др.)
│   │   calculations.py  # Метрики (MAE, RMSE)
│   │   preprocessing.py # Подготовка данных
│
├───utils/
│   │   visualization.py # Визуализация
│   │   config_loader.py # Загрузчик YAML
│
├───data/
│   │   historical.csv   # Основные исторические данные
│   │   actual.csv       # Актуальные данные для проверки
│   │   holidays.csv     # Праздники РФ
│   │   exog_features.csv # Внешние признаки (опционально)
│
└───configs/             # Доп. конфиги для экспериментов
    │   prod.yaml        # Продакшен-настройки
    │   dev.yaml         # Настройки для разработки
```

## Архитектурный подход

Все модели используют **комбинированную стратегию**, где:
1. Базовый прогноз строится моделью Хольта-Винтерса
2. Остатки анализируются дополнительными методами
3. Результаты интегрируются в финальный прогноз

```
[Исходный ряд]
│
├── [Holt-Winters] → [Базовый прогноз]
│
└── [Остатки] → [Доп.модель] → [Коррекция]
```

## Модели и их особенности

### 1. `HW_ARMIMA`
- **Тип**: Комбинация HW + ARIMA  
- **Работа с остатками**: ARIMA моделирует остатки базовой HW модели  
- **Формула**: `Прогноз = HW_forecast + ARIMA(residuals)`  
- **Применение**: Лучше всего подходит для рядов с устойчивыми линейными паттернами в остатках
- **Инициализация**: 
  ```python
  hw_model = ExponentialSmoothing(ts, trend='add', seasonal='add',seasonal_periods=12).fit()
  forecast = HW_ARMIMA(ts, hw_model)

### 2. `HW_LSTM`
- **Тип**: Гибридная нейросетевая модель  
- **Работа с остатками**: LSTM обучается на остатках HW  
- **Особенность**: Использует MinMax-нормализацию остатков  
- **Применение**: Эффективна для сложных нелинейных зависимостей в остатках  

### 3. `hw_prophet_ensemble`
- **Тип**: Ансамбль с внешними факторами  
- **Работа с остатками**: Prophet анализирует остатки с учетом праздников  
- **Формула**: `Прогноз = HW_forecast + Prophet(residuals)`  
- **Применение**: Оптимален для данных с известными внешними событиями  

### 4. `build_hw_tcn_model`
- **Тип**: Временные сверточные сети  
- **Работа с остатками**: TCN архитектура для сложных паттернов в остатках  
- **Особенность**: Каузальные свертки с дилатацией  
- **Применение**: Для рядов с долгосрочными зависимостями  

### 5. `hw_bayesian_ensemble`
- **Тип**: Байесовский подход  
- **Работа с остатками**: Моделирует остатки как Gaussian Random Walk  
- **Выход**: Вероятностный прогноз  
- **Применение**: Когда важна оценка неопределенности  

### 6. `clustered_hw`
- **Тип**: Кластерно-взвешенный подход  
- **Работа с остатками**: Не использует остатки напрямую (анализ кластеров истории)  
- **Особенность**: DTW-метрика для кластеризации  
- **Применение**: Для данных с ярко выраженными режимами/фазами  

### 7. `wavelet_hw`
- **Тип**: Частотный анализ  
- **Работа с остатками**: Вейвлет-разложение всего ряда (не только остатков)  
- **Фильтрация**: Подавление высокочастотного шума  
- **Применение**: Для рядов с мультимасштабными паттернами  

### 8. `hw_xgboost_ensemble`
- **Тип**: Ансамбль с машинным обучением  
- **Работа с остатками**: XGBoost моделирует остатки с использованием:  
  - Лаговых признаков (если нет внешних данных)  
  - Внешних факторов (экономические индикаторы, праздники и т.д.)  
- **Формула**: `Прогноз = HW_forecast + XGBoost(residuals)`  
- **Особенности**:  
  - Автоматический отбор признаков (RFE)  
  - Адаптация к наличию/отсутствию внешних данных  

### 9. `naive_forecast`
- **Тип**: Базовый метод  
- **Работа с остатками**: Не использует  
- **Логика**: Простое повторение последнего значения  
- **Применение**: Базовый benchmark для сравнения

### 10. `hw_garch`
- **Тип**: Моделирование волатильности остатков  
- **Работа с остатками**: GARCH(p=1, q=1) для кластеров волатильности  
- **Формула**: `Прогноз = HW_forecast + GARCH(residuals)`  
- **Применение**: Когда остатки демонстрируют периоды высокой/низкой волатильности  
- **Инициализация**:  
  ```python
  forecast = hw_garch(ts, hw_model)

```diff
! Важное отличие: 
- clustered_hw и wavelet_hw работают не с остатками, а с исходным рядом/кластерами
+ Остальные модели следуют схеме "HW + коррекция остатков"
```

## Ключевые особенности системы

1. **Модульность**:
   - Каждая модель может работать автономно
   - Возможность комбинирования подходов

2. **Отказоустойчивость**:
   ```python
   try:
       # Основная логика
   except:
       # Fallback на наивный прогноз

## Установка зависимостей

1. Создайте виртуальное окружение:
    ```bash
    python3.9 -m venv hw_env
    source hw_env/bin/activate  # Linux/Mac
    hw_env\Scripts\activate     # Windows

2. Установите зависимости
    ```bash
   pip install -r requirements.txt

### Ключевые зависимости:
- Базовые:  
  ![Python](https://img.shields.io/badge/python-3.8%2B-blue)
  ![Pandas](https://img.shields.io/badge/pandas-1.5%2B-blueviolet)
  ![NumPy](https://img.shields.io/badge/numpy-1.23%2B-013243)

- Моделирование:  
  ![ARCH](https://img.shields.io/badge/ARCH-5.3+-yellow)
  ![Statsmodels](https://img.shields.io/badge/statsmodels-0.14.1-8B0000)
  ![PyMC](https://img.shields.io/badge/PyMC-5.12.0-FFD43B)

- Нейросети:  
  ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00)
  ![Keras](https://img.shields.io/badge/Keras-3.10.0-D00000)