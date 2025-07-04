![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14.1-8B0000?logo=mathworks&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?logo=tensorflow)
![PyMC](https://img.shields.io/badge/PyMC-5.12.0-FFD43B?logo=python&logoColor=blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-green?logo=xgboost)
![Keras](https://img.shields.io/badge/Keras-3.10.0-D00000?logo=keras)

# Ансамблевые модели прогнозирования временных рядов (Holt-Winters)
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
│   main.py              # Основной скрипт
│   config.py            # Конфигурация (исходные данные)
│   README.md            # Пользовательское описание
│   CHANGELOG.md         # Записи обновлений
│
├───core/                # Вычислительное ядро
│   │   models.py        # Функции моделей
│   │   calculations.py  # Вычисления и метрики
│   │   preprocessing.py # Подготовка данных
│
├───utils/
│    │   visualization.py # Визуализация
│  
└───data/
     │     holidays.csv   # Праздничные дни (РФ)
     │   exog_features.csv # Опциональные внешние признаки
 
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

## Модели и их работа с остатками

### 1. `HW_ARMIMA`
- **Тип**: Комбинация HW + ARIMA
- **Работа с остатками**: ARIMA моделирует остатки базовой HW модели
- **Формула**: `Прогноз = HW_forecast + ARIMA(residuals)`

### 2. `HW_LSTM`
- **Тип**: Гибридная нейросетевая модель
- **Работа с остатками**: LSTM обучается на остатках HW
- **Особенность**: Использует MinMax-нормализацию остатков

### 3. `hw_prophet_ensemble`
- **Тип**: Ансамбль с внешними факторами
- **Работа с остатками**: Prophet анализирует остатки с учетом праздников
- **Формула**: `Прогноз = HW_forecast + Prophet(residuals)`

### 4. `build_hw_tcn_model`
- **Тип**: Временные сверточные сети
- **Работа с остатками**: TCN архитектура для сложных паттернов в остатках
- **Особенность**: Каузальные свертки с дилатацией

### 5. `hw_bayesian_ensemble`
- **Тип**: Байесовский подход
- **Работа с остатками**: Моделирует остатки как Gaussian Random Walk
- **Выход**: Вероятностный прогноз

### 6. `clustered_hw`
- **Тип**: Кластерно-взвешенный подход
- **Работа с остатками**: Не использует остатки напрямую (анализ кластеров истории)
- **Особенность**: DTW-метрика для кластеризации

### 7. `wavelet_hw`
- **Тип**: Частотный анализ
- **Работа с остатками**: Вейвлет-разложение остатков
- **Фильтрация**: Подавление высокочастотного шума

### 8. `hw_xgboost_ensemble`
- **Тип**: Ансамбль с машинным обучением
- **Работа с остатками**: XGBoost моделирует остатки с использованием:
  - Лаговых признаков (если нет внешних данных)
  - Внешних факторов (экономические индикаторы, праздники и т.д.)
- **Формула**: `Прогноз = HW_forecast + XGBoost(residuals)`
- **Особенности**:
  - Автоматический отбор признаков (RFE)
  - Адаптация к наличию/отсутствию внешних данных
  ```python
  # Пример использования без внешних признаков:
  forecast = hw_xgboost_ensemble(ts, hw_model)
  
  # С внешними признаками:
  forecast = hw_xgboost_ensemble(ts, hw_model, exog_features)

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
