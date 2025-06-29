import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from keras.layers import Input, Conv1D, Dense
from keras.models import Model
import pymc as pm
from tslearn.clustering import TimeSeriesKMeans
from typing import Tuple
from warnings import warn
import pywt

def build_model(data, trend='add',seasonal_type='add', seasonal_periods=12):
    """
    Строит и обучает модель Хольта-Винтерса для временных рядов.

    Параметры:
    - data: pandas.Series - временной ряд для обучения
    - trend (str): 'add' для аддитивного или 'mul' для мультипликативного тренда
    - seasonal_type: str - тип сезонности ('add' - аддитивная, 'mul' - мультипликативная)
    - seasonal_periods: int - количество периодов в сезонном цикле (по умолчанию 12 для месячных данных)

    Возвращает:
    - Обученную модель ExponentialSmoothing
    """
    model = ExponentialSmoothing(
        data,
        trend=trend,
        seasonal=seasonal_type,
        seasonal_periods=seasonal_periods,
        initialization_method='heuristic',
        damped_trend=False
    ).fit()
    return model

def HW_ARMIMA(ts, hw_model):
    """
    Комбинированная модель Хольта-Винтерса и ARIMA для остатков.
    Использует преимущества обеих моделей: HW для основной компоненты и ARIMA для остатков.

    Параметры:
    - ts: pandas.Series - исходный временной ряд
    - hw_model: обученная модель Хольта-Винтерса

    Возвращает:
    - pandas.Series - комбинированный прогноз на 12 периодов с датами в индексе

    Алгоритм:
    1. Получает прогноз HW на 12 периодов
    2. Вычисляет остатки (факт - прогноз HW)
    3. Строит ARIMA модель на остатках
    4. Суммирует прогнозы HW и ARIMA

    """
    hw_forecast = hw_model.forecast(12) # Получаем прогноз Хольта-Винтерса
    residuals = ts - hw_model.fittedvalues # Вычисляем остатки (фактические значения - fitted values модели)

    arima_model = auto_arima( # Строим модель на остатках
        residuals.dropna(),
        seasonal=False,
        suppress_warnings=True,
        error_action='ignore'
    )

    arima_forecast = arima_model.predict(n_periods=12) # Прогнозируем остатки на 12 периодов
    forecast_dates = pd.date_range(
        start=ts.index[-1] + pd.DateOffset(months=1),
        periods=12,
        freq='MS'
    )

    return pd.Series(hw_forecast.values + arima_forecast, index=forecast_dates) # Комбинируем прогнозы

def HW_LSTM(ts, hw_model=None, trend='add', seasonal_type='add', n_steps=3, n_epochs=50, n_neurons=50):
    """
    Комбинированная модель Хольта-Винтерса + LSTM.

    Параметры:
    - ts: исходный временной ряд (pandas.Series)
    - hw_model: обученная модель Хольта-Винтерса
    - trend (str): 'add' для аддитивного или 'mul' для мультипликативного тренда
    - seasonal_type (str): 'add' для аддитивной или 'mul' для мультипликативной сезонности
    - n_steps: количество временных шагов для LSTM (по умолчанию 3)
    - n_epochs: количество эпох обучения LSTM (по умолчанию 50)
    - n_neurons: количество нейронов в LSTM слое (по умолчанию 50)

    Возвращает:
    - Прогноз на 12 периодов (pandas.Series)
    """

    if hw_model is None:
        hw_model = ExponentialSmoothing(
            ts,
            trend=trend,
            seasonal=seasonal_type,
            seasonal_periods=12
        ).fit()

    hw_forecast = hw_model.forecast(12)
    residuals = ts - hw_model.fittedvalues
    residuals = residuals.dropna().values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    residuals_scaled = scaler.fit_transform(residuals)

    def create_dataset(data, n_steps):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i + n_steps, 0])
            y.append(data[i + n_steps, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(residuals_scaled, n_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(n_neurons, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=n_epochs, verbose=0)

    last_residuals = residuals_scaled[-n_steps:].reshape(1, n_steps, 1)
    lstm_forecast_scaled = []
    for _ in range(12):
        pred = model.predict(last_residuals, verbose=0)
        lstm_forecast_scaled.append(pred[0, 0])
        last_residuals = np.append(last_residuals[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    # Исправление формы данных для inverse_transform
    lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast_scaled).reshape(-1, 1)).flatten()

    forecast_dates = pd.date_range(
        start=ts.index[-1] + pd.DateOffset(months=1),
        periods=12,
        freq='MS'
    )

    return pd.Series(hw_forecast.values + lstm_forecast, index=forecast_dates)

def hw_prophet_ensemble(ts, hw_model=None, holidays_df=None, trend='add', seasonal_type='add'):
    """
    Комбинированная модель Хольта-Винтерса и Prophet с выбором типа сезонности

    Параметры:
        ts (pd.Series): Временной ряд с DatetimeIndex
        hw_model (optional): Готовая модель ExponentialSmoothing (если None - будет создана новая)
        holidays_df (pd.DataFrame, optional): Даты праздников в формате Prophet
        - trend (str): 'add' для аддитивного или 'mul' для мультипликативного тренда
        seasonal_type (str): 'add' для аддитивной или 'mul' для мультипликативной сезонности

    Возвращает:
        pd.Series: Комбинированный прогноз на 12 периодов

    Пример использования:
    # Вариант 1: Классический вызов (создает модель внутри)
    forecast = hw_prophet_ensemble(ts, holidays_df, seasonal_type='add')

    # Вариант 2: С готовой моделью HW (оптимизированный)
    hw_model = ExponentialSmoothing(ts, seasonal='add').fit()
    forecast = hw_prophet_ensemble(ts, hw_model=hw_model, holidays_df)

    # Вариант 3: Без праздников
    forecast = hw_prophet_ensemble(ts, seasonal_type='mul')
    """
    # 1. Компонента Хольта-Винтерса (используем готовую модель или создаем новую)
    if hw_model is None:
        hw_model = ExponentialSmoothing(
            ts,
            trend=trend,
            seasonal=seasonal_type,
            seasonal_periods=12
        ).fit()

    # 2. Определяем тип сезонности для Prophet
    current_seasonal_type = hw_model.seasonal if hasattr(hw_model, 'seasonal') else seasonal_type

    # 3. Настройка Prophet
    prophet_model = Prophet(
        holidays=holidays_df,
        seasonality_mode='multiplicative' if current_seasonal_type == 'mul' else 'additive',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    # 4. Подготовка данных для Prophet (работаем с остатками)
    residuals = ts - hw_model.fittedvalues
    prophet_data = residuals.reset_index()
    prophet_data.columns = ['ds', 'y']

    # 5. Обучение и прогноз Prophet
    prophet_model.fit(prophet_data.dropna())
    future = prophet_model.make_future_dataframe(periods=12, freq='MS')
    prophet_forecast = prophet_model.predict(future)['yhat'][-12:]

    # 6. Комбинация прогнозов
    return hw_model.forecast(12) + prophet_forecast.values

# def hw_xgboost_ensemble(ts, exog_features):
#     # HW компонента
#     hw = ExponentialSmoothing(ts).fit()
#     residuals = ts - hw.fittedvalues
#
#     # XGBoost на остатках
#     model = XGBRegressor()
#     selector = RFE(model, n_features_to_select=5)
#     selector.fit(exog_features[:-12], residuals.dropna())
#     xgb_forecast = selector.predict(exog_features[-12:])
#
#     return hw.forecast(12) + xgb_forecast

def build_hw_tcn_model(hw_model, ts, n_steps=24, forecast_steps=12):
    """
    Создает и возвращает комбинированный прогноз HW-TCN

    Параметры:
        hw_model: обученная модель Хольта-Винтерса
        ts: исходный временной ряд (pd.Series)
        n_steps: размер окна для TCN
        forecast_steps: количество шагов прогноза

    Возвращает:
        tuple: (combined_forecast, hw_forecast, tcn_forecast, forecast_dates)
    """
    # Вычисляем остатки модели
    residuals = ts - hw_model.fittedvalues

    # Функция для создания последовательностей (вложенная)
    def create_sequences(data, n_steps):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i + n_steps])
            y.append(data[i + n_steps])
        return np.array(X), np.array(y)

    def build_tcn_residual_model(input_shape):
        """Создает архитектуру TCN модели для остатков"""
        inputs = Input(shape=input_shape)
        x = Conv1D(64, kernel_size=3, dilation_rate=1, padding='causal')(inputs)
        x = Conv1D(64, kernel_size=3, dilation_rate=2, padding='causal')(x)
        outputs = Dense(1)(x)
        return Model(inputs, outputs)

    # Подготовка данных для TCN
    X, y = create_sequences(residuals.dropna().values, n_steps=n_steps)
    X = X.reshape(*X.shape, 1)

    # Построение и обучение TCN
    tcn_model = build_tcn_residual_model(input_shape=(n_steps, 1))
    tcn_model.compile(optimizer='adam', loss='mse')
    tcn_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    # Прогнозирование остатков
    last_sequence = residuals[-n_steps:].values.reshape(1, n_steps, 1)
    tcn_forecast = []
    for _ in range(forecast_steps):
        pred = tcn_model.predict(last_sequence, verbose=0)[0, -1, 0]
        tcn_forecast.append(pred)
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = pred

    # Прогноз основной модели
    hw_forecast = hw_model.forecast(forecast_steps)

    # Комбинированный прогноз
    combined_forecast = hw_forecast.values + tcn_forecast

    # Даты для прогноза
    forecast_dates = pd.date_range(
        start=ts.index[-1] + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq='MS'
    )

    return combined_forecast, hw_forecast, tcn_forecast, forecast_dates

def hw_bayesian_ensemble(ts, hw_model, forecast_steps=12):
    """
    Байесовский ансамбль с моделью Хольта-Винтерса

    Параметры:
        ts: pandas.Series - исходный временной ряд
        hw_model: обученная модель Хольта-Винтерса
        forecast_steps: int - количество прогнозируемых периодов

    Возвращает:
        tuple: (forecast_series, trace) - прогноз и trace модели
    """
    # 1. Получаем fitted values и остатки
    fitted = hw_model.fittedvalues
    residuals = ts - fitted

    # 2. Создаем байесовскую модель для остатков
    with pm.Model() as model:
        # Моделируем остатки как случайное блуждание
        sigma = pm.HalfNormal('sigma', 1)
        resid_process = pm.GaussianRandomWalk(
            'resid_process',
            sigma=sigma,
            shape=len(residuals)
        )

        # Модель наблюдений
        pm.Normal(
            'obs',
            mu=resid_process,
            sigma=0.1,
            observed=residuals.values
        )

        # Сэмплирование
        trace = pm.sample(
            1000,
            tune=1000,
            chains=2,
            target_accept=0.9,
            progressbar=False
        )

        # Прогнозирование остатков
        with model:
            # Создаем новые узлы для прогноза
            resid_forecast = pm.GaussianRandomWalk(
                'resid_forecast',
                sigma=sigma,
                shape=forecast_steps
            )

            # Генерируем прогноз
            forecast = pm.sample_posterior_predictive(
                trace,
                var_names=['resid_forecast']
            )

    # 3. Комбинируем с прогнозом HW
    hw_forecast = hw_model.forecast(forecast_steps)
    bayesian_resid_forecast = forecast.posterior_predictive['resid_forecast'].mean(axis=(0, 1))
    combined_forecast = hw_forecast + bayesian_resid_forecast

    # 4. Формируем результат
    forecast_dates = pd.date_range(
        start=ts.index[-1] + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq='MS'
    )

    return pd.Series(combined_forecast, index=forecast_dates), trace

def clustered_hw(ts, n_clusters=3, hw_model=None, trend='add', seasonal_type='add'):
    """
    Упрощённая кластерно-взвешенная модель Хольта-Винтерса

    Параметры:
        ts: Исходный временной ряд (pandas Series)
        n_clusters: Количество кластеров (по умолчанию 3)
        hw_model: Готовая модель ExponentialSmoothing (опционально)
        trend: 'add' или 'mul' тренд
        seasonal_type: 'add' или 'mul' сезонность

    Возвращает:
        Кортеж: (прогноз, веса кластеров) - оба pandas Series
    """
    # 1. Проверка входных данных
    if not isinstance(ts, pd.Series):
        raise TypeError("ts должен быть pandas Series")

    if not isinstance(ts.index, pd.DatetimeIndex):
        raise ValueError("Индекс ts должен быть DatetimeIndex")

    ts_values = ts.values
    ts_length = len(ts_values)

    # Минимальное количество данных - 1 год
    if ts_length < 12:
        raise ValueError("Необходим минимум 12 месяцев данных")

    # Проверка параметров модели
    if trend not in ['add', 'mul']:
        raise ValueError("trend должен быть 'add' или 'mul'")

    if seasonal_type not in ['add', 'mul']:
        raise ValueError("seasonal_type должен быть 'add' или 'mul'")

    # 2. Если модель не предоставлена, инициализируем новую
    if hw_model is None:
        hw_model = ExponentialSmoothing(
            ts_values,
            trend=trend,
            seasonal=seasonal_type,
            seasonal_periods=12
        ).fit()
    else:
        # Проверка переданной модели
        if not isinstance(hw_model, ExponentialSmoothing):
            raise TypeError("hw_model должен быть ExponentialSmoothing")

        if not hasattr(hw_model, 'fittedvalues'):
            raise ValueError("Модель должна быть обучена (иметь fittedvalues)")

    # 3. Группировка по годам с дополнением
    n_years = ts_length // 12
    remainder = ts_length % 12

    if remainder > 0:
        # Дополняем последний год средними значениями
        last_year_mean = np.mean(ts_values[-remainder:])
        padded_values = np.concatenate([
            ts_values,
            np.full(12 - remainder, last_year_mean)
        ])
        n_years += 1
    else:
        padded_values = ts_values

    X = padded_values.reshape(n_years, 12)

    # 3. Автокоррекция числа кластеров
    n_clusters = min(n_clusters, n_years)
    if n_clusters < 1:
        n_clusters = 1

    # 4. Кластеризация с обработкой ошибок
    try:
        km = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric="dtw",
            random_state=42,
            n_init=3
        )
        clusters = km.fit_predict(X.reshape(n_years, 12, 1))
    except Exception as e:
        warn(f"Ошибка кластеризации: {str(e)}. Используется один кластер.")
        clusters = np.zeros(n_years)

    # 5. Прогнозирование для кластеров
    forecasts = []
    cluster_weights = []

    for c in range(n_clusters):
        cluster_mask = (clusters == c)
        if sum(cluster_mask) == 0:
            continue

        cluster_data = X[cluster_mask]
        cluster_weight = len(cluster_data) / n_years
        cluster_weights.append(cluster_weight)

        # Усредненный ряд кластера
        cluster_ts = pd.Series(cluster_data.mean(axis=0))

        # Простая модель для кластера
        try:
            forecast = cluster_ts.values[-12:]  # Используем последний год как прогноз
            forecasts.append(forecast)
        except:
            forecasts.append(np.array([cluster_ts.mean()] * 12))

    # 6. Взвешенная комбинация прогнозов
    if len(forecasts) == 0:
        final_forecast = np.array([ts_values.mean()] * 12)
    else:
        final_forecast = np.average(forecasts, axis=0, weights=cluster_weights)

    # 7. Формирование результата
    forecast_dates = pd.date_range(
        start=ts.index[-1] + pd.DateOffset(months=1),
        periods=12,
        freq='MS'
    )

    return pd.Series(final_forecast, index=forecast_dates), pd.Series(
        cluster_weights,
        index=[f'Cluster{i + 1}' for i in range(len(cluster_weights))]
    )

def wavelet_hw(ts, wavelet='db4', hw_model=None, trend='add', seasonal_type='add', forecast_periods=12):
    """
    Улучшенная вейвлет-модель Хольта-Винтерса

    Параметры:
        ts: pd.Series - временной ряд с DatetimeIndex
        wavelet: str - тип вейвлета ('db4', 'haar' и др.)
        hw_model: модель ExponentialSmoothing (опционально)
        trend: str - 'add' или 'mul' тренд
        seasonal_type: str - 'add' или 'mul' сезонность
        forecast_periods: int - количество периодов прогноза

    Возвращает:
        pd.Series - прогноз на указанное число периодов
    """
    # Проверки входных данных
    if not isinstance(ts, pd.Series):
        raise TypeError("ts должен быть pandas Series")
    if not isinstance(ts.index, pd.DatetimeIndex):
        raise ValueError("Индекс ts должен быть DatetimeIndex")
    if len(ts) < 24:  # Увеличили минимальное количество данных
        raise ValueError("Необходим минимум 24 наблюдения (2 полных сезона)")

    try:
        # Нормализация данных
        ts_values = ts.values.astype(float)
        mean_val, std_val = ts_values.mean(), ts_values.std()
        ts_normalized = (ts_values - mean_val) / std_val if std_val > 0 else ts_values - mean_val

        # Вейвлет-разложение
        max_level = pywt.dwt_max_level(len(ts_normalized), pywt.Wavelet(wavelet))
        level = min(3, max_level) if max_level is not None else 3
        coeffs = pywt.wavedec(ts_normalized, wavelet, level=level, mode='per')

        # Прогнозирование компонент
        forecast_coeffs = []
        for i, coeff in enumerate(coeffs):
            if i == 0:  # Только для аппроксимационной компоненты
                if hw_model is None:
                    try:
                        model = ExponentialSmoothing(
                            coeff,
                            trend=trend,
                            seasonal=seasonal_type if len(coeff) >= 24 else None,
                            # Отключаем сезонность для коротких рядов
                            seasonal_periods=12
                        ).fit()
                    except ValueError:
                        # Fallback для случаев, когда не хватает данных для сезонности
                        model = ExponentialSmoothing(
                            coeff,
                            trend=trend,
                            seasonal=None,
                            seasonal_periods=12
                        ).fit()
                else:
                    model = hw_model
                fc = model.forecast(forecast_periods)
            else:
                fc = np.zeros(forecast_periods)
            forecast_coeffs.append(fc)

        # Вейвлет-реконструкция
        forecast_normalized = pywt.waverec(forecast_coeffs, wavelet)[:forecast_periods]

        # Обратное преобразование нормализации
        forecast = forecast_normalized * std_val + mean_val

        # Формирование результата
        forecast_dates = pd.date_range(
            start=ts.index[-1] + pd.DateOffset(months=1),
            periods=forecast_periods,
            freq='MS'
        )

        return pd.Series(forecast, index=forecast_dates)

    except Exception as e:
        warn(f"Ошибка в wavelet_hw: {str(e)}. Возвращаем наивный прогноз.")
        # Fallback - наивный прогноз
        return pd.Series([ts.iloc[-1]] * forecast_periods, index=pd.date_range(
            start=ts.index[-1] + pd.DateOffset(months=1),
            periods=forecast_periods,
            freq='MS'
        ))