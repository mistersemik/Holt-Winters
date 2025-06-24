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

#
# from xgboost import XGBRegressor
# from sklearn.feature_selection import RFE
#
#
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

# import pymc as pm
#
# def hw_bayesian_ensemble(ts):
#     with pm.Model():
#         # HW сезонность как фиксированный компонент
#         hw_seasonal = pm.Deterministic('hw_seasonal', hw_model.seasonal)
#
#         # Байесовский тренд
#         trend = pm.GaussianRandomWalk('trend', sigma=0.1, shape=len(ts))
#
#         # Комбинированная модель
#         y_obs = pm.Normal('y_obs',
#                           mu=hw_seasonal + trend,
#                           observed=ts)
#
#         trace = pm.sample(1000)
#         forecast = pm.sample_posterior_predictive(trace, var_names=['trend'])
#     return forecast['trend'].mean(axis=0)[-12:]
#
#
# from tslearn.clustering import TimeSeriesKMeans
#
#
# def clustered_hw(ts, n_clusters=3):
#     # Преобразуем в 3D-массив (примеры, время, 1 признак)
#     X = ts.values.reshape(-1, 12, 1)  # Группируем по годам
#
#     # Кластеризация
#     km = TimeSeriesKMeans(n_clusters=n_clusters)
#     clusters = km.fit_predict(X)
#
#     # Прогноз для каждого кластера
#     forecasts = []
#     for c in range(n_clusters):
#         cluster_ts = pd.Series(X[clusters == c].mean(axis=0).flatten())
#         hw = ExponentialSmoothing(cluster_ts, seasonal='add').fit()
#         forecasts.append(hw.forecast(12))
#
#     return np.mean(forecasts, axis=0)
#
#
# import pywt
#
#
# def wavelet_hw(ts, wavelet='db4'):
#     # Декомпозиция
#     coeffs = pywt.wavedec(ts, wavelet, level=3)
#
#     # Прогноз для каждой компоненты
#     reconstructed = []
#     for i, coeff in enumerate(coeffs):
#         if i == 0:  # Аппроксимационная компонента
#             hw = ExponentialSmoothing(coeff, seasonal=None).fit()
#             recon = hw.forecast(len(coeff))
#         else:  # Детализирующие компоненты
#             recon = np.zeros_like(coeff)  # Шумовые компоненты не прогнозируем
#         reconstructed.append(recon)
#
#     # Реконструкция
#     return pywt.waverec(reconstructed, wavelet)[:12]