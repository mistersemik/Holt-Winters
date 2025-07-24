import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from tensorflow.keras.layers import LSTM, Dense, Input, Conv1D
from keras.models import Model
import pymc as pm
from tslearn.clustering import TimeSeriesKMeans
from warnings import warn
import pywt
from arch import arch_model
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.nonparametric.kernel_regression import KernelReg


def HW_ARMIMA(ts: pd.Series, hw_model: ExponentialSmoothing):
    """
    Комбинированная модель Хольта-Винтерса и ARIMA для остатков.
    Использует преимущества обеих моделей:
    HW для основной компоненты и ARIMA для остатков.

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

    Пример:
        >>> model = ExponentialSmoothing(ts, trend='add', seasonal='add').fit()
        >>> forecast = HW_ARMIMA(ts, model)

    """

    if hw_model is None:
        raise TypeError(
            "hw_model является обязательным параметром. "
            "Пример: hw_model = build_model(data,"
            "trend='add', seasonal_type='add')"
        )

    # Получаем прогноз Хольта-Винтерса
    hw_forecast = hw_model.forecast(12)

    # Вычисляем остатки (фактические значения - fitted values модели)
    residuals = ts - hw_model.fittedvalues

    # Строим модель на остатках
    arima_model = auto_arima(
        residuals.dropna(),
        seasonal=False,
        suppress_warnings=True,
        error_action="ignore",
    )

    # Прогнозируем остатки на 12 периодов
    arima_forecast = arima_model.predict(n_periods=12)
    forecast_dates = pd.date_range(
        start=ts.index[-1] + pd.DateOffset(months=1), periods=12, freq="MS"
    )

    # Комбинируем прогнозы
    return pd.Series(
        hw_forecast.values + arima_forecast, index=forecast_dates
    )


def HW_LSTM(
    ts: pd.Series,
    hw_model: ExponentialSmoothing,
    n_steps=3,
    n_epochs=50,
    n_neurons=50,
):
    """
    Комбинированная модель Хольта-Винтерса + LSTM.

    Параметры:
    - ts: исходный временной ряд (pandas.Series)
    - hw_model: обученная модель Хольта-Винтерса
    - n_steps: количество временных шагов для LSTM (по умолчанию 3)
    - n_epochs: количество эпох обучения LSTM (по умолчанию 50)
    - n_neurons: количество нейронов в LSTM слое (по умолчанию 50)

    Возвращает:
    - Прогноз на 12 периодов (pandas.Series)
    """

    if hw_model is None:
        raise TypeError(
            "hw_model является обязательным параметром. "
            "Пример: hw_model = build_model(data,"
            "trend='add', seasonal_type='add')"
        )

    hw_forecast = hw_model.forecast(12)
    residuals = ts - hw_model.fittedvalues
    residuals = residuals.dropna().values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    residuals_scaled = scaler.fit_transform(residuals)

    def create_dataset(data, n_steps):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i : i + n_steps, 0])
            y.append(data[i + n_steps, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(residuals_scaled, n_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([LSTM(n_neurons, activation="relu"), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=n_epochs, verbose=0)

    last_residuals = residuals_scaled[-n_steps:].reshape(1, n_steps, 1)
    lstm_forecast_scaled = []
    for _ in range(12):
        pred = model.predict(last_residuals, verbose=0)
        lstm_forecast_scaled.append(pred[0, 0])
        last_residuals = np.append(
            last_residuals[:, 1:, :], pred.reshape(1, 1, 1), axis=1
        )

    # Исправление формы данных для inverse_transform
    lstm_forecast = scaler.inverse_transform(
        np.array(lstm_forecast_scaled).reshape(-1, 1)
    ).flatten()

    forecast_dates = pd.date_range(
        start=ts.index[-1] + pd.DateOffset(months=1), periods=12, freq="MS"
    )

    return pd.Series(hw_forecast.values + lstm_forecast, index=forecast_dates)


def hw_prophet_ensemble(
    ts: pd.Series, hw_model: ExponentialSmoothing, holidays_df=None
):
    """
    Комбинированная модель Хольта-Винтерса и Prophet для временных рядов

    Параметры:
        ts (pd.Series): Временной ряд с DatetimeIndex
        hw_model: Обученная модель ExponentialSmoothing (обязательный параметр)
        holidays_df (pd.DataFrame,
        optional): Даты праздников в формате Prophet:

            - Обязательные колонки: 'ds' (дата), 'holiday' (название праздника)

    Возвращает:
        pd.Series: Прогноз на 12 периодов с DatetimeIndex

    Исключения:
        TypeError: Если hw_model не предоставлена
        ValueError: Если входные данные некорректны

    Пример использования:
    >>> hw_model = build_model(ts, trend='add', seasonal_type='add')
    >>> holidays = pd.DataFrame({
    ...     'ds': pd.to_datetime(['2023-01-01', '2023-12-31']),
    ...     'holiday': ['Новый год', 'Канун Нового года']
    ... })
    >>> forecast = hw_prophet_ensemble(ts, hw_model, holidays_df=holidays)
    """

    if hw_model is None:
        raise TypeError(
            "hw_model является обязательным параметром. "
            "Пример: hw_model = build_model(data,"
            "trend='add', seasonal_type='add')"
        )
    if not isinstance(ts, pd.Series):
        raise TypeError("ts должен быть pandas Series")
    if not isinstance(ts.index, pd.DatetimeIndex):
        raise ValueError("Индекс ts должен быть DatetimeIndex")

    # Определение типа сезонности из модели HW
    seasonal_mode = (
        "multiplicative"
        if getattr(hw_model, "seasonal", None) == "mul"
        else "additive"
    )

    # Инициализация Prophet
    model = Prophet(
        holidays=holidays_df,
        seasonality_mode=seasonal_mode,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )

    # Работа с остатками
    residuals = ts - hw_model.fittedvalues
    prophet_data = residuals.reset_index()
    prophet_data.columns = ["ds", "y"]

    # Обучение и прогноз
    model.fit(prophet_data.dropna())
    future = model.make_future_dataframe(periods=12, freq="MS")
    prophet_forecast = model.predict(future)["yhat"][-12:].values

    # Комбинированный результат
    forecast_dates = pd.date_range(
        start=ts.index[-1] + pd.DateOffset(months=1), periods=12, freq="MS"
    )

    return pd.Series(
        hw_model.forecast(12).values + prophet_forecast, index=forecast_dates
    )


def hw_xgboost_ensemble(
    ts: pd.Series,
    hw_model: ExponentialSmoothing,
    exog_features=None,
    forecast_steps=12,
):
    """
    Улучшенная версия комбинированной модели HW + XGBoost

    Параметры:
        ts: pd.Series - временной ряд с DatetimeIndex
        hw_model: обученная модель Хольта-Винтерса
        exog_features: pd.DataFrame - внешние признаки (по умолчанию None)
        forecast_steps: int - количество периодов прогноза

    Возвращает:
        pd.Series - комбинированный прогноз
    """
    try:
        # 1. Проверка входных данных
        if hw_model is None:
            raise ValueError("hw_model не может быть None")

        if len(ts) < 24:  # Минимум 2 года данных
            raise ValueError(
                "Недостаточно данных " "для обучения (минимум 24 периода)"
            )

        # 2. Прогноз HW
        hw_forecast = hw_model.forecast(forecast_steps)

        # 3. Вычисление остатков
        residuals = ts - hw_model.fittedvalues
        residuals = residuals.dropna()

        # 4. Подготовка признаков
        if exog_features is None:
            # Создаем лаговые признаки с защитой от недостатка данных
            # 3 лага + прогнозируемые периоды
            min_required_length = 4 + forecast_steps
            if len(residuals) < min_required_length:
                raise ValueError(
                    f"Недостаточно"
                    f"данных для лагов. Нужно"
                    f"{min_required_length}, "
                    f"есть {len(residuals)}"
                )

            lag_data = pd.DataFrame(
                {
                    "lag1": residuals.shift(1),
                    "lag2": residuals.shift(2),
                    "lag3": residuals.shift(3),
                    "target": residuals,
                }
            ).dropna()

            # Разделение на обучающую и тестовую выборки
            X = lag_data[["lag1", "lag2", "lag3"]]
            y = lag_data["target"]

            split_point = len(X) - forecast_steps
            if split_point <= 0:
                raise ValueError(
                    "Недостаточно данных " "для разделения на train/test"
                )

            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train = y.iloc[:split_point]

        else:
            # Проверка внешних признаков
            if not isinstance(exog_features, pd.DataFrame):
                raise TypeError("exog_features должен быть DataFrame")

            if len(exog_features) != len(ts):
                exog_features = exog_features.reindex(ts.index)

            X_train = exog_features.iloc[:-forecast_steps]
            y_train = residuals.iloc[:-forecast_steps]
            X_test = exog_features.iloc[-forecast_steps:]

        # 5. Обучение модели
        from xgboost import XGBRegressor

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )

        model.fit(X_train, y_train)

        # 6. Прогнозирование остатков
        resid_forecast = model.predict(X_test)

        # 7. Комбинирование прогнозов
        combined_forecast = hw_forecast.values + resid_forecast

        # 8. Формирование результата
        forecast_dates = pd.date_range(
            start=ts.index[-1] + pd.DateOffset(months=1),
            periods=forecast_steps,
            freq="MS",
        )

        return pd.Series(combined_forecast, index=forecast_dates)

    except Exception as e:
        print(f"\nОшибка в hw_xgboost_ensemble: {str(e)}")
        print("Возвращаем прогноз только по модели Хольта-Винтерса")
        return hw_model.forecast(forecast_steps)


def build_hw_tcn_model(
    hw_model: ExponentialSmoothing,
    ts: pd.Series,
    n_steps=24,
    forecast_steps=12,
):
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

    if hw_model is None:
        raise TypeError(
            "hw_model является обязательным параметром. "
            "Пример: hw_model = build_model(data,"
            "trend='add', seasonal_type='add')"
        )

    # Вычисляем остатки модели
    residuals = ts - hw_model.fittedvalues

    # Функция для создания последовательностей (вложенная)
    def create_sequences(data, n_steps):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i : i + n_steps])
            y.append(data[i + n_steps])
        return np.array(X), np.array(y)

    def build_tcn_residual_model(input_shape):
        """Создает архитектуру TCN модели для остатков"""
        inputs = Input(shape=input_shape)
        x = Conv1D(64, kernel_size=3, dilation_rate=1, padding="causal")(
            inputs
        )
        x = Conv1D(64, kernel_size=3, dilation_rate=2, padding="causal")(x)
        outputs = Dense(1)(x)
        return Model(inputs, outputs)

    # Подготовка данных для TCN
    X, y = create_sequences(residuals.dropna().values, n_steps=n_steps)
    X = X.reshape(*X.shape, 1)

    # Построение и обучение TCN
    tcn_model = build_tcn_residual_model(input_shape=(n_steps, 1))
    tcn_model.compile(optimizer="adam", loss="mse")
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
        freq="MS",
    )

    return combined_forecast, hw_forecast, tcn_forecast, forecast_dates


def hw_bayesian_ensemble(
    ts: pd.Series, hw_model: ExponentialSmoothing, forecast_steps=12
):
    """
    Байесовский ансамбль с моделью Хольта-Винтерса

    Комбинирует прогноз модели Хольта-Винтерса с байесовской моделью для остатков,
    что позволяет учитывать неопределенность и улучшить точность прогнозирования.

    Параметры:
        ts: pandas.Series
            Исходный временной ряд с DatetimeIndex
        hw_model: ExponentialSmoothing
            Обученная модель Хольта-Винтерса (обязательный параметр)
        forecast_steps: int, optional
            Количество прогнозируемых периодов (по умолчанию 12)

    Возвращает:
        tuple: (forecast_series, trace)
            - forecast_series: pd.Series - комбинированный прогноз на заданное число периодов
            - trace: pymc.MultiTrace - результаты байесовского моделирования (None в случае ошибки)

    Алгоритм работы:
        1. Вычисляет остатки (разница между фактическими значениями и прогнозом HW)
        2. Строит байесовскую модель (AR(1) процесс) для остатков
        3. Прогнозирует остатки на заданное число периодов
        4. Комбинирует прогнозы основной модели и остатков
        5. Возвращает итоговый прогноз и результаты моделирования

    Пример использования:
        >>> hw_model = ExponentialSmoothing(ts, trend='add', seasonal='add').fit()
        >>> forecast, trace = hw_bayesian_ensemble(ts, hw_model)
    """
    try:
        # Получаем fitted values и остатки
        fitted = hw_model.fittedvalues
        residuals = ts - fitted

        with pm.Model() as model:
            # Моделируем остатки как случайное блуждание
            sigma = pm.HalfNormal("sigma", 1)
            resid_process = pm.GaussianRandomWalk(
                "resid_process", sigma=sigma, shape=len(residuals)
            )

            # Модель наблюдений
            pm.Normal(
                "obs", mu=resid_process, sigma=0.1, observed=residuals.values
            )

            # Сэмплирование
            trace = pm.sample(
                1000,
                tune=1000,
                chains=2,
                target_accept=0.9,
                progressbar=False,
            )

            # Прогнозирование остатков
            with model:
                resid_forecast = pm.GaussianRandomWalk(
                    "resid_forecast", sigma=sigma, shape=forecast_steps
                )

                # Генерируем прогноз
                forecast = pm.sample_posterior_predictive(
                    trace, var_names=["resid_forecast"]
                )

        # 3. Комбинируем с прогнозом HW
        hw_forecast = hw_model.forecast(forecast_steps)
        bayesian_resid_forecast = forecast.posterior_predictive[
            "resid_forecast"
        ].mean(axis=(0, 1))

        combined_forecast = hw_forecast + bayesian_resid_forecast

        # 4. Формируем результат
        forecast_dates = pd.date_range(
            start=ts.index[-1] + pd.DateOffset(months=1),
            periods=forecast_steps,
            freq="MS",
        )

        return pd.Series(combined_forecast, index=forecast_dates), trace

    except Exception as e:
        print(f"Ошибка в hw_bayesian_ensemble: {str(e)}")
        # Возвращаем наивный прогноз в случае ошибки
        naive_fc = naive_forecast(ts, forecast_steps)
        return naive_fc, None


def clustered_hw(ts: pd.Series, hw_model: ExponentialSmoothing, n_clusters=3):
    """
    Гибридная модель: Хольт-Винтерс + кластеризация остатков.
    Работает по схеме:
        [Holt-Winters] → [Базовый прогноз]
        [Остатки] → [Кластеризация] → [Коррекция]

    Параметры:
        ts: pd.Series - временной ряд
        hw_model: обученная модель ExponentialSmoothing
        n_clusters: int - число кластеров для остатков

    Возвращает:
        tuple: (forecast, cluster_weights)
    """

    if hw_model is None:
        raise ValueError(
            "hw_model является обязательным параметром. "
            "Пример: hw_model = build_model(data,"
            "trend='add', seasonal_type='add')"
        )

    if len(ts) < 24:
        raise ValueError("Нужно минимум 24 периода данных")

    # 1. Базовый прогноз HW
    hw_forecast = hw_model.forecast(12)

    # 2. Вычисление остатков
    residuals = ts - hw_model.fittedvalues
    residuals = residuals.dropna()

    # 3. Подготовка остатков для кластеризации (группировка по годам)
    resid_values = residuals.values
    n_years = len(resid_values) // 12
    X = resid_values[: n_years * 12].reshape(n_years, 12)

    # Автокоррекция числа кластеров
    n_clusters = min(n_clusters, n_years)

    # 4. Кластеризация остатков
    try:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X.T).T

        km = TimeSeriesKMeans(
            n_clusters=n_clusters, metric="dtw", random_state=42
        )
        clusters = km.fit_predict(X_scaled.reshape(n_years, 12, 1))
    except Exception as e:
        warn(f"Ошибка кластеризации: {e}. Используем один кластер.")
        clusters = np.zeros(n_years)

    # 5. Прогнозирование поправок по кластерам
    corrections = []
    cluster_weights = []

    for cluster_id in range(n_clusters):
        if not any(clusters == cluster_id):
            continue

        # Данные кластера (последние остатки)
        # Берем последние 6 месяцев
        cluster_data = X[clusters == cluster_id][:, -6:]

        # Средняя поправка по кластеру
        # Усредняем последние 3 месяца
        correction = np.mean(cluster_data, axis=0)[-3:].mean()
        corrections.append(correction)

        # Вес = доля кластера
        cluster_weights.append(np.sum(clusters == cluster_id) / n_years)

    # 6. Взвешенная поправка
    if not corrections:
        final_correction = 0
        cluster_weights = [1.0]
    else:
        final_correction = np.average(corrections, weights=cluster_weights)

    # 7. Итоговый прогноз
    forecast = hw_forecast + final_correction

    return (
        forecast,
        pd.Series(
            cluster_weights,
            index=[f"Cluster_{i}" for i in range(len(cluster_weights))],
        ),
    )


def wavelet_hw(
    ts: pd.Series,
    hw_model: ExponentialSmoothing,
    wavelet="sym5",  # Изменили на sym5 как более стабильный вариант
    forecast_periods=12,
):
    """
    Окончательная стабильная версия вейвлет-модели Хольта-Винтерса
    с полной обработкой всех ошибок

    Параметры:
        ts: pd.Series - временной ряд с DatetimeIndex
        hw_model: обученная модель Хольта-Винтерса
        wavelet: str - тип вейвлета (рекомендуется 'sym5' или 'db4')
        forecast_periods: int - количество периодов прогноза

    Возвращает:
        pd.Series - прогноз на указанное число периодов
    """
    try:
        # 1. Инициализация и проверка данных
        if not isinstance(ts, pd.Series):
            ts = pd.Series(ts)

        # Установка месячной частоты без предупреждений
        try:
            ts = ts.asfreq("MS")
        except:
            raise ValueError(
                "Невозможно установить месячную частоту для ряда"
            )

        if len(ts) < 24:  # Вернули к 24 как минимальному reasonable значению
            raise ValueError("Требуется минимум 24 месяца данных")

        # 2. Вычисление остатков с проверкой
        residuals = ts - hw_model.fittedvalues
        residuals = residuals.dropna()

        if len(residuals) < 12:
            raise ValueError("Недостаточно остатков для анализа")

        # 3. Нормализация с защитой
        scaler = StandardScaler()
        try:
            residuals_norm = scaler.fit_transform(
                residuals.values.reshape(-1, 1)
            ).flatten()
        except:
            residuals_norm = residuals.values / (np.std(residuals) + 1e-6)

        # 4. Вейвлет-разложение с безопасными параметрами
        try:
            level = min(2, pywt.dwt_max_level(len(residuals_norm), wavelet))
            coeffs = pywt.wavedec(
                residuals_norm, wavelet, level=level, mode="periodization"
            )
        except:
            # Если вейвлет-разложение не удалось, используем только HW
            warn(
                "Не удалось выполнить вейвлет-разложение. Используем HW прогноз"
            )
            return hw_model.forecast(forecast_periods)

        # 5. Упрощенное и безопасное прогнозирование компонент
        forecast_coeffs = []
        for i, coeff in enumerate(coeffs):
            if i == 0:  # Только для аппроксимационной компоненты
                try:
                    model = auto_arima(
                        coeff, seasonal=False, suppress_warnings=True
                    )
                    fc = model.predict(n_periods=forecast_periods)
                except:
                    fc = np.ones(forecast_periods) * np.mean(coeff)
            else:
                fc = np.zeros(forecast_periods)
            forecast_coeffs.append(fc)

        # 6. Гарантированно рабочая реконструкция
        try:
            # Простое сложение компонент вместо waverec
            forecast_norm = np.sum(forecast_coeffs, axis=0)[:forecast_periods]

            # Если все нули (может случиться для детализирующих компонент)
            if np.all(forecast_norm == 0):
                forecast_norm = np.ones(forecast_periods) * np.mean(
                    residuals_norm
                )
        except:
            forecast_norm = np.zeros(forecast_periods)

        # 7. Обратное преобразование и комбинирование
        try:
            forecast_resid = scaler.inverse_transform(
                forecast_norm.reshape(-1, 1)
            ).flatten()
        except:
            forecast_resid = forecast_norm * (
                np.std(residuals) + 1e-6
            ) + np.mean(residuals)

        hw_forecast = hw_model.forecast(forecast_periods)
        combined_forecast = hw_forecast + forecast_resid

        # 8. Формирование результата
        forecast_dates = pd.date_range(
            start=ts.index[-1] + pd.DateOffset(months=1),
            periods=forecast_periods,
            freq="MS",
        )

        return pd.Series(combined_forecast, index=forecast_dates)

    except Exception as e:
        warn(f"Используем базовый прогноз HW. Причина: {str(e)}")
        return hw_model.forecast(forecast_periods)


def naive_forecast(ts: pd.Series, periods: int = 12) -> pd.Series:
    """
    Наивный прогноз (последнее известное значение)

    Параметры:
        ts: Исходный временной ряд
        periods: Количество прогнозируемых периодов

    Возвращает:
        pd.Series с прогнозом и DatetimeIndex
    """
    return pd.Series(
        [ts.iloc[-1]] * periods,
        index=pd.date_range(
            start=ts.index[-1] + pd.DateOffset(months=1),
            periods=periods,
            freq="MS",
        ),
    )


def hw_garch(ts: pd.Series, hw_model: ExponentialSmoothing):
    residuals = ts - hw_model.fittedvalues
    garch = arch_model(residuals.dropna(), vol="Garch", p=1, q=1).fit(
        disp="off"
    )

    garch_forecast = garch.forecast(horizon=12).mean.iloc[-1].values
    return hw_model.forecast(12) + garch_forecast


def hw_sarima_ks(ts: pd.Series, hw_model: ExponentialSmoothing):
    """
    Гибридная модель: SARIMA + Kernel Smoothing для остатков

    Параметры:
        ts: Исходный временной ряд
        hw_model: Обученная модель Хольта-Винтерса

    Возвращает:
        Комбинированный прогноз на 12 периодов
    """
    residuals = ts - hw_model.fittedvalues

    try:
        # SARIMA компонента
        sarima = SARIMAX(
            residuals.dropna(), order=(1, 0, 1), seasonal_order=(1, 0, 1, 12)
        ).fit(disp=False)
        sarima_fc = sarima.forecast(12)

        # Kernel Smoothing
        kr = KernelReg(
            residuals.dropna().values,
            np.arange(len(residuals.dropna())),
            var_type="c",
        )
        kr_fc = kr.fit(np.arange(len(ts), len(ts) + 12))[0]

        # Комбинированный прогноз (взвешенное среднее)
        combined = hw_model.forecast(12) + (0.7 * sarima_fc + 0.3 * kr_fc)
        return combined

    except Exception as e:
        print(f"Ошибка в SARIMA+KS: {str(e)}")
        return hw_model.forecast(12)
