# anomaly_detection.py

import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies(residuals, contamination=0.01):
    """
    Параметры:
        residuals (np.array): Остатки предсказаний модели.
        contamination (float): Пропорция выбросов в данных.
    Возвращает:
        np.array: Индексы обнаруженных аномалий.
    """
    residuals = residuals.reshape(-1, 1)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomalies = iso_forest.fit_predict(residuals)
    anomaly_indices = np.where(anomalies == -1)[0]
    return anomaly_indices
