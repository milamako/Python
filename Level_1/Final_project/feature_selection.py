# feature_selection.py

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

def select_features(X, y, k=10):
    """
    Параметры:
        X (pd.DataFrame): Матрица признаков.
        y (pd.Series): Целевая переменная.
        k (int): Количество выбираемых признаков.
    Возвращает:
        np.array: Матрица признаков с отобранными признаками.
        list: Названия отобранных признаков.
    """
    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    
    # маска отобранных признаков
    mask = selector.get_support()
    selected_features = X.columns[mask]
    
    return X_new, selected_features
