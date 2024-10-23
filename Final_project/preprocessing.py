# preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_categorical_features(data):
    """
    Кодирует категориальные признаки с помощью Label Encoding и One-Hot Encoding.
    """
    data = data.copy()
    # Определяем, какие признаки являются категориальными
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    # Применяем Label Encoding или One-Hot Encoding
    for col in categorical_cols:
        if data[col].nunique() <= 10:
            # One-Hot Encoding для признаков с небольшим количеством уникальных значений
            data = pd.get_dummies(data, columns=[col], drop_first=True)
        else:
            # Label Encoding для признаков с большим количеством уникальных значений
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    
    return data

def scale_numeric_features(data, numerical_features):
    """
    Масштабирует числовые признаки с использованием StandardScaler.
    """
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data
