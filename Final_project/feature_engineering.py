# feature_engineering.py

import pandas as pd

def feature_engineering(data):
    """
    Выполняет фиче инжиниринг для создания новых признаков.
    """
    data = data.copy()
    # Извлекаем числовое значение из 'Product_Size'
    data['Product_Size_ml'] = data['Product_Size'].str.extract(r'(\d+)').astype(float)
    
    # Создаем признак 'Price_per_ml'
    data['Price_per_ml'] = data['Price_USD'] / data['Product_Size_ml']
    
    # Создаем бинарный признак 'High_Rating' (1, если Rating >= 4, иначе 0)
    data['High_Rating'] = (data['Rating'] >= 4).astype(int)
    
    # Кодируем признак 'Gender_Target'
    gender_map = {'Male': 0, 'Female': 1, 'Unisex': 2}
    data['Gender_Target'] = data['Gender_Target'].map(gender_map)
    
    return data
