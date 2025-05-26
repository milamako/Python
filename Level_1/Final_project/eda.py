# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data):
    
    # 1. Распределение цены
    plt.figure(figsize=(12, 6))
    sns.histplot(data['Price_USD'], bins=30, kde=True)
    plt.title('Распределение цен на продукты')
    plt.xlabel('Цена (USD)')
    plt.ylabel('Количество продуктов')
    plt.show()
    
    # 2. Корреляционная матрица
    plt.figure(figsize=(10, 5))
    corr = data.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Корреляционная матрица признаков')
    plt.show()
    
    # 3. Взаимосвязь цены и рейтинга
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Rating', y='Price_USD', data=data)
    plt.title('Распределение цены по рейтингу')
    plt.xlabel('Рейтинг')
    plt.ylabel('Цена (USD)')
    plt.show()
    
    # 4. Распределение рейтингов
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Rating', data=data)
    plt.title('Распределение рейтингов')
    plt.xlabel('Rating')
    plt.ylabel('Количество')
    plt.show()
    
    # 5. Взаимосвязь цены и размера продукта
    if 'Product_Size_ml' in data.columns:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='Product_Size_ml', y='Price_USD', data=data)
        plt.title('Взаимосвязь между Размером продукта и Ценой')
        plt.xlabel('Product_Size_ml')
        plt.ylabel('Price_USD')
        plt.show()
    
    # 6. Взаимосвязь цены и количества отзывов
    plt.figure(figsize=(12, 6))
    sns.kdeplot(x='Number_of_Reviews', y='Price_USD', data=data, cmap='Blues', shade=True)
    plt.title('Зависимость цены от количества отзывов (Тепловая карта плотности)')
    plt.xlabel('Количество отзывов')
    plt.ylabel('Цена (USD)')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Number_of_Reviews', y='Price_USD', data=data)
    plt.xscale('log')
    plt.title('Зависимость цены от количества отзывов (логарифмическая шкала)')
    plt.xlabel('Количество отзывов (логарифмическая шкала)')
    plt.ylabel('Цена (USD)')
    plt.show()


    '''plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Number_of_Reviews', y='Price_USD', data=data)
    plt.title('Взаимосвязь между Количеством отзывов и Ценой')
    plt.xlabel('Number_of_Reviews')
    plt.ylabel('Price_USD')
    plt.show()'''
    
    # 7. Влияние категории на цену
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Category', y='Price_USD', data=data)
    plt.title('Цена по категориям продуктов')
    plt.xlabel('Category')
    plt.ylabel('Price_USD')
    plt.xticks(rotation=45)
    plt.show()
