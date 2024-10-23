# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_real_vs_pred(y_test, y_pred, model_name):
    """
    Строит график реальных и предсказанных значений.
    """
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Фактические значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'Фактические vs Предсказанные значения: {model_name}')
    plt.show()

def plot_prediction_errors(y_test, y_pred, model_name):
    """
    Строит график ошибок прогнозирования.
    """
    errors = y_test - y_pred
    plt.figure(figsize=(12, 6))
    sns.histplot(errors, bins=30, kde=True)
    plt.title(f'Распределение ошибок прогнозирования: {model_name}')
    plt.xlabel('Ошибка')
    plt.ylabel('Частота')
    plt.show()

def plot_feature_importance(model, feature_names, model_name):
    """
    Строит график важности признаков, если модель поддерживает это.
    """
    if hasattr(model, 'coef_') and model.coef_ is not None:
        importance = model.coef_
        indices = np.argsort(np.abs(importance))[::-1]
        top_features = np.array(feature_names)[indices]
        top_importance = importance[indices]
        
        plt.figure(figsize=(12,6))
        sns.barplot(x=top_importance, y=top_features)
        plt.title(f'Важность признаков: {model_name}')
        plt.xlabel('Значение коэффициента')
        plt.ylabel('Признак')
        plt.show()
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        top_features = np.array(feature_names)[indices]
        top_importance = importance[indices]
        
        plt.figure(figsize=(12,6))
        sns.barplot(x=top_importance, y=top_features)
        plt.title(f'Важность признаков: {model_name}')
        plt.xlabel('Важность')
        plt.ylabel('Признак')
        plt.show()
    else:
        print(f"Модель {model_name} не поддерживает вывод важности признаков.")

def plot_metrics(metrics_df):
    """
    Строит графики метрик качества моделей.
    """
    # График MAE
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='MAE', data=metrics_df)
    plt.title('MAE моделей')
    plt.xlabel('Модель')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    plt.show()
        
    # График R2 Score
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='R2 Score', data=metrics_df)
    plt.title('R2 Score моделей')
    plt.xlabel('Модель')
    plt.ylabel('R2 Score')
    plt.xticks(rotation=45)
    plt.show()

def plot_anomalies(residuals, anomaly_indices, model_name):
    """
    Строит график остатков с выделением аномалий.

    Параметры:
        residuals (np.array): Остатки предсказаний модели.
        anomaly_indices (np.array): Индексы обнаруженных аномалий.
        model_name (str): Название модели.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(residuals, label='Остатки', color='blue')
    plt.scatter(anomaly_indices, residuals[anomaly_indices], color='red', label='Аномалии', marker='x')
    plt.xlabel('Индекс')
    plt.ylabel('Остаток')
    plt.title(f'Обнаруженные аномалии в остатках: {model_name}')
    plt.legend()
    plt.show()

