import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Функция для анализа распределения данных
def analyze_distribution(df):
    """
    Анализ распределения данных в DataFrame.
    
    :param df: DataFrame с данными для анализа.
    """
    # Определение числовых столбцов
    numerical_columns = [
        'Administrative', 'Administrative_Duration', 'Informational', 
        'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'OperatingSystems', 
        'Browser', 'Region', 'TrafficType'
    ]
    
    # Построение гистограмм для каждого числового признака
    plt.figure(figsize=(16, 12))
    for i, column in enumerate(numerical_columns, 1):
        plt.subplot(4, 4, i)
        sns.histplot(df[column], kde=True, color='c')
        plt.title(f'{column} distribution')
    plt.tight_layout()
    plt.show()

    # Вычисление асимметрии (skewness) и эксцесса (kurtosis)
    skewness_kurtosis = {
        "Feature": [],
        "Skewness": [],
        "Kurtosis": []
    }

    for column in numerical_columns:
        skewness_kurtosis["Feature"].append(column)
        skewness_kurtosis["Skewness"].append(skew(df[column]))
        skewness_kurtosis["Kurtosis"].append(kurtosis(df[column]))

    # Преобразование в DataFrame для удобства
    skew_kurt_df = pd.DataFrame(skewness_kurtosis)
    print(skew_kurt_df)

