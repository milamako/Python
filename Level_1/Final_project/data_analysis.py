import pandas as pd

def count_missing_values(df):
    """
    Подсчитывает количество и процент пропущенных значений для каждого столбца.
    Возвращает словарь с результатами.
    """
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_data = {
        column: {
            'Missing Values': missing_values[column],
            'Percentage (%)': missing_percent[column]
        }
        for column in df.columns if missing_values[column] > 0
    }
    # Сортируем по проценту пропущенных значений
    sorted_missing_data = dict(sorted(missing_data.items(), key=lambda item: item[1]['Percentage (%)'], reverse=True))
    return sorted_missing_data

def get_data_types(df):
    """
    Возвращает информацию о типах данных в каждом столбце в виде словаря.
    """
    return df.dtypes.to_dict()

def describe_dataset(df):
    """
    Описывает датасет: количество строк и столбцов, количество уникальных значений, пропущенные значения.
    """
    num_rows, num_columns = df.shape
    missing_values = count_missing_values(df)
    
    description = {
        "Number of Rows": num_rows,
        "Number of Columns": num_columns,
        "Missing Data Summary": missing_values,
        "Data Types": get_data_types(df),
        "Memory Usage (MB)": df.memory_usage(deep=True).sum() / (1024 ** 2),
        "Unique Values Count": df.nunique().to_dict()
    }
    
    return description

def report_on_dataset(df):
    """
    Генерирует подробный отчёт о датасете, включая типы данных, пропущенные значения и основные статистики.
    """
    print("Количество строк и столбцов:")
    print(f"{df.shape[0]} строк, {df.shape[1]} столбцов")
    
    print("\nПропущенные значения:")
    missing_data = count_missing_values(df)
    if not missing_data:
        print("Пропущенные значения отсутствуют.")
    else:
        for column, data in missing_data.items():
            print(f"{column}: {data['Missing Values']} пропущенных значений ({data['Percentage (%)']:.2f}%)")
    
    print("\nТипы данных в столбцах:")
    data_types = get_data_types(df)
    for column, dtype in data_types.items():
        print(f"{column}: {dtype}")
    
    print("\nОсновная статистика для числовых столбцов:")
    print(df.describe().to_string())
    
    print("\nОсновная статистика для категориальных столбцов:")
    categorical_columns = df.select_dtypes(include=['object']).columns
    if not categorical_columns.empty:
        print(df[categorical_columns].describe().to_string())
    else:
        print("Категориальных столбцов нет.")
    
    print("\nОбщая информация о датасете:")
    print(f"Используемая память: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")






