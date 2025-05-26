import pandas as pd

class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def count_missing_values(self):
        # Подсчитывает количество пропущенных значений в каждом столбце
        return self.data.isnull().sum()

    def report_missing_values(self):
        # Выводит отчет о пропущенных значениях
        missing_values = self.count_missing_values()
        report = missing_values[missing_values > 0]
        if report.empty:
            print('Нет пропущенных значений.')
        else:
            print('Отчет о пропущенных значениях:\n', report)
            
    def get_categorical_columns(self):
        # Возвращает список категориальных столбцов
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        return categorical_cols.tolist()

    def get_numerical_columns(self):
        # Возвращает список числовых столбцов
        numerical_cols = self.data.select_dtypes(include=['number']).columns
        return numerical_cols.tolist()

    def report_column_types(self):
        # Выводит список категориальных и числовых столбцов
        categorical_cols = self.get_categorical_columns()
        numerical_cols = self.get_numerical_columns()
        
        print('Категориальные столбцы:')
        print(categorical_cols)
        
        print('Числовые столбцы:')
        print(numerical_cols)

    def fill_missing_values(self, strategy='mean'):
        # Заполнение числовых данных
        numerical_cols = self.get_numerical_columns()
        for col in numerical_cols:
            if strategy == 'mean':
                self.data[col] = self.data[col].fillna(self.data[col].mean())
            elif strategy == 'median':
                self.data[col] = self.data[col].fillna(self.data[col].median())
            elif strategy == 'mode':
                self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        
        # Заполнение категориальных данных
        categorical_cols = self.get_categorical_columns()
        for col in categorical_cols:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        
        print('Пропущенные значения были успешно заполнены с использованием стратегии:', strategy)
