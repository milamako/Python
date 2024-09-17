import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class VisualizationModule:
    def __init__(self, data):
        self.data = data
        self.color_palette = 'viridis'  # Цветовая схема по умолчанию

    def set_color_palette(self, palette):
        """Устанавливает цветовую схему"""
        self.color_palette = palette

    def bar_chart(self, column):
        """Столбчатая диаграмма для категориальных данных"""
        self.clear_plots()
        if self.data[column].dtype == 'object':
            plt.figure(figsize=(10, 6))  # Установка размера графика
            self.data[column].value_counts().plot(kind='bar', color=sns.color_palette(self.color_palette))
            plt.title(f'Столбчатая диаграмма для: {column}')
            plt.xlabel(column)
            plt.ylabel('Частота')
            plt.show()
        else:
            print(f"{column} не является категориальной переменной.")

    def pie_chart(self, column):
        """Круговая диаграмма для категориальных данных"""
        self.clear_plots()
        if self.data[column].dtype == 'object':
            plt.figure(figsize=(8, 8))  # Установка размера графика
            self.data[column].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette(self.color_palette))
            plt.title(f'Круговая диаграмма для: {column}')
            plt.ylabel('')  # Удаление метки оси Y
            plt.show()
        else:
            print(f"{column} не является категориальной переменной.")

    def donut_chart(self, column):
        """Секционная диаграмма для категориальных данных"""
        self.clear_plots()
        if self.data[column].dtype == 'object':
            plt.figure(figsize=(8, 8))  # Установка размера графика
            data = self.data[column].value_counts()
            plt.pie(data, labels=data.index, autopct='%1.1f%%', colors=sns.color_palette(self.color_palette), wedgeprops={'width': 0.3})
            plt.title(f'Секционная диаграмма для: {column}')
            plt.show()
        else:
            print(f"{column} не является категориальной переменной.")

    def histogram(self, column):
        """Гистограмма для числовых данных"""
        self.clear_plots()
        if pd.api.types.is_numeric_dtype(self.data[column]):
            plt.figure(figsize=(10, 6))  # Установка размера графика
            sns.histplot(self.data[column], kde=True, color=sns.color_palette(self.color_palette)[0])
            plt.title(f'Гистограмма для: {column}')
            plt.xlabel(column)
            plt.ylabel('Частота')
            plt.show()
        else:
            print(f"{column} не является числовой переменной.")
    
    def box_plot(self, column):
        """Ящик с усами для числовых данных"""
        self.clear_plots()
        if pd.api.types.is_numeric_dtype(self.data[column]):
            plt.figure(figsize=(8, 8))  # Установка размера графика: ширина, высота
            sns.boxplot(data=self.data, y=column, color=sns.color_palette(self.color_palette)[0])
            plt.title(f'Ящик с усами для: {column}')
            plt.ylabel(column)
            plt.show()
        else:
            print(f"{column} не является числовой переменной.")

    def scatter_plot(self, column_x, column_y):
        """Диаграмма рассеяния для числовых данных"""
        self.clear_plots()
        if pd.api.types.is_numeric_dtype(self.data[column_x]) and pd.api.types.is_numeric_dtype(self.data[column_y]):
            plt.figure(figsize=(10, 6))  # Установка размера графика
            sns.scatterplot(x=self.data[column_x], y=self.data[column_y], color=sns.color_palette(self.color_palette)[0])
            plt.title(f'Диаграмма рассеяния: {column_x} vs {column_y}')
            plt.xlabel(column_x)
            plt.ylabel(column_y)
            plt.show()
        else:
            print(f"{column_x} или {column_y} не являются числовыми переменными.")

    def line_chart(self, column_x, column_y):
        """Линейный график для числовых данных"""
        self.clear_plots()
        if pd.api.types.is_numeric_dtype(self.data[column_x]) and pd.api.types.is_numeric_dtype(self.data[column_y]):
            plt.figure(figsize=(10, 6))  # Установка размера графика
            sns.lineplot(x=self.data[column_x], y=self.data[column_y], color=sns.color_palette(self.color_palette)[0])
            plt.title(f'Линейный график: {column_x} vs {column_y}')
            plt.xlabel(column_x)
            plt.ylabel(column_y)
            plt.show()
        else:
            print(f"{column_x} или {column_y} не являются числовыми переменными.")
        
    def heatmap(self):
        """Тепловая карта для корреляции числовых данных"""
        self.clear_plots()
        numeric_data = self.data.select_dtypes(include='number')
        correlation_matrix = numeric_data.corr()
    
        plt.figure(figsize=(10, 10))  # Установка размера графика
        sns.heatmap(
            correlation_matrix, 
            annot=True,  # Показать значения в ячейках
            fmt=".1f",   # Форматирование значений с одним знаком после запятой
            cmap=self.color_palette,  # Цветовая палитра для тепловой карты
            vmin=-1, vmax=1,  # Диапазон значений для цветовой карты
            center=0  # Центр цветовой карты на 0
        )
        plt.title('Тепловая карта')
        plt.show()

    def clear_plots(self):
        """Очищает все текущие графики."""
        plt.close('all')
