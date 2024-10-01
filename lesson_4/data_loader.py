import pandas as pd

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        # Устанавливаем опцию для отображения всех столбцов
        pd.set_option('display.max_columns', None)
        
    def load_data(self): 
        # Определяет тип файла и загружает данные
        if self.file_path.endswith('.csv'):
            return self._load_csv()
        elif self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
            return self._load_excel()
        else:
            raise ValueError("Неподдерживаемый формат файла. Поддерживаются только CSV и Excel форматы.")

    def _load_csv(self):
        # Загружает данные из CSV файла
        try:
            data = pd.read_csv(self.file_path)
            return data
        except Exception as e:
            raise IOError(f"Ошибка при загрузке CSV файла: {e}")

    def _load_excel(self):
        # Загружает данные из Excel файла
        try:
            data = pd.read_excel(self.file_path)
            return data
        except Exception as e:
            raise IOError(f'Ошибка при загрузке Excel файла: {e}')

