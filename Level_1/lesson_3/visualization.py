from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DB_CONFIG

def fetch_data_from_db():
    try:
        connection_string = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        engine = create_engine(connection_string)
        
        query = """
        SELECT p.*, t.type_name AS primary_type, secondary.type_name AS secondary_type, a.ability_name
        FROM pokemons p
        LEFT JOIN pokemon_types pt ON p.id = pt.pokemon_id
        LEFT JOIN types t ON pt.type_id = t.id
        LEFT JOIN pokemon_types st ON p.id = st.pokemon_id AND st.type_id != pt.type_id
        LEFT JOIN types secondary ON st.type_id = secondary.id
        LEFT JOIN pokemon_abilities pa ON p.id = pa.pokemon_id
        LEFT JOIN abilities a ON pa.ability_id = a.id;
        """
        
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"Ошибка при извлечении данных: {e}")
        return None

def new_pokemons_per_generation(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df[df['is_legendary'] == False], x='generation', discrete=True, hue='generation', palette='viridis', multiple="stack")
    plt.title('Новые покемоны в каждом поколении')
    plt.xlabel('Поколение')
    plt.ylabel('Количество')
    plt.show()
    
def pokemons_by_type_combination(df):
    # Создаем диаграмму рассеяния
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='primary_type',
        y='secondary_type',
        size='base_total',  # Размер точки может зависеть от общей статистики покемона
        hue='is_legendary',  # Цвет точки может зависеть от того, является ли покемон легендарным
        palette=['blue', 'orange'],
        sizes=(20, 200),
        alpha=0.6
    )
    plt.title('Распределение покемонов по комбинациям типов')
    plt.xlabel('Основной тип')
    plt.ylabel('Вторичный тип')
    plt.xticks(rotation=90)  # Поворот меток на оси X для лучшей читаемости
    plt.legend(title='Легендарный', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def common_legendary_types(df):
    legendary_df = df[df['is_legendary'] == True]
    plt.figure(figsize=(10, 6))
    sns.countplot(y='primary_type', data=legendary_df, order=legendary_df['primary_type'].value_counts().index, hue='primary_type', palette='plasma')
    plt.title('Наиболее распространенные типы легендарных покемонов')
    plt.xlabel('Количество')
    plt.ylabel('Тип')
    plt.show()

def primary_type_change_by_generation(df):
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='generation', y='primary_type', data=df, hue='primary_type', palette='Set2', inner="stick", dodge=False)

    plt.title('Изменение основного типа по поколениям')
    plt.xlabel('Поколение')
    plt.ylabel('Основной тип')
    plt.show()
    
def easiest_generation_to_catch(df):
    # Добавить фиктивную категориальную переменную для использования в hue
    df['dummy_gen'] = df['generation']  # Используем 'generation' как фиктивную переменную

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='generation', y='capture_rate', data=df, hue='dummy_gen', palette='Blues', dodge=False, legend=False)

    plt.title('Уровень поимки по поколениям')
    plt.xlabel('Поколение')
    plt.ylabel('Уровень поимки')
    plt.show()

def easiest_type_to_catch(df):
    # Добавить фиктивную категориальную переменную для использования в hue
    df['dummy_type'] = df['primary_type']  # Используем 'primary_type' как фиктивную переменную

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='primary_type', y='capture_rate', data=df, hue='dummy_type', palette='Greens', dodge=False, legend=False)

    plt.title('Уровень поимки по основным типам')
    plt.xlabel('Основной тип')
    plt.ylabel('Уровень поимки')
    plt.xticks(rotation=90)  # Поворот подписей на оси X для лучшей читаемости
    plt.show()


def average_abilities_per_pokemon(df):
    ability_counts = df.groupby('name')['ability_name'].nunique()
    plt.figure(figsize=(10, 6))
    sns.histplot(ability_counts, bins=range(1, ability_counts.max() + 2), kde=True, color='purple')
    plt.title('Распределение количества способностей у покемонов')
    plt.xlabel('Количество способностей')
    plt.ylabel('Количество')
    plt.show()

def tallest_and_heaviest_pokemon(df):
    # Построить диаграмму рассеяния для всех покемонов
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='weight_kg',
        y='height_m',
        hue='is_legendary',
        palette=['darkgreen', 'lightgreen'],
        alpha=0.7
    )

    # Настроить внешний вид графика
    plt.title('Покемоны по весу и росту')
    plt.xlabel('Вес (кг)')
    plt.ylabel('Рост (м)')
    sns.despine(top=True, right=True)

    # Настройка легенды
    plt.legend(title='Легендарный', borderpad=0, markerscale=0.5, handlelength=0, loc='upper left')

    # Найти и аннотировать топ-5 покемонов по росту и весу
    top5_weight_height_merged = pd.concat([
        df.nlargest(5, 'height_m'),
        df.nlargest(5, 'weight_kg')
    ]).drop_duplicates(subset=['name'])

    for index, row in top5_weight_height_merged.iterrows():
        plt.annotate(row['name'], xy=(row['weight_kg'] + 10, row['height_m']), fontsize=9, color='green')

    plt.show()

def best_generation(df):
    # Вычислить средние характеристики для каждого поколения
    gen_avg_stats = df.groupby('generation')[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].mean().sum(axis=1)
    best_gen = gen_avg_stats.idxmax()
    print(f"Лучшее поколение с точки зрения средних характеристик: Поколение {best_gen}")

    # Построить ящик с усами для распределения характеристик по поколениям
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='generation', y=df[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].mean(axis=1))

    # Настроить внешний вид графика
    plt.title('Распределение средних характеристик покемонов по поколениям')
    plt.xlabel('Поколение')
    plt.ylabel('Средние характеристики')
    plt.show()

def attribute_correlations(df):
    plt.figure(figsize=(12, 10))
    corr = df[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Корреляция между атрибутами покемонов')
    plt.show()

def best_pokemon(df):
    # Удаление дубликатов по имени покемона
    df_unique = df.drop_duplicates(subset='name')
    
    # Найти 10 лучших покемонов по суммарным характеристикам
    top_10_pokemons = df_unique.nlargest(10, 'base_total')

    print("10 лучших покемонов по суммарным характеристикам:")
    for index, row in top_10_pokemons.iterrows():
        print(f"{row['name']} с суммарными характеристиками {row['base_total']}")

    # Добавить фиктивную категориальную переменную для использования в hue
    top_10_pokemons['rank'] = range(len(top_10_pokemons))

    # Построить столбчатую диаграмму для 10 лучших покемонов с разными цветами
    plt.figure(figsize=(12, 6))
    sns.barplot(x='name', y='base_total', data=top_10_pokemons, hue='rank', dodge=False, palette='viridis', legend=False)

    # Настроить внешний вид графика
    plt.title('Топ-10 покемонов по суммарным характеристикам')
    plt.xlabel('Имя покемона')
    plt.ylabel('Суммарные характеристики')
    plt.xticks(rotation=45)  # Поворот меток на оси X для лучшей читаемости
    plt.tight_layout()
    plt.show()