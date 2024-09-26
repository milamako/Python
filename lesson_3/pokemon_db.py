import psycopg2
import pandas as pd
from psycopg2.extras import execute_values
from requests.exceptions import RequestException
from config import DB_CONFIG

# Подключение к базе данных
def connect_to_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Успешное подключение к базе данных")
        return conn
    except psycopg2.Error as e:
        print(f"Ошибка подключения к базе данных: {e}")
        raise

# Функция для создания таблиц
def create_tables(conn):
    try:
        cur = conn.cursor()
        # Создание таблицы pokemons
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pokemons (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                japanese_name VARCHAR(100),
                pokedex_number INT UNIQUE NOT NULL,
                height_m FLOAT,
                weight_kg FLOAT,
                hp INT,
                attack INT,
                defense INT,
                sp_attack INT,
                sp_defense INT,
                speed INT,
                base_total INT,
                base_happiness INT,
                base_egg_steps INT,
                capture_rate FLOAT,
                percentage_male FLOAT,
                experience_growth INT,
                generation INT,
                is_legendary BOOLEAN
            );
        """)
        print("Таблица pokemons создана")

        # Создание таблицы types
        cur.execute("""
            CREATE TABLE IF NOT EXISTS types (
                id SERIAL PRIMARY KEY,
                type_name VARCHAR(50) UNIQUE NOT NULL
            );
        """)
        print("Таблица types создана")

        # Создание таблицы abilities
        cur.execute("""
            CREATE TABLE IF NOT EXISTS abilities (
                id SERIAL PRIMARY KEY,
                ability_name VARCHAR(100) UNIQUE NOT NULL
            );
        """)
        print("Таблица abilities создана")

        # Создание таблицы pokemon_types
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pokemon_types (
                pokemon_id INT REFERENCES pokemons(id),
                type_id INT REFERENCES types(id),
                PRIMARY KEY (pokemon_id, type_id)
            );
        """)
        print("Таблица pokemon_types создана")

        # Создание таблицы pokemon_abilities
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pokemon_abilities (
                pokemon_id INT REFERENCES pokemons(id),
                ability_id INT REFERENCES abilities(id),
                PRIMARY KEY (pokemon_id, ability_id)
            );
        """)
        print("Таблица pokemon_abilities создана")

        conn.commit()
        cur.close()
    except psycopg2.Error as e:
        print(f"Ошибка при создании таблиц: {e}")
        conn.rollback()
        raise

# Функция для импорта данных
def import_data(conn, df):
    try:
        cur = conn.cursor()

        # Преобразуйте столбец is_legendary в boolean
        df['is_legendary'] = df['is_legendary'].astype(bool)
        print("Столбец is_legendary преобразован в boolean")

        # Импорт покемонов
        pokemon_values = [
            (row['name'], row['japanese_name'], row['pokedex_number'], row['height_m'], row['weight_kg'],
             row['hp'], row['attack'], row['defense'], row['sp_attack'], row['sp_defense'], row['speed'],
             row['base_total'], row['base_happiness'], row['base_egg_steps'], row['capture_rate'],
             row['percentage_male'], row['experience_growth'], row['generation'], row['is_legendary'])
            for _, row in df.iterrows()
        ]
        execute_values(cur, """
            INSERT INTO pokemons (name, japanese_name, pokedex_number, height_m, weight_kg, 
                                  hp, attack, defense, sp_attack, sp_defense, speed, 
                                  base_total, base_happiness, base_egg_steps, capture_rate, 
                                  percentage_male, experience_growth, generation, is_legendary)
            VALUES %s
            ON CONFLICT (pokedex_number) DO NOTHING;
        """, pokemon_values)
        print("Данные покемонов импортированы")

        # Импорт типов
        types = pd.DataFrame({'type_name': pd.concat([df['type1'], df['type2']]).dropna().unique()})
        type_values = [(row['type_name'],) for _, row in types.iterrows()]
        execute_values(cur, """
            INSERT INTO types (type_name) VALUES %s
            ON CONFLICT (type_name) DO NOTHING;
        """, type_values)
        print("Типы покемонов импортированы")

        # Импорт способностей
        abilities = df['abilities'].apply(lambda x: x.split(', '))
        ability_values = set(ability for abilities_list in abilities for ability in abilities_list)
        ability_values = [(ability,) for ability in ability_values]
        execute_values(cur, """
            INSERT INTO abilities (ability_name) VALUES %s
            ON CONFLICT (ability_name) DO NOTHING;
        """, ability_values)
        print("Способности покемонов импортированы")

        # Импорт связей между покемонами и типами
        for _, row in df.iterrows():
            if pd.notna(row['type1']):
                cur.execute("""
                    INSERT INTO pokemon_types (pokemon_id, type_id)
                    VALUES (
                        (SELECT id FROM pokemons WHERE pokedex_number = %s),
                        (SELECT id FROM types WHERE type_name = %s)
                    ) ON CONFLICT DO NOTHING;
                """, (row['pokedex_number'], row['type1']))

            if pd.notna(row['type2']):
                cur.execute("""
                    INSERT INTO pokemon_types (pokemon_id, type_id)
                    VALUES (
                        (SELECT id FROM pokemons WHERE pokedex_number = %s),
                        (SELECT id FROM types WHERE type_name = %s)
                    ) ON CONFLICT DO NOTHING;
                """, (row['pokedex_number'], row['type2']))
        print("Связи между покемонами и типами импортированы")

        # Импорт связей между покемонами и способностями
        for _, row in df.iterrows():
            abilities_list = row['abilities'].split(', ')
            for ability in abilities_list:
                cur.execute("""
                    INSERT INTO pokemon_abilities (pokemon_id, ability_id)
                    VALUES (
                        (SELECT id FROM pokemons WHERE pokedex_number = %s),
                        (SELECT id FROM abilities WHERE ability_name = %s)
                    ) ON CONFLICT DO NOTHING;
                """, (row['pokedex_number'], ability))
        print("Связи между покемонами и способностями импортированы")

        conn.commit()
        cur.close()
    except psycopg2.Error as e:
        print(f"Ошибка при импорте данных: {e}")
        conn.rollback()
        raise
    except RequestException as e:
        print(f"Ошибка запроса: {e}")
        raise