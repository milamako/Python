# database_aggregates.py

import psycopg2
from config import DB_CONFIG

def get_total_attack():
    query = """
    SELECT SUM(attack) AS total_attack
    FROM pokemons;
    """
    return execute_query(query)

def get_average_defense():
    query = """
    SELECT AVG(defense) AS average_defense
    FROM pokemons;
    """
    return execute_query(query)

def get_pokemon_count():
    query = """
    SELECT COUNT(*) AS total_pokemons
    FROM pokemons;
    """
    return execute_query(query)

def get_new_pokemons_per_generation():
    query = """
    SELECT generation, COUNT(*) AS num_pokemons
    FROM pokemons
    GROUP BY generation
    ORDER BY generation;
    """
    return execute_query(query)

def get_heaviest_and_tallest_pokemons():
    query = """
    SELECT name, pokedex_number, weight_kg, height_m
    FROM pokemons
    WHERE (weight_kg = (SELECT MAX(weight_kg) FROM pokemons WHERE weight_kg IS NOT NULL))
       OR (height_m = (SELECT MAX(height_m) FROM pokemons WHERE height_m IS NOT NULL));
    """
    return execute_query(query)


def get_most_common_types():
    query = """
    SELECT t.type_name, COUNT(*) AS type_count
    FROM pokemon_types pt
    JOIN types t ON pt.type_id = t.id
    GROUP BY t.type_name
    ORDER BY type_count DESC;
    """
    return execute_query(query)

def execute_query(query, params=None):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(query, params)
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results
    except psycopg2.Error as e:
        print(f"Ошибка выполнения запроса: {e}")
        return None
