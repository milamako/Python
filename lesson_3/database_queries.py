import psycopg2
from config import DB_CONFIG

def get_legendary_pokemons():
    query = """
    SELECT name, pokedex_number
    FROM pokemons
    WHERE is_legendary = TRUE;
    """
    return execute_query(query)

def get_pokemons_by_type(pokemon_type):
    query = """
    SELECT p.name, p.pokedex_number
    FROM pokemons p
    JOIN pokemon_types pt ON p.id = pt.pokemon_id
    JOIN types t ON pt.type_id = t.id
    WHERE t.type_name = %s;
    """
    return execute_query(query, (pokemon_type,))


def get_pokemons_by_ability(ability):
    query = """
    SELECT p.name, p.pokedex_number
    FROM pokemons p
    JOIN pokemon_abilities pa ON p.id = pa.pokemon_id
    JOIN abilities a ON pa.ability_id = a.id
    WHERE a.ability_name = %s;
    """
    return execute_query(query, (ability,))

def get_pokemons_with_high_attack(min_attack):
    query = """
    SELECT name, pokedex_number, attack
    FROM pokemons
    WHERE attack > %s;
    """
    return execute_query(query, (min_attack,))

def get_non_legendary_first_gen_pokemons():
    query = """
    SELECT name, pokedex_number
    FROM pokemons
    WHERE is_legendary = FALSE AND generation = 1;
    """
    return execute_query(query)

def get_happiest_pokemon():
    query = """
    SELECT name, pokedex_number, base_happiness
    FROM pokemons
    ORDER BY base_happiness DESC
    LIMIT 1;
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