import pandas as pd

def load_match_data():
    """Die Daten werden geladen."""
    
    file_path = 'data/Fifa_world_cup_matches.csv'
    
    try:
        df = pd.read_csv(file_path, )
        print("Daten erfolgreich geladen!")
        return df
    except FileNotFoundError:
        print(f"Fehler: Die Datei unter '{file_path}' wurde nicht gefunden.")
        return None