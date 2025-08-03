import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df['total_goals'] = df['number of goals team1'] + df['number of goals team2']
    df['outcome'] = df.apply(
        lambda x: "Heimsieg" if x['number of goals team1'] > x['number of goals team2']
        else ("Auswärtssieg" if x['number of goals team1'] < x['number of goals team2'] else "Unentschieden"),
        axis=1
    )
    print("Feature Engineering abgeschlossen.")
    return df