import pandas as pd
import numpy as np

def add_match_results(df: pd.DataFrame) -> pd.DataFrame:
    
    conditions = [
        (df['number of goals team1'] > df['number of goals team2']),
        (df['number of goals team1'] == df['number of goals team2']),
        (df['number of goals team1'] < df['number of goals team2']) 
    ]
    
    choices = [3, 1, 0]
    
    df['points'] = np.select(conditions, choices, default=0)
    return df

import pandas as pd
import numpy as np

def create_team_profiles(df: pd.DataFrame) -> pd.DataFrame:
    df = add_match_results(df).copy()
    
    stats_basename = [
        'number of goals', 'possession', 'total attempts', 'conceded', 
        'on target attempts', 'attempts inside the penalty area',
        'receptions between midfield and defensive lines', 'attempted line breaks',
        'completed line breaks', 'attempted defensive line breaks',
        'completed defensive line breaks', 'passes', 'passes completed',
        'crosses', 'crosses completed', 'corners', 'free kicks',
        'goal preventions', 'forced turnovers', 'defensive pressures applied'
    ]

    team1_df = df.copy()
    team1_df['Team'] = team1_df['team1']
    team1_df['points_final'] = team1_df['points']
    for stat in stats_basename:
        t1_col = [c for c in df.columns if stat in c and 'team1' in c][0]
        team1_df[stat] = team1_df[t1_col]

    team2_df = df.copy()
    team2_df['Team'] = team2_df['team2']
    team2_df['points_final'] = team2_df['points'].replace({3: 0, 0: 3, 1: 1})
    for stat in stats_basename:
        t2_col = [c for c in df.columns if stat in c and 'team2' in c][0]
        team2_df[stat] = team2_df[t2_col]

    keep_cols = ['Team', 'category', 'points_final'] + stats_basename
    long_df = pd.concat([team1_df[keep_cols], team2_df[keep_cols]], ignore_index=True)

    if 'possession' in long_df.columns:
        long_df['possession'] = long_df['possession'].astype(str).str.replace('%', '').astype(float)

    group_stage_df = long_df.sort_values(by=['Team', 'category']).groupby('Team').head(3)

    team_profiles = group_stage_df.groupby('Team').mean(numeric_only=True).reset_index()
    
    team_profiles = team_profiles.drop(columns=['category', 'match_no'], errors='ignore')
    team_profiles = team_profiles.rename(columns={'points_final': 'points'})
    
    return team_profiles

def create_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = df.corr(numeric_only=True)
    return matrix

def get_top_corr(df: pd.DataFrame, target_col: str, threshold: float = 0.3) -> pd.Series:
    corr_matrix = create_correlation_matrix(df)
    top_corr = corr_matrix[target_col].sort_values(ascending=False).head(10).iloc[1:]
    return top_corr