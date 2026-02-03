import pandas as pd
import numpy as np

STATS_COLUMNS = [
    'number of goals', 'possession', 'total attempts', 'conceded', 
    'on target attempts', 'attempts inside the penalty area',
    'receptions between midfield and defensive lines', 'attempted line breaks',
    'completed line breaks', 'attempted defensive line breaks',
    'completed defensive line breaks', 'passes', 'passes completed',
    'crosses', 'crosses completed', 'corners', 'free kicks',
    'goal preventions', 'forced turnovers', 'defensive pressures applied'
]

# Remove duplicate matches (team1 vs team2 and team2 vs team1)
def remove_duplicate_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['match_id'] = df.apply(lambda row: '_'.join(sorted([row['team1'], row['team2']])), axis=1)
    df_unique = df.drop_duplicates(subset=['match_id'])
    return df_unique.drop(columns=['match_id'])

# Calculate match results and assign points for each team
def add_match_results(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        (df['number of goals team1'] > df['number of goals team2']),
        (df['number of goals team1'] == df['number of goals team2']),
        (df['number of goals team1'] < df['number of goals team2']) 
    ]
    choices = [3, 1, 0]
    df['points_team1'] = np.select(conditions, choices, default=0)
    df['points_team2'] = df['points_team1'].replace({3: 0, 0: 3, 1: 1})
    return df

# Prepare data in long format for team-wise analysis
def prepare_long_format(df: pd.DataFrame) -> pd.DataFrame:
    df = add_match_results(df).copy()
    
    t1 = df.copy()
    t1['Team'] = t1['team1']
    t1['points_final'] = t1['points_team1']
    
    t2 = df.copy()
    t2['Team'] = t2['team2']
    t2['points_final'] = t2['points_team2']
    
    for stat in STATS_COLUMNS:
        t1[stat] = t1[f"{stat} team1"]
        t2[stat] = t2[f"{stat} team2"]

    keep_cols = ['Team', 'category', 'points_final'] + STATS_COLUMNS
    long_df = pd.concat([t1[keep_cols], t2[keep_cols]], ignore_index=True)

    if 'possession' in long_df.columns:
        long_df['possession'] = long_df['possession'].astype(str).str.replace('%', '').astype(float)
    
    return long_df

def get_group_games(df_teams: pd.DataFrame) -> pd.DataFrame:
    group_stage_df = df_teams[df_teams['category'].str.contains('Group', na=False)]
    return group_stage_df.sort_values(by=['Team']).groupby('Team').head(3)

# Create team profiles by averaging stats over group stage matches
def create_team_profiles(df: pd.DataFrame) -> pd.DataFrame:
    long_df = prepare_long_format(df)
    
    group_stage_df = get_group_games(long_df)

    profiles = group_stage_df.groupby('Team').mean(numeric_only=True).reset_index()
    profiles = profiles.rename(columns={'points_final': 'avg_points'})
    
    return profiles

# Create correlation matrix for the team profiles
def create_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(numeric_only=True)

# Get top correlated features with respect to a target column
def get_top_corr(df: pd.DataFrame, target_col: str) -> pd.Series:
    corr_matrix = create_correlation_matrix(df)
    top_corr = corr_matrix[target_col].sort_values(ascending=False).head(10).iloc[1:]
    return top_corr

# Calculate the difference in profiles between two teams
def get_profile_difference(profiles_df: pd.DataFrame, team1_name: str, team2_name: str) -> pd.Series:
    profiles_indexed = profiles_df.set_index('Team')
    
    profil_team1 = profiles_indexed.loc[team1_name]
    profil_team2 = profiles_indexed.loc[team2_name]
    
    diff = profil_team1 - profil_team2
    
    diff.index = [f"{col}_diff" for col in diff.index]
    
    return diff

