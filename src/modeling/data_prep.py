import pandas as pd
import numpy as np
from src.processing import engineer_features

def add_noise(features: pd.Series, noise_level: float = 0.05) -> pd.Series:
    noisy_features = features.copy()
    
    for col in noisy_features.index:
        if col == 'target':
            continue  
        val = noisy_features[col]
        
        sigma = abs(val) * noise_level if val != 0 else 0.01
        noise = np.random.normal(0, sigma)
        
        noisy_features[col] = val + noise
        
    return noisy_features

def create_training_data(matches_df: pd.DataFrame, profiles_df: pd.DataFrame, use_noise: bool = True) -> pd.DataFrame:
    data = []
    for index, row in matches_df.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        goals_team1 = row['number of goals team1']
        goals_team2 = row['number of goals team2']
        goal_dif = goals_team1 - goals_team2
        if goal_dif > 0:
            winner = 1
        elif goal_dif < 0:
            winner = -1
        else:
            winner = 0
        features= engineer_features.get_profile_difference(profiles_df= profiles_df, team1_name= team1, team2_name= team2)
        features['target'] = winner
        data.append(features)

        features_flipped = features.copy()
        # Flip the feature differences by multiplying by -1
        # so that team2 becomes team1, 
        # for example possession difference of 5 becomes -5 for the other team 
        cols_to_flip = [c for c in features_flipped.index if c != 'target']
        features_flipped[cols_to_flip] = features_flipped[cols_to_flip] * -1
        
        # Flip the target outcome
        features_flipped['target'] = winner * -1
        data.append(features_flipped)

        if use_noise:
            noisy_base = add_noise(features, noise_level=0.1) 
            data.append(noisy_base)

            noisy_flipped = add_noise(features_flipped, noise_level=0.1)
            data.append(noisy_flipped)

    data_df = pd.DataFrame(data)
    return data_df

