import pandas as pd
import numpy as np
from src.processing import engineer_features
from collections import defaultdict

def simulate_match(model, profiles_df, team1, team2):
    features = engineer_features.get_profile_difference(profiles_df, team1, team2)
    features_df = features.to_frame().T
    
    probs = model.predict_proba(features_df)[0]
    outcome = np.random.choice(model.classes_, p=probs)
    
    if outcome == 1:
        return team1
    elif outcome == -1:
        return team2
    else:
        return np.random.choice([team1, team2])

def run_single_tournament(model, profiles_df, starting_bracket):
    current_round_matches = starting_bracket
    reached_rounds = defaultdict(str) 

    # Initial teams have reached Round of 16
    for t1, t2 in starting_bracket:
        reached_rounds[t1] = "Round of 16"
        reached_rounds[t2] = "Round of 16"

    rounds = ["Quarter-final", "Semi-final", "Final", "Winner"]
    
    for round_name in rounds:
        next_round_teams = []
        
        for t1, t2 in current_round_matches:
            winner = simulate_match(model, profiles_df, t1, t2)
            next_round_teams.append(winner)
            reached_rounds[winner] = round_name

        if len(next_round_teams) == 1:
            break
        
        # Prepare matchups for the next round
        new_matchups = []
        for i in range(0, len(next_round_teams), 2):
            if i+1 < len(next_round_teams):
                new_matchups.append((next_round_teams[i], next_round_teams[i+1]))
        
        current_round_matches = new_matchups
        
    return reached_rounds

def simulate_tournament_monte_carlo(model, matches_df, profiles_df, n_simulations=10000):
    #Get starting bracket from Round of 16 matches
    r16_matches = matches_df[matches_df['category'] == 'Round of 16']
    starting_bracket = []
    for index, row in r16_matches.iterrows():
        starting_bracket.append((row['team1'], row['team2']))
        
    print(f"Starting {n_simulations} simulations of the tournament")
    
    counters = defaultdict(lambda: defaultdict(int))
    teams = set()

    for t1, t2 in starting_bracket:
        teams.add(t1)
        teams.add(t2)
    
    for _ in range(n_simulations):
        results = run_single_tournament(model, profiles_df, starting_bracket)
        
        for team, round_reached in results.items():
            counters[team][round_reached] += 1
    
            if round_reached == "Winner":
                counters[team]["Final"] += 1
                counters[team]["Semi-final"] += 1
                counters[team]["Quarter-final"] += 1
            elif round_reached == "Final":
                counters[team]["Semi-final"] += 1
                counters[team]["Quarter-final"] += 1
            elif round_reached == "Semi-final":
                counters[team]["Quarter-final"] += 1

    stats_data = []
    for team in teams:
        row = {'Team': team}
        row['Quarter-Final %'] = (counters[team]['Quarter-final'] / n_simulations) * 100
        row['Semi-Final %'] = (counters[team]['Semi-final'] / n_simulations) * 100
        row['Final %'] = (counters[team]['Final'] / n_simulations) * 100
        row['Winner %'] = (counters[team]['Winner'] / n_simulations) * 100
        stats_data.append(row)
        
    df_stats = pd.DataFrame(stats_data)
    
    return df_stats.sort_values(by='Winner %', ascending=False).reset_index(drop=True)