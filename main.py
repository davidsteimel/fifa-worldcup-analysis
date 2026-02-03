from src.data_loader.loader import load_match_data
from src.visualization.plots import plot_goal_distribution, plot_outcome_distribution
from src.processing.engineer_features import create_team_profiles, create_correlation_matrix, remove_duplicate_matches
from src.modeling.data_prep import create_training_data
from src.modeling.train import train_model, simulate_match_outcome
from src.modeling.predict import run_single_tournament, simulate_tournament_monte_carlo

df_raw = load_match_data()

df_raw.columns = df_raw.columns.str.replace("team1", " team1", regex=False)
df_raw.columns = df_raw.columns.str.replace("team2", " team2", regex=False)
# replace multiple spaces with single space
df_raw.columns = df_raw.columns.str.replace(r"\s+", " ", regex=True)
df_raw.columns = df_raw.columns.str.strip()

df_clean = remove_duplicate_matches(df_raw)
team_profiles = create_team_profiles(df_clean)

df_train_matches = df_clean[df_clean['category'].str.contains('Group')].copy()
df_ko_matches = df_clean[~df_clean['category'].str.contains('Group')].copy()

df_train = create_training_data(matches_df= df_train_matches, profiles_df= team_profiles)


model, importances = train_model(df_train)

print("\nTop 5 influencing features for winning:")
print(importances.head(5))

tournament_results = simulate_tournament_monte_carlo(
    model, 
    df_ko_matches, 
    team_profiles, 
    n_simulations=1000
)

print("\n Tournament Simulation Results:")
print(tournament_results.to_string(formatters={
    'Quarter-Final %': '{:.1f}%'.format,
    'Semi-Final %': '{:.1f}%'.format,
    'Final %': '{:.1f}%'.format,
    'Winner %': '{:.1f}%'.format
}))