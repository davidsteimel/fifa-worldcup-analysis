from src.data_loader.loader import load_match_data
from src.visualization.plots import plot_goal_distribution, plot_outcome_distribution
from src.processing.engineer_features import create_team_profiles, create_correlation_matrix

wm_data = load_match_data()

team_profiles = create_team_profiles(wm_data)
corr_matrix = create_correlation_matrix(team_profiles)

print(corr_matrix['points'].sort_values(ascending=False))