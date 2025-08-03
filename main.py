from src.data_loader.loader import load_match_data
from src.processing.engineer_features import create_features
from src.visualization.plots import plot_goal_distribution, plot_outcome_distribution

wm_data = load_match_data()

row_count, column_count = wm_data.shape

nan_count = wm_data.isna().sum()

print(f"Die Daten enthalten {row_count} Zeilen und {column_count} Spalten.")

processed_df = create_features(wm_data)

# 3. Die neue Plot-Funktion aufrufen und ihr den bearbeiteten DataFrame übergeben
plot_goal_distribution(processed_df)
plot_outcome_distribution(processed_df)