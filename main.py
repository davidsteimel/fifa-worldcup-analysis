from src.data_loader.loader import load_match_data

wm_data = load_match_data()

row_count, column_count = wm_data.shape

nan_count = wm_data.isna().sum()

print(f"Die Daten enthalten {row_count} Zeilen und {column_count} Spalten.")
print(f"Die Daten enthalten {nan_count} fehlende Werte.")
