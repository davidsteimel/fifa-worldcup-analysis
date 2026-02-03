# fifa-worldcup-analysis

# FIFA World Cup 2022 Prediction Model

This project uses **Machine Learning (Random Forest)** and **Monte Carlo Simulations** to predict the outcome of the FIFA World Cup 2022.

By analyzing team performance statistics from the group stage, the model learns which factors contribute most to winning and simulates the entire knockout tournament 10,000 times to determine the most probable World Cup winner.

## Key Features

* **Data Pipeline:** Automated cleaning and processing of match statistics (goals, possession, xG, defensive pressures, etc.).
* **Feature Engineering:** Creates "Team Profiles" based on group stage performance and calculates differential features for every match matchup (e.g., *Team A Possession vs. Team B Possession*).
* **Data Augmentation:** Overcomes the small dataset size (48 group games) using:
***Symmetry Flipping:** Every match `A vs B` is mirrored as `B vs A`.
* **Noise Injection:** Synthetic data generation by adding statistical noise to robustly train the model against overfitting.

* **Monte Carlo Simulation:** Simulates the entire tournament bracket (Round of 16 → Final) 10,000 times to calculate the probability of each team reaching specific stages.

## Installation & Usage

1. **Clone the repository:**
```bash
git clone https://github.com/davidsteimel/fifa-worldcup-analysis.git
cd fifa-worldcup-analysis

```
2. **Install dependencies:**
```bash
pip install -r requirements.txt

```
3. **Run the analysis:**
```bash
python main.py

```

## Key Findings

### Model Performance

After data augmentation, the Random Forest model achieved an accuracy of **~90%** on the test set.

### Feature Importance (The "Winning Formula")

The model identified the following statistics as the strongest predictors for victory:

1. **Average Points Difference:** Current form in the group stage is the best predictor.
2. **Goals Conceded Difference:** Defensive stability is weighted higher than offensive output.
3. **Crosses Completed:** A proxy for control in the final third.
4. **Corners:** Indicates game dominance and pressure.

### Simulation Results

Based on pure group stage statistics, the model favors teams with high efficiency and strong defenses (e.g., England, Netherlands) over teams that had "messy" group stages (like Argentina losing to Saudi Arabia), highlighting the contrast between statistical analysis and real-world tournament dynamics.

## License

This project is open-source and available for educational purposes.