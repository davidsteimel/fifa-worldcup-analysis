import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_goal_distribution(df: pd.DataFrame):
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='total_goals', bins=10, kde=True)
    plt.title('Distribution of Total Goals per Match')
    plt.xlabel('Number of Goals')
    plt.ylabel('Number of Matches')
    
def plot_outcome_distribution(df: pd.DataFrame):
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='outcome', order=['Heimsieg', 'Auswärtssieg', 'Unentschieden'])
    plt.title('Match Outcomes Distribution')
    plt.xlabel('Outcome')
    plt.ylabel('Number of Matches')
    #plt.show()