import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt



def train_model(train_df: pd.DataFrame):
    X = train_df.drop(columns=['target'])
    y = train_df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print("\nDetaillierter Bericht:\n", classification_report(y_test, y_pred))

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return model, importances

def simulate_match_outcome(model, feature_row: pd.Series):
    # Reshape feature_row for prediction into matrix
    input_data = feature_row.values.reshape(1, -1)
    
    probs = model.predict_proba(input_data)[0]
    
    outcome = np.random.choice(model.classes_, p=probs)
    
    return outcome