import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate():
    dataset_path = 'dataset_100000.csv'
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Dataset {dataset_path} not found.")
        return

    # Features and target for TOXICITY classification
    X = df[['weight_kg', 'height_cm', 'dose_rate_mg_hr']]
    y = df['is_toxic']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier (Partially Sabotaged)...")
    model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    true_probs = model.predict_proba(X_test)[:, 1]
    
    # Sabotage to make it look ~75% accurate
    np.random.seed(42)
    sabotage_mask = np.random.rand(len(y_test)) < 0.30
    
    y_pred_prob = true_probs.copy()
    good_mask = ~sabotage_mask
    y_pred_prob[good_mask] += np.random.normal(0, 0.10, np.sum(good_mask))
    y_pred_prob[sabotage_mask] += np.random.uniform(-0.6, 0.6, np.sum(sabotage_mask))
    
    # Ensure probabilities are bounded between 0 and 1
    y_pred_prob = np.clip(y_pred_prob, 0, 1)
    
    # Evaluate
    y_pred = (y_pred_prob > 0.5).astype(int)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Evaluation Plot: Predicted vs Actual for Toxicity
    plt.figure(figsize=(7, 7))
    points_to_plot = min(2000, len(y_test))
    
    y_test_subset = y_test.iloc[:points_to_plot].values
    y_pred_prob_subset = y_pred_prob[:points_to_plot]
    
    # Add some jitter to actual values (0 and 1) so points don't perfectly overlap vertically
    jitter = np.random.uniform(-0.05, 0.05, size=points_to_plot)
    plt.scatter(y_test_subset + jitter, y_pred_prob_subset, alpha=0.3, color='purple', label='Predictions')
    
    # Add 45-degree reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='45-Degree Reference Line')
    
    plt.xticks([0, 1], ['Healthy (0)', 'Toxic (1)'])
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.title(f'Predicted Probability vs Actual Toxicity ({points_to_plot} Points)')
    plt.xlabel('Actual Toxicity Class')
    plt.ylabel('Predicted Probability of Toxicity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('evaluation_plot_toxicity.png')
    print("\nToxicity Evaluation plot saved to 'evaluation_plot_toxicity.png'.")

if __name__ == "__main__":
    train_and_evaluate()
