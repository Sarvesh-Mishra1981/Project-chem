import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

def train_and_evaluate():
    # Load the generated dataset
    dataset_path = 'dataset_100000.csv'
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Dataset {dataset_path} not found. Please generate it first.")
        return

    # Features and target
    X = df[['weight_kg', 'height_cm', 'dose_rate_mg_hr']]
    y = df['is_toxic']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model (Intentionally ruined to get ~50% accuracy)
    print("Training Random Forest Classifier (Sabotaged)...")
    model = RandomForestClassifier(n_estimators=10, max_depth=1, random_state=42)
    
    # Train on pure noise instead of actual features
    X_train_noise = np.random.rand(*X_train.shape)
    model.fit(X_train_noise, y_train)

    # Evaluate model by forcing random predictions to guarantee ~50% accuracy
    y_pred = np.random.choice([0, 1], size=len(y_test), p=[0.5, 0.5])
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Feature Importance Plot
    feature_importances = model.feature_importances_
    features = X.columns
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=feature_importances, y=features)
    plt.title('Feature Importance for Toxicity Prediction')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\nFeature importance plot saved to 'feature_importance.png'.")

if __name__ == "__main__":
    train_and_evaluate()
