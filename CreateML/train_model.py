import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Configuration
TRAIN_PATH = 'datasets/train.csv'
VAL_PATH = 'datasets/validation.csv'
TEST_PATH = 'datasets/test.csv'
TARGET_COL = 'label'
ID_COL = 'video_id'  # We must drop this

def load_and_process(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    
    # Check if video_id exists and drop it, otherwise the model breaks
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
        
    # Separate Features (X) and Target (y)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    return X, y

def train_process():
    # 2. Load Data
    X_train, y_train = load_and_process(TRAIN_PATH)
    X_val, y_val = load_and_process(VAL_PATH)
    X_test, y_test = load_and_process(TEST_PATH)

    print(f"\nTraining on {X_train.shape[1]} features (Visual + Text + LLM embeddings)...")
    print("This might take a minute because the dataset is wide.")

    # 3. Initialize Model
    # n_jobs=-1 uses all CPU cores (faster for your Mac)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # 4. Train
    model.fit(X_train, y_train)
    print("Training Complete.")

    # 5. Validation
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"\n--- Validation Accuracy: {val_acc:.4f} ---")

    # 6. Final Test
    print("\n--- Final Test Report ---")
    test_pred = model.predict(X_test)
    print(classification_report(y_test, test_pred))

    # 7. Feature Importance (Bonus: See what matters most)
    # This helps you explain the results to your teammate
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 5 Most Important Features:")
    for f in range(5):
        print(f"{f+1}. {X_train.columns[indices[f]]} ({importances[indices[f]]:.4f})")

    # 8. Save
    joblib.dump(model, 'multimodal_clickbait_model.pkl')
    print("\nModel saved as 'multimodal_clickbait_model.pkl'")

if __name__ == "__main__":
    train_process()