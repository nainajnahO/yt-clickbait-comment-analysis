import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

# Import from late_fusion
from late_fusion import ClickbaitClassifier, load_data, prepare_datasets, create_dataloaders, evaluate


# Get the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "CreateML", "datasets", "cnn_vllm")

MODEL_PATH = os.path.join(SCRIPT_DIR, "cnn_vllm.pth")
val_path = os.path.join(DATA_DIR, "valid.parquet")
train_path = os.path.join(DATA_DIR, "train.parquet")


def load_model(model_path, input_dim, device):
    """Load a trained model from disk."""
    model = ClickbaitClassifier(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def run_inference(model, data_loader, device):
    """Run inference on data and return predictions and probabilities."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            
            outputs = model(X_batch)
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(y_batch.numpy().flatten())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


if __name__ == "__main__":
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load validation data
    print(f"\nLoading data from: {DATA_DIR}")
    df_train, df_val = load_data(train_path, val_path)
    label_col = "clickbait_label"
    
    (X_train, y_train), (X_val, y_val) = prepare_datasets(df_train, df_val, label_col)
    
    # Create validation dataloader
    _, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
    
    # Get input dimension from training data
    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")
    print(f"Validation samples: {len(X_val)}")
    
    # Load the trained model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH, input_dim, device)
    print("Model loaded successfully!")
    
    # Run inference
    print("\n" + "="*50)
    print("Running Inference on Validation Set")
    print("="*50)
    
    predictions, probabilities, labels = run_inference(model, val_loader, device)
    
    # Calculate and display metrics
    criterion = nn.BCELoss()
    metrics = evaluate(model, val_loader, criterion, device)
    
    print(f"\nResults:")
    print(f"  Loss:      {metrics['loss']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    # Print sample predictions
    print(f"\n" + "="*50)
    print("Sample Predictions (first 10)")
    print("="*50)
    print(f"{'Index':<8} {'True Label':<12} {'Predicted':<12} {'Probability':<12}")
    print("-" * 44)
    for i in range(min(10, len(predictions))):
        print(f"{i:<8} {int(labels[i]):<12} {int(predictions[i]):<12} {probabilities[i]:.4f}")
    
    # Class distribution in predictions
    print(f"\n" + "="*50)
    print("Prediction Distribution")
    print("="*50)
    print(f"Predicted Clickbait:     {np.sum(predictions == 1)} ({100*np.mean(predictions == 1):.1f}%)")
    print(f"Predicted Non-Clickbait: {np.sum(predictions == 0)} ({100*np.mean(predictions == 0):.1f}%)")