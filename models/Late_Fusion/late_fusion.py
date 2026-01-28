import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import os


class ClickbaitClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)
    

def load_data(train_path, val_path):
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    return df_train, df_val


def prepare_datasets(df_train, df_val, label_col):
    # Get feature columns (exclude label and any non-numeric columns)
    feature_cols = [col for col in df_train.columns if col != label_col]
    
    # Select only numeric columns
    numeric_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < len(feature_cols):
        dropped_cols = set(feature_cols) - set(numeric_cols)
        print(f"Dropping non-numeric columns: {dropped_cols}")
    
    X_train = df_train[numeric_cols].values.astype(np.float32)
    y_train = df_train[label_col].values.astype(np.float32)

    X_val = df_val[numeric_cols].values.astype(np.float32)
    y_val = df_val[label_col].values.astype(np.float32)

    return (X_train, y_train), (X_val, y_val)


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    """Create PyTorch DataLoaders for training and validation."""
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, pbar=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader.dataset)


def evaluate(model, val_loader, criterion, device):
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(y_batch.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(val_loader.dataset)
    
    # Calculate metrics
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc_roc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    }
    
    return metrics


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu', early_stopping_patience=5):
    """Full training loop with early stopping."""
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        # Training with batch progress bar
        batch_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", 
                          leave=False, unit="batch")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, pbar=batch_pbar)
        batch_pbar.close()
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_metrics["loss"]:.4f}',
            'acc': f'{val_metrics["accuracy"]:.4f}',
            'f1': f'{val_metrics["f1"]:.4f}',
            'prec': f'{val_metrics["precision"]:.4f}',
        })
        
        # Early stopping check
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                tqdm.write(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model





if __name__ == "__main__":

    # Get the project root directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    DATA_DIR = os.path.join(PROJECT_ROOT, "CreateML", "datasets", "cnn_vllm_clip_max_mean_incong")

    train_path = os.path.join(DATA_DIR, "train.parquet")
    val_path = os.path.join(DATA_DIR, "valid.parquet")

    df_train, df_val = load_data(train_path, val_path)
    label_col = "clickbait_label"

    (X_train, y_train), (X_val, y_val) = prepare_datasets(df_train, df_val, label_col)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    input_dim = X_train.shape[1]
    model = ClickbaitClassifier(input_dim)
    print(f"Model initialized with input dimension: {input_dim}")

    # Train model
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50 + "\n")
    
    model = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=50, 
        lr=0.001, 
        device=device,
        early_stopping_patience=5
    )

    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation on Validation Set")
    print("="*50)
    
    criterion = nn.BCELoss()
    final_metrics = evaluate(model, val_loader, criterion, device)
    
    print(f"Loss:      {final_metrics['loss']:.4f}")
    print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall:    {final_metrics['recall']:.4f}")
    print(f"F1 Score:  {final_metrics['f1']:.4f}")
    print(f"AUC-ROC:   {final_metrics['auc_roc']:.4f}")

    # Save model
    model_save_path = os.path.join(SCRIPT_DIR, "cnn_vllm_clip_max_mean_incong.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to: {model_save_path}")
