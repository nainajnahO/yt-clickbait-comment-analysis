"""
Improved CNN for Thumbnail Classification
Includes: Batch Normalization, Data Augmentation, Deeper Architecture
"""

import os
import json
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get the project root directory (parent of D folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ThumbnailTruthData")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "D")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

# Image Training augmentation
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.05
    ),
    transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
])

# Validation/test - no augmentation, just normalize
val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# ============================================================================
# DATASET CLASS
# ============================================================================

class ThumbnailDataset(Dataset):
    """Custom PyTorch Dataset for thumbnail images with augmentation"""

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to image files
            labels: List of labels (0 or 1)
            transform: Torchvision transforms to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and preprocess image

        Returns:
            image_tensor: Tensor of shape (3, 224, 224), normalized to [0, 1]
            label: Label as float tensor
        """
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # Apply transforms (resize, augment, convert to tensor)
        if self.transform:
            img_tensor = self.transform(img)
        else:
            # Fallback if no transform provided
            img = img.resize(IMG_SIZE)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return img_tensor, label

# ============================================================================
# DATA LOADING & SPLITTING
# ============================================================================

def load_and_split_data():
    """
    Load images from MTV and NMTV directories and split into train/val/test sets

    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders
        dataset_info: Dictionary with dataset statistics
    """
    print("Loading data from directories...")

    # Load MTV images (misleading, label=1)
    mtv_dir = os.path.join(DATA_DIR, "MTV_Thumbnails")
    mtv_paths = glob.glob(os.path.join(mtv_dir, "*.jpg"))
    mtv_labels = [1] * len(mtv_paths)

    # Load NMTV images (non-misleading, label=0)
    nmtv_dir = os.path.join(DATA_DIR, "NMTV_Thumbnails")
    nmtv_paths = glob.glob(os.path.join(nmtv_dir, "*.jpg"))
    nmtv_labels = [0] * len(nmtv_paths)

    # Combine all paths and labels
    all_paths = mtv_paths + nmtv_paths
    all_labels = mtv_labels + nmtv_labels

    print(f"Total images: {len(all_paths)}")
    print(f"  MTV (misleading): {len(mtv_paths)}")
    print(f"  NMTV (non-misleading): {len(nmtv_paths)}")

    # First split: 70% train, 30% temp
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=all_labels
    )

    # Second split: 30% temp -> 15% val, 15% test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_labels
    )

    print(f"\nData split:")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Validation: {len(val_paths)} images")
    print(f"  Test: {len(test_paths)} images")

    # Create datasets with appropriate transforms
    train_dataset = ThumbnailDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ThumbnailDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = ThumbnailDataset(test_paths, test_labels, transform=val_transform)

    print("\nData augmentation:")
    print("  Train: RandomFlip, RandomRotation, ColorJitter")
    print("  Val/Test: No augmentation")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    dataset_info = {
        'total_images': len(all_paths),
        'mtv_count': len(mtv_paths),
        'nmtv_count': len(nmtv_paths),
        'train_count': len(train_paths),
        'val_count': len(val_paths),
        'test_count': len(test_paths)
    }

    return train_loader, val_loader, test_loader, dataset_info

# ============================================================================
# IMPROVED MODEL CLASS
# ============================================================================

class ImprovedCNN(nn.Module):
    """
    Improved CNN with:
    - 4 convolutional layers (deeper network)
    - Batch normalization (better training)
    - Progressive filter increase (32→64→128→256)
    """

    def __init__(self):
        super(ImprovedCNN, self).__init__()

        # Block 1: 224x224 -> 112x112
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 112x112 -> 56x56
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 56x56 -> 28x28
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4: 28x28 -> 14x14
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Calculate flattened size: 224 -> 112 -> 56 -> 28 -> 14
        self.flatten_size = 256 * 14 * 14

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output tensor of shape (batch_size, 1) - logits
        """
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Block 2: Conv -> BN -> ReLU -> Pool
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Block 3: Conv -> BN -> ReLU -> Pool
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Block 4: Conv -> BN -> ReLU -> Pool
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Flatten
        x = x.view(-1, self.flatten_size)

        # Fully connected layers with batch norm and dropout
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience=5):
    """
    Train the CNN model with learning rate scheduling and early stopping

    Args:
        model: ImprovedCNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        patience: Early stopping patience (default: 5 epochs)

    Returns:
        history: Dictionary with training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print(f"\nTraining on device: {DEVICE}")
    print(f"Early stopping patience: {patience} epochs")
    print(f"Learning rate scheduling: ReduceLROnPlateau")
    print("=" * 60)

    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # ============================
        # Training phase
        # ============================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move to device
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track statistics
            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        # Average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # ============================
        # Validation phase
        # ============================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        # Average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)

        # Learning rate scheduling - reduce LR when validation stops improving
        scheduler.step(val_accuracy)
        current_lr = optimizer.param_groups[0]['lr']

        # Track best validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        print(f"  LR: {current_lr:.6f} | No Improve: {epochs_no_improve}/{patience}")
        print("-" * 60)

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")
            break

    print(f"\nTraining finished!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")
    return history

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_loader):
    """
    Evaluate model on test set

    Args:
        model: Trained ImprovedCNN model
        test_loader: Test data loader

    Returns:
        metrics: Dictionary with evaluation metrics
    """
    print("\nEvaluating on test set...")

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()

            all_predictions.extend(predictions.flatten())
            all_labels.extend(labels.numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    precision = precision_score(all_labels, all_predictions) * 100
    recall = recall_score(all_labels, all_predictions) * 100
    f1 = f1_score(all_labels, all_predictions) * 100

    metrics = {
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1
    }

    print("=" * 60)
    print("Test Set Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall: {recall:.2f}%")
    print(f"  F1-Score: {f1:.2f}%")
    print("=" * 60)

    return metrics

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(model, history, metrics, dataset_info):
    """
    Save model, training history, metrics, and visualizations

    Args:
        model: Trained ImprovedCNN model
        history: Training history dictionary
        metrics: Test metrics dictionary
        dataset_info: Dataset statistics
    """
    print("\nSaving results...")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Save model
    model_path = os.path.join(OUTPUT_DIR, "improved_cnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved to: {model_path}")

    # 2. Save training history
    history_path = os.path.join(OUTPUT_DIR, "improved_training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History saved to: {history_path}")

    # 3. Generate metrics report
    report_path = os.path.join(OUTPUT_DIR, "improved_metrics_report.txt")
    with open(report_path, 'w') as f:
        f.write("Improved CNN Thumbnail Classifier\n")
        f.write("=" * 60 + "\n\n")

        f.write("Improvements:\n")
        f.write("  - 4 convolutional layers (vs 2 in basic model)\n")
        f.write("  - Batch normalization after each layer\n")
        f.write("  - Data augmentation (flip, rotation, color jitter)\n")
        f.write("  - Progressive filters: 32→64→128→256\n\n")

        f.write("Dataset:\n")
        f.write(f"  Misleading (MTV): {dataset_info['mtv_count']} images\n")
        f.write(f"  Non-misleading (NMTV): {dataset_info['nmtv_count']} images\n")
        f.write(f"  Total: {dataset_info['total_images']} images\n\n")

        f.write("Train/Val/Test Split:\n")
        f.write(f"  Train: {dataset_info['train_count']} images (70%)\n")
        f.write(f"  Validation: {dataset_info['val_count']} images (15%)\n")
        f.write(f"  Test: {dataset_info['test_count']} images (15%)\n\n")

        f.write("Training Configuration:\n")
        f.write(f"  Epochs: {EPOCHS}\n")
        f.write(f"  Batch Size: {BATCH_SIZE}\n")
        f.write(f"  Learning Rate: {LEARNING_RATE}\n")
        f.write(f"  Device: {DEVICE}\n\n")

        f.write("Final Training Metrics:\n")
        f.write(f"  Training Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Training Accuracy: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"  Validation Loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Validation Accuracy: {history['val_acc'][-1]:.2f}%\n\n")

        f.write("Test Set Metrics:\n")
        f.write(f"  Accuracy: {metrics['test_accuracy']:.2f}%\n")
        f.write(f"  Precision: {metrics['test_precision']:.2f}%\n")
        f.write(f"  Recall: {metrics['test_recall']:.2f}%\n")
        f.write(f"  F1-Score: {metrics['test_f1']:.2f}%\n")

    print(f"  Report saved to: {report_path}")

    # 4. Create training curves plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs_range = range(1, len(history['train_loss']) + 1)

    # Loss plot
    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs_range, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs_range, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "improved_training_curves.png")
    plt.savefig(plot_path, dpi=150)
    print(f"  Plot saved to: {plot_path}")

    print("\nAll results saved successfully!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""

    print("=" * 60)
    print("Improved CNN Thumbnail Classifier")
    print("=" * 60)
    print("\nImprovements over basic model:")
    print("  ✓ 4 convolutional layers (deeper)")
    print("  ✓ Batch normalization (faster, more stable)")
    print("  ✓ Gentler data augmentation (better balance)")
    print("  ✓ Learning rate scheduling (adaptive LR)")
    print("  ✓ Early stopping (prevents overfitting)")
    print("  ✓ Progressive filters: 32→64→128→256")
    print("=" * 60)

    # 1. Load and split data
    train_loader, val_loader, test_loader, dataset_info = load_and_split_data()

    # 2. Build model
    print("\nBuilding model...")
    model = ImprovedCNN().to(DEVICE)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # 3. Define loss, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler: reduce LR when validation accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize validation accuracy
        factor=0.5,  # Reduce LR by half
        patience=3  # Wait 3 epochs before reducing
    )

    # 4. Train model with early stopping
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=EPOCHS, patience=5
    )

    # 5. Evaluate on test set
    metrics = evaluate_model(model, test_loader)

    # 6. Save all results
    save_results(model, history, metrics, dataset_info)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()