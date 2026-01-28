"""
Feature importance analysis for the Late Fusion neural network model.
Analyzes ALL individual features without grouping and plots the top 10.
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm

from late_fusion import ClickbaitClassifier

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "CreateML", "datasets", "cnn_vllm_clip_max_mean_incong")
MODEL_PATH = os.path.join(SCRIPT_DIR, "cnn_vllm_clip_max_mean_incong.pth")

test_path = os.path.join(DATA_DIR, "test.parquet")
train_path = os.path.join(DATA_DIR, "train.parquet")


def load_model(model_path, input_dim, device):
    """Load trained model."""
    model = ClickbaitClassifier(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def get_accuracy(model, X, y, device):
    """Calculate model accuracy."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        preds = (outputs.cpu().numpy() >= 0.5).astype(int).flatten()
    return np.mean(preds == y)


def permutation_importance(model, X, y, feature_indices, device, n_repeats=5):
    """
    Calculate permutation importance for a set of features.
    Decreased n_repeats to 5 for individual feature analysis speed.
    """
    baseline_acc = get_accuracy(model, X, y, device)

    importance_scores = []
    for _ in range(n_repeats):
        X_permuted = X.copy()
        perm_idx = np.random.permutation(len(X))
        X_permuted[:, feature_indices] = X_permuted[perm_idx][:, feature_indices]

        permuted_acc = get_accuracy(model, X_permuted, y, device)
        importance_scores.append(baseline_acc - permuted_acc)

    return np.mean(importance_scores), np.std(importance_scores)


def calculate_feature_importance(model, X, y, columns, device, n_repeats=5):
    """
    Calculate importance for ALL individual features.
    """
    results = {}

    print(f"\nCalculating importance for {len(columns)} individual features...")
    for idx, feat in enumerate(tqdm(columns, desc="Features")):
        mean_imp, std_imp = permutation_importance(model, X, y, [idx], device, n_repeats)
        results[feat] = {'mean': mean_imp, 'std': std_imp}

    return results


def plot_feature_importance(results, top_n=25, save_path=None):
    """Create bar plot of top N feature importance."""
    # Sort by importance
    sorted_items = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)[:top_n]

    names = [item[0] for item in sorted_items]
    means = [item[1]['mean'] for item in sorted_items]
    stds = [item[1]['std'] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(names))
    ax.barh(y_pos, means, xerr=stds, color='#3498db', capsize=3, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # Highest importance at top
    ax.set_xlabel('Importance (Accuracy Drop when Permuted)')
    ax.set_title(f'Top {top_n} Individual Feature Importance')

    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + std + 0.0001, i, f'{mean:.4f}', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    df_test = pd.read_parquet(test_path)
    df_train = pd.read_parquet(train_path)

    label_col = "clickbait_label"

    # Prepare features
    feature_cols = [col for col in df_test.columns if col not in [label_col, 'video_id']]
    numeric_cols = df_test[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X_test = df_test[numeric_cols].values.astype(np.float32)
    y_test = df_test[label_col].values

    # Get input dim from training data
    train_feature_cols = [col for col in df_train.columns if col not in [label_col, 'video_id']]
    train_numeric_cols = df_train[train_feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    input_dim = len(train_numeric_cols)

    print(f"Test samples: {len(X_test)}")
    print(f"Total features: {len(numeric_cols)}")

    # Load model
    print(f"\nLoading model...")
    model = load_model(MODEL_PATH, input_dim, device)

    # Calculate baseline accuracy
    baseline_acc = get_accuracy(model, X_test, y_test, device)
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    # Calculate feature importance
    print("\n" + "="*60)
    print("CALCULATING INDIVIDUAL FEATURE IMPORTANCE")
    print("="*60)

    # Note: Reduced n_repeats to 3 for individual feature analysis speed if it takes too long
    # But 5 is a good balance for now.
    results = calculate_feature_importance(
        model, X_test, y_test, numeric_cols, device, n_repeats=5
    )

    # Print top results
    print("\n" + "="*60)
    print("TOP 10 FEATURE IMPORTANCE RESULTS")
    print("="*60)
    print(f"\n{'Feature':<45} {'Importance':>12} {'Std':>10}")
    print("-" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)[:25]
    for name, vals in sorted_results:
        print(f"{name:<45} {vals['mean']:>12.4f} {vals['std']:>10.4f}")

    # Plot
    plot_path = os.path.join(SCRIPT_DIR, "feature_importance_top10.png")
    plot_feature_importance(results, top_n=25, save_path=plot_path)

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
Importance score = accuracy drop when an individual feature dimension is randomly shuffled.
- With 2800+ features, individual dimensions may show very small importance scores.
- This view helps identify which specific embedding dimensions or interpretable features
  the model is most sensitive to.
""")
