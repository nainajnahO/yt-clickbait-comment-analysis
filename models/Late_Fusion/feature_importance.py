"""
Feature importance analysis for the Late Fusion neural network model.
Uses permutation importance to measure how much each feature/feature group
contributes to model performance.
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

# Embedding prefixes
EMBEDDING_PREFIXES = ['cnn_', 'vllm_', 'sent_emb_']


def get_embedding_cols(columns, prefix):
    """Get all columns matching an embedding prefix."""
    pattern = re.compile(f'^{prefix}\\d+$')
    return sorted([col for col in columns if pattern.match(col)],
                  key=lambda x: int(x.split('_')[-1]))


def get_interpretable_features(columns):
    """Get list of interpretable (non-embedding) feature columns."""
    exclude = ['video_id', 'clickbait_label']
    all_embedding_cols = set()
    for prefix in EMBEDDING_PREFIXES:
        all_embedding_cols.update(get_embedding_cols(columns, prefix))
    return [col for col in columns
            if col not in exclude and col not in all_embedding_cols]


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


def permutation_importance(model, X, y, feature_indices, device, n_repeats=10):
    """
    Calculate permutation importance for a set of features.

    Args:
        model: Trained model
        X: Feature matrix
        y: Labels
        feature_indices: List of column indices to permute together
        device: torch device
        n_repeats: Number of permutation repeats

    Returns:
        Mean importance score (accuracy drop)
    """
    baseline_acc = get_accuracy(model, X, y, device)

    importance_scores = []
    for _ in range(n_repeats):
        X_permuted = X.copy()
        # Shuffle the specified columns together
        perm_idx = np.random.permutation(len(X))
        X_permuted[:, feature_indices] = X_permuted[perm_idx][:, feature_indices]

        permuted_acc = get_accuracy(model, X_permuted, y, device)
        importance_scores.append(baseline_acc - permuted_acc)

    return np.mean(importance_scores), np.std(importance_scores)


def calculate_feature_importance(model, X, y, columns, device, n_repeats=10):
    """
    Calculate importance for interpretable features and embedding groups.
    """
    col_to_idx = {col: idx for idx, col in enumerate(columns)}

    results = {}

    # Get interpretable features
    interpretable = get_interpretable_features(columns)

    # Calculate importance for each interpretable feature
    print("\nCalculating importance for interpretable features...")
    for feat in tqdm(interpretable, desc="Interpretable features"):
        if feat in col_to_idx:
            idx = [col_to_idx[feat]]
            mean_imp, std_imp = permutation_importance(model, X, y, idx, device, n_repeats)
            results[feat] = {'mean': mean_imp, 'std': std_imp, 'type': 'interpretable'}

    # Calculate importance for embedding groups
    print("\nCalculating importance for embedding groups...")
    for prefix in tqdm(EMBEDDING_PREFIXES, desc="Embedding groups"):
        emb_cols = get_embedding_cols(columns, prefix)
        if emb_cols:
            indices = [col_to_idx[col] for col in emb_cols if col in col_to_idx]
            if indices:
                mean_imp, std_imp = permutation_importance(model, X, y, indices, device, n_repeats)
                name = f"{prefix.rstrip('_').upper()} embedding ({len(indices)} dims)"
                results[name] = {'mean': mean_imp, 'std': std_imp, 'type': 'embedding'}

    return results


def plot_feature_importance(results, save_path=None):
    """Create bar plot of feature importance."""
    # Sort by importance
    sorted_items = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)

    names = [item[0] for item in sorted_items]
    means = [item[1]['mean'] for item in sorted_items]
    stds = [item[1]['std'] for item in sorted_items]
    types = [item[1]['type'] for item in sorted_items]

    # Colors based on type
    colors = ['#2ecc71' if t == 'interpretable' else '#3498db' for t in types]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, means, xerr=stds, color=colors, capsize=3, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # Highest importance at top
    ax.set_xlabel('Importance (Accuracy Drop when Permuted)')
    ax.set_title('Feature Importance - Permutation Method')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Interpretable Features'),
        Patch(facecolor='#3498db', label='Embedding Groups')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + std + 0.001, i, f'{mean:.4f}', va='center', fontsize=9)

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
    print(f"Features: {len(numeric_cols)}")

    # Load model
    print(f"\nLoading model...")
    model = load_model(MODEL_PATH, input_dim, device)

    # Calculate baseline accuracy
    baseline_acc = get_accuracy(model, X_test, y_test, device)
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    # Calculate feature importance
    print("\n" + "="*60)
    print("CALCULATING FEATURE IMPORTANCE")
    print("="*60)

    results = calculate_feature_importance(
        model, X_test, y_test, numeric_cols, device, n_repeats=10
    )

    # Print results
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE RESULTS")
    print("="*60)
    print(f"\n{'Feature':<45} {'Importance':>12} {'Std':>10}")
    print("-" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    for name, vals in sorted_results:
        print(f"{name:<45} {vals['mean']:>12.4f} {vals['std']:>10.4f}")

    # Plot
    plot_path = os.path.join(SCRIPT_DIR, "feature_importance.png")
    plot_feature_importance(results, save_path=plot_path)

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
Importance score = accuracy drop when feature is randomly shuffled.
- Higher values = more important (model relies on this feature)
- Near zero = feature contributes little to predictions
- Negative = shuffling improves accuracy (possible overfitting)

Note: Embedding groups are shuffled together, so their importance
reflects the combined contribution of all dimensions.
""")
