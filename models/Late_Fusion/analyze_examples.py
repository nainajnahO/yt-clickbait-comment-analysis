"""
Analyze model predictions with concrete examples.
Shows true positives, true negatives, false positives, and false negatives.
Displays interpretable features compared to class means.
For embedding vectors, shows angular distance to class mean vectors.
"""
import torch
import pandas as pd
import numpy as np
import os
import re

from late_fusion import ClickbaitClassifier

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "CreateML", "datasets", "cnn_vllm_clip_max_mean_incong")
DESCRIPTIONS_PATH = os.path.join(PROJECT_ROOT, "CreateML", "features", "vllm_descriptions.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "cnn_vllm_clip_max_mean_incong.pth")

test_path = os.path.join(DATA_DIR, "test.parquet")
train_path = os.path.join(DATA_DIR, "train.parquet")

# Embedding prefixes to group
EMBEDDING_PREFIXES = ['cnn_', 'vllm_', 'sent_emb_']


def get_embedding_cols(df, prefix):
    """Get all columns matching an embedding prefix."""
    pattern = re.compile(f'^{prefix}\\d+$')
    return sorted([col for col in df.columns if pattern.match(col)],
                  key=lambda x: int(x.split('_')[-1]))


def get_interpretable_features(df):
    """Get list of interpretable (non-embedding) feature columns."""
    exclude = ['video_id', 'clickbait_label']
    all_embedding_cols = set()
    for prefix in EMBEDDING_PREFIXES:
        all_embedding_cols.update(get_embedding_cols(df, prefix))
    return [col for col in df.columns
            if col not in exclude and col not in all_embedding_cols]


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def angle_degrees(v1, v2):
    """Calculate angle in degrees between two vectors."""
    cos_sim = cosine_similarity(v1, v2)
    # Clamp to valid range for arccos
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return np.degrees(np.arccos(cos_sim))


def huber_mean(data, delta=1.35):
    """
    Calculate Huber mean (robust to outliers).
    Uses iterative reweighting with Huber loss function.
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.nan

    mu = np.median(data)
    for _ in range(10):
        residuals = data - mu
        weights = np.where(np.abs(residuals) <= delta, 1.0, delta / np.abs(residuals))
        mu = np.sum(weights * data) / np.sum(weights)
    return mu


def calculate_class_stats(df, label_col, features):
    """Calculate Huber mean for each interpretable feature by class."""
    stats = {}
    for label in [0, 1]:
        class_data = df[df[label_col] == label]
        stats[label] = {}
        for feat in features:
            if feat in class_data.columns:
                stats[label][feat] = huber_mean(class_data[feat].values)
    return stats


def calculate_embedding_class_means(df, label_col):
    """Calculate mean embedding vectors for each class."""
    embedding_means = {}
    for prefix in EMBEDDING_PREFIXES:
        cols = get_embedding_cols(df, prefix)
        if not cols:
            continue
        embedding_means[prefix] = {}
        for label in [0, 1]:
            class_data = df[df[label_col] == label]
            # Use regular mean for embeddings (Huber doesn't apply well to vectors)
            mean_vector = class_data[cols].mean().values
            embedding_means[prefix][label] = mean_vector
    return embedding_means


def load_model(model_path, input_dim, device):
    model = ClickbaitClassifier(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def get_predictions(model, X, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        probs = outputs.cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
    return preds, probs


def print_feature_comparison(sample_features, class_stats, true_label,
                             sample_embeddings, embedding_means):
    """Print sample features compared to class means, including embedding angles."""
    print(f"\n  Feature Comparison (Sample vs Class Means):")
    print(f"  {'Feature':<25} {'Sample':>10} {'Clickbait':>12} {'Non-CB':>12} {'Deviation':>12}")
    print(f"  {'-'*71}")

    # Interpretable features
    for feat, val in sample_features.items():
        cb_mean = class_stats[1].get(feat, np.nan)
        ncb_mean = class_stats[0].get(feat, np.nan)

        expected_mean = cb_mean if true_label == 1 else ncb_mean
        if not np.isnan(expected_mean) and expected_mean != 0:
            deviation = ((val - expected_mean) / abs(expected_mean)) * 100
            dev_str = f"{deviation:+.1f}%"
        else:
            dev_str = "N/A"

        print(f"  {feat:<25} {val:>10.4f} {cb_mean:>12.4f} {ncb_mean:>12.4f} {dev_str:>12}")

    # Embedding vector angles
    print(f"  {'-'*71}")
    print(f"  {'Embedding (angle to mean)':<25} {'---':>10} {'to CB mean':>12} {'to Non-CB':>12} {'Diff':>12}")
    print(f"  {'-'*71}")

    for prefix, sample_vec in sample_embeddings.items():
        if prefix not in embedding_means:
            continue

        cb_mean_vec = embedding_means[prefix][1]
        ncb_mean_vec = embedding_means[prefix][0]

        angle_to_cb = angle_degrees(sample_vec, cb_mean_vec)
        angle_to_ncb = angle_degrees(sample_vec, ncb_mean_vec)

        # Deviation: positive means closer to clickbait, negative means closer to non-clickbait
        deviation = angle_to_ncb - angle_to_cb

        # Clean up prefix for display
        name = prefix.rstrip('_').upper()
        print(f"  {name:<25} {'':>10} {angle_to_cb:>11.1f}° {angle_to_ncb:>11.1f}° {deviation:>+11.1f}°")


def print_example(row, pred, prob, category, interpretable_features, class_stats,
                  embedding_means, df_columns):
    """Print example with feature comparison."""
    print(f"\n{'='*80}")
    print(f"[{category}] Video ID: {row['video_id']}")
    print(f"True Label: {'CLICKBAIT' if row['true_label'] == 1 else 'NON-CLICKBAIT'}")
    print(f"Predicted:  {'CLICKBAIT' if pred == 1 else 'NON-CLICKBAIT'} (confidence: {prob:.1%})")

    # Get sample feature values (interpretable)
    sample_features = {feat: row[feat] for feat in interpretable_features if feat in row.index}

    # Extract sample embedding vectors
    sample_embeddings = {}
    for prefix in EMBEDDING_PREFIXES:
        cols = get_embedding_cols(pd.DataFrame(columns=df_columns), prefix)
        if cols and all(c in row.index for c in cols):
            sample_embeddings[prefix] = row[cols].values.astype(float)

    if sample_features or sample_embeddings:
        print_feature_comparison(sample_features, class_stats, row['true_label'],
                                 sample_embeddings, embedding_means)

    if pd.notna(row.get('vllm_description')):
        desc = str(row['vllm_description'])[:400]
        print(f"\n  Thumbnail Description:\n  {desc}...")

    print(f"\n  YouTube URL: https://youtube.com/watch?v={row['video_id']}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data...")
    df_test = pd.read_parquet(test_path)
    df_train = pd.read_parquet(train_path)
    df_desc = pd.read_csv(DESCRIPTIONS_PATH)

    label_col = "clickbait_label"

    # Identify interpretable features
    interpretable_features = get_interpretable_features(df_train)
    print(f"\nInterpretable features found: {interpretable_features}")

    # Calculate class statistics using Huber mean
    print(f"\nCalculating class statistics (Huber mean)...")
    class_stats = calculate_class_stats(df_train, label_col, interpretable_features)

    # Calculate embedding class means
    print(f"Calculating embedding class means...")
    embedding_means = calculate_embedding_class_means(df_train, label_col)

    # Print class means for interpretable features
    print(f"\n{'='*80}")
    print("CLASS FEATURE MEANS (Huber Mean - Robust to Outliers)")
    print(f"{'='*80}")
    print(f"\n{'Feature':<25} {'Clickbait (1)':>15} {'Non-Clickbait (0)':>18} {'Difference':>12}")
    print(f"{'-'*70}")
    for feat in interpretable_features:
        cb_mean = class_stats[1].get(feat, np.nan)
        ncb_mean = class_stats[0].get(feat, np.nan)
        diff = cb_mean - ncb_mean if not (np.isnan(cb_mean) or np.isnan(ncb_mean)) else np.nan
        print(f"{feat:<25} {cb_mean:>15.4f} {ncb_mean:>18.4f} {diff:>+12.4f}")

    # Print angle between class mean embeddings
    print(f"\n{'='*80}")
    print("EMBEDDING CLASS MEAN SEPARATION (Angle between class centroids)")
    print(f"{'='*80}")
    print(f"\n{'Embedding':<25} {'Dimensions':>12} {'Angle between means':>20}")
    print(f"{'-'*60}")
    for prefix in EMBEDDING_PREFIXES:
        if prefix in embedding_means:
            cb_vec = embedding_means[prefix][1]
            ncb_vec = embedding_means[prefix][0]
            angle = angle_degrees(cb_vec, ncb_vec)
            dims = len(cb_vec)
            name = prefix.rstrip('_').upper()
            print(f"{name:<25} {dims:>12} {angle:>19.2f}°")

    # Prepare features for model
    video_ids = df_test['video_id'].values if 'video_id' in df_test.columns else df_test.index.values
    feature_cols = [col for col in df_test.columns if col not in [label_col, 'video_id']]
    numeric_cols = df_test[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X_test = df_test[numeric_cols].values.astype(np.float32)
    y_test = df_test[label_col].values

    # Get input_dim from training data
    train_feature_cols = [col for col in df_train.columns if col not in [label_col, 'video_id']]
    train_numeric_cols = df_train[train_feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    input_dim = len(train_numeric_cols)

    print(f"\nLoading model (input dim: {input_dim})...")
    model = load_model(MODEL_PATH, input_dim, device)

    predictions, probabilities = get_predictions(model, X_test, device)

    # Results
    accuracy = np.mean(predictions == y_test)
    print(f"\n{'='*80}")
    print(f"TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Total samples: {len(y_test)}")

    tp_mask = (predictions == 1) & (y_test == 1)
    tn_mask = (predictions == 0) & (y_test == 0)
    fp_mask = (predictions == 1) & (y_test == 0)
    fn_mask = (predictions == 0) & (y_test == 1)

    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Positives (correctly identified clickbait):     {np.sum(tp_mask)}")
    print(f"  True Negatives (correctly identified non-clickbait): {np.sum(tn_mask)}")
    print(f"  False Positives (non-clickbait marked as clickbait): {np.sum(fp_mask)}")
    print(f"  False Negatives (clickbait missed):                  {np.sum(fn_mask)}")

    # Create results dataframe with all data
    results = df_test.copy()
    results['predicted'] = predictions
    results['probability'] = probabilities
    results['true_label'] = y_test

    # Merge descriptions
    results = results.merge(df_desc[['video_id', 'vllm_description']], on='video_id', how='left')

    # Store column names for embedding extraction
    df_columns = list(df_test.columns)

    # Show examples
    n_examples = 2

    print(f"\n\n{'#'*80}")
    print("EXAMPLE ANALYSIS WITH FEATURE COMPARISON")
    print(f"{'#'*80}")

    # True Positives
    print(f"\n\n{'='*80}")
    print("TRUE POSITIVES - Model correctly identifies clickbait")
    print("These show typical clickbait patterns the model learned")
    print(f"{'='*80}")
    tp_results = results[(results['predicted'] == 1) & (results['true_label'] == 1)].head(n_examples)
    for _, row in tp_results.iterrows():
        print_example(row, 1, row['probability'], "TRUE POSITIVE",
                      interpretable_features, class_stats, embedding_means, df_columns)

    # True Negatives
    print(f"\n\n{'='*80}")
    print("TRUE NEGATIVES - Model correctly identifies non-clickbait")
    print("These show authentic content patterns")
    print(f"{'='*80}")
    tn_results = results[(results['predicted'] == 0) & (results['true_label'] == 0)].head(n_examples)
    for _, row in tn_results.iterrows():
        print_example(row, 0, row['probability'], "TRUE NEGATIVE",
                      interpretable_features, class_stats, embedding_means, df_columns)

    # False Positives
    print(f"\n\n{'='*80}")
    print("FALSE POSITIVES - Non-clickbait incorrectly flagged as clickbait")
    print("These may have visual elements similar to clickbait")
    print(f"{'='*80}")
    fp_results = results[(results['predicted'] == 1) & (results['true_label'] == 0)].head(n_examples)
    if len(fp_results) > 0:
        for _, row in fp_results.iterrows():
            print_example(row, 1, row['probability'], "FALSE POSITIVE",
                          interpretable_features, class_stats, embedding_means, df_columns)
    else:
        print("\nNo false positives found in test set!")

    # False Negatives
    print(f"\n\n{'='*80}")
    print("FALSE NEGATIVES - Clickbait that the model missed")
    print("These may use subtler manipulation tactics")
    print(f"{'='*80}")
    fn_results = results[(results['predicted'] == 0) & (results['true_label'] == 1)].head(n_examples)
    if len(fn_results) > 0:
        for _, row in fn_results.iterrows():
            print_example(row, 0, row['probability'], "FALSE NEGATIVE",
                          interpretable_features, class_stats, embedding_means, df_columns)
    else:
        print("\nNo false negatives found in test set!")

    # Summary
    print(f"\n\n{'#'*80}")
    print("SUMMARY FOR DISCUSSION")
    print(f"{'#'*80}")
    print("""
RELEVANCE OF THIS STUDY:
- YouTube clickbait affects billions of users daily
- Misleading thumbnails waste viewer time and erode trust
- Platforms struggle to balance creator freedom with user protection

PRACTICAL APPLICATIONS:

1. Platform Policy & Content Moderation:
   - Automated flagging system for review queues
   - Reduce monetization for repeat clickbait offenders
   - Transparency labels ("This thumbnail may be exaggerated")

2. Creator Tools:
   - Pre-upload thumbnail feedback
   - Suggestions for more authentic representations
   - A/B testing with authenticity scores

3. User Empowerment:
   - Browser extensions showing clickbait probability
   - Feed filtering options
   - Educational awareness about manipulation tactics

4. Research Applications:
   - Track clickbait trends over time
   - Study cross-platform manipulation patterns
   - Analyze impact on viewer behavior

LIMITATIONS TO CONSIDER:
- Model trained on specific dataset; may not generalize to all content types
- Cultural context affects what's considered "clickbait"
- Evolving tactics may require continuous model updates
- False positives could unfairly penalize legitimate creators
""")