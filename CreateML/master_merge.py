import pandas as pd
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split


def ask_yes_no(prompt):
    """Ask a yes/no question and return True for yes, False for no."""
    while True:
        response = input(f"{prompt} (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please answer 'yes' or 'no'.")


def get_user_feature_selection():
    """Prompt the user to select which features to include."""
    print("\n" + "="*60)
    print("FEATURE SELECTION FOR MASTER DATASET")
    print("="*60 + "\n")
    
    selections = {}
    
    # Ask about CNN visual features
    selections['use_cnn'] = ask_yes_no("Include CNN visual features (ResNet embeddings)?")
    
    # Ask about VLLM text embeddings
    selections['use_vllm'] = ask_yes_no("Include VLLM text embeddings?")
    
    # Ask about CLIP similarity scores
    selections['use_clip'] = ask_yes_no("Include CLIP similarity scores?")
    
    if selections['use_clip']:
        selections['clip_max'] = ask_yes_no("  - Include CLIP max similarity?")
        selections['clip_mean'] = ask_yes_no("  - Include CLIP mean similarity?")
        # If user said yes to CLIP but no to both sub-options, warn and disable CLIP
        if not selections['clip_max'] and not selections['clip_mean']:
            print("  Warning: No CLIP features selected, disabling CLIP.")
            selections['use_clip'] = False
    else:
        selections['clip_max'] = False
        selections['clip_mean'] = False
    
    # Ask about incongruence scores
    selections['use_incong'] = ask_yes_no("Include incongruence scores?")
    
    # Check if at least one feature is selected
    if not any([selections['use_cnn'], selections['use_vllm'], 
                selections['use_clip'], selections['use_incong']]):
        print("\nError: You must select at least one feature type!")
        return None
    
    return selections


def split_dataset(final_df, output_dir, test_size=0.15, valid_size=0.15, random_state=42):
    """Split the dataset into train, validation, and test sets with stratification."""
    
    # Calculate actual validation size from remaining data after test split
    valid_ratio = valid_size / (1 - test_size)
    
    # First split: separate test set
    train_valid_df, test_df = train_test_split(
        final_df,
        test_size=test_size,
        random_state=random_state,
        stratify=final_df['clickbait_label']
    )
    
    # Second split: separate validation from training
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=valid_ratio,
        random_state=random_state,
        stratify=train_valid_df['clickbait_label']
    )
    
    # Save splits
    train_df.to_parquet(os.path.join(output_dir, 'train.parquet'), index=False)
    valid_df.to_parquet(os.path.join(output_dir, 'valid.parquet'), index=False)
    test_df.to_parquet(os.path.join(output_dir, 'test.parquet'), index=False)
    
    # Get label counts for each split
    train_labels = train_df['clickbait_label'].value_counts().to_dict()
    valid_labels = valid_df['clickbait_label'].value_counts().to_dict()
    test_labels = test_df['clickbait_label'].value_counts().to_dict()
    
    total = len(final_df)
    
    split_info = {
        'random_state': random_state,
        'train': {
            'samples': len(train_df),
            'ratio': f"{round(len(train_df) / total * 100, 1)}%",
            'label_0': train_labels.get(0, 0),
            'label_1': train_labels.get(1, 0)
        },
        'validation': {
            'samples': len(valid_df),
            'ratio': f"{round(len(valid_df) / total * 100, 1)}%",
            'label_0': valid_labels.get(0, 0),
            'label_1': valid_labels.get(1, 0)
        },
        'test': {
            'samples': len(test_df),
            'ratio': f"{round(len(test_df) / total * 100, 1)}%",
            'label_0': test_labels.get(0, 0),
            'label_1': test_labels.get(1, 0)
        }
    }
    
    print(f"\nDataset split:")
    print(f"  Train: {split_info['train']['samples']} samples ({split_info['train']['ratio']})")
    print(f"  Valid: {split_info['validation']['samples']} samples ({split_info['validation']['ratio']})")
    print(f"  Test:  {split_info['test']['samples']} samples ({split_info['test']['ratio']})")
    
    return split_info


def generate_folder_name(selections):
    """Generate a folder name based on selected features."""
    parts = []
    
    if selections['use_cnn']:
        parts.append('cnn')
    if selections['use_vllm']:
        parts.append('vllm')
    if selections['use_clip']:
        clip_parts = []
        if selections['clip_max']:
            clip_parts.append('max')
        if selections['clip_mean']:
            clip_parts.append('mean')
        parts.append('clip_' + '_'.join(clip_parts))
    if selections['use_incong']:
        parts.append('incong')
    
    return '_'.join(parts)


def generate_dataset_summary(final_df, selections, output_dir, feature_dims, split_info=None):
    """Generate and save a JSON summary of the dataset."""
    
    # Calculate class balance
    label_counts = final_df['clickbait_label'].value_counts().to_dict()
    total_samples = len(final_df)
    
    # Calculate balance ratio
    if len(label_counts) == 2:
        min_class = min(label_counts.values())
        max_class = max(label_counts.values())
        balance_ratio = min_class / max_class if max_class > 0 else 0
    else:
        balance_ratio = None
    
    # Count feature columns (excluding video_id and clickbait_label)
    feature_cols = [col for col in final_df.columns if col not in ['video_id', 'clickbait_label']]
    total_feature_dim = len(feature_cols)
    
    summary = {
        "created_at": datetime.now().isoformat(),
        "total_samples": total_samples,
        "total_feature_dimensions": total_feature_dim,
        "features_included": {
            "cnn_visual_features": selections['use_cnn'],
            "vllm_text_embeddings": selections['use_vllm'],
            "clip_similarity": {
                "enabled": selections['use_clip'],
                "max_similarity": selections['clip_max'],
                "mean_similarity": selections['clip_mean']
            },
            "incongruence_score": selections['use_incong']
        },
        "feature_dimensions": feature_dims,
        "class_distribution": {
            "label_counts": {str(k): v for k, v in label_counts.items()},
            "label_percentages": {str(k): round(v / total_samples * 100, 2) for k, v in label_counts.items()},
            "balance_ratio": round(balance_ratio, 4) if balance_ratio else None,
            "is_balanced": balance_ratio >= 0.8 if balance_ratio else None
        },
        "data_split": split_info,
        "columns": list(final_df.columns)
    }
    
    summary_path = os.path.join(output_dir, 'dataset_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Dataset summary saved to: {summary_path}")
    return summary


def load_and_merge_features(cnn_path, vllm_path, clip_path, incong_path, output_base_dir, selections):
    """Load and merge features based on user selections."""
    
    # 1. LOAD DATA SOURCES (only load what's needed)
    print("\nLoading feature files...")
    
    dataframes = []
    feature_dims = {}
    
    # Always load clip_df for clickbait_label and video_id (base dataset)
    clip_df = pd.read_csv(clip_path)
    base_df = clip_df[['video_id', 'clickbait_label']].copy()
    
    # Add CLIP features if selected
    if selections['use_clip']:
        clip_cols = []
        if selections['clip_max']:
            clip_cols.append('clip_max_similarity')
            feature_dims['clip_max_similarity'] = 1
        if selections['clip_mean']:
            clip_cols.append('clip_mean_similarity')
            feature_dims['clip_mean_similarity'] = 1
        base_df = base_df.merge(clip_df[['video_id'] + clip_cols], on='video_id')
    
    master_df = base_df
    
    # Load and merge CNN features if selected
    if selections['use_cnn']:
        print("  Loading CNN visual features...")
        cnn_df = pd.read_parquet(cnn_path)
        master_df = master_df.merge(cnn_df, on='video_id')
        # Get dimension from first row
        cnn_dim = len(cnn_df['v_cnn'].iloc[0])
        feature_dims['cnn_embedding'] = cnn_dim
    
    # Load and merge VLLM features if selected
    if selections['use_vllm']:
        print("  Loading VLLM text embeddings...")
        vllm_emb_df = pd.read_parquet(vllm_path)
        master_df = master_df.merge(vllm_emb_df, on='video_id')
        # Get dimension from first row
        vllm_dim = len(vllm_emb_df['T_vllm'].iloc[0])
        feature_dims['vllm_embedding'] = vllm_dim
    
    # Load and merge incongruence scores if selected
    if selections['use_incong']:
        print("  Loading incongruence scores...")
        incong_df = pd.read_csv(incong_path)
        master_df = master_df.merge(incong_df, on='video_id')
        feature_dims['incongruence_score'] = 1

    print(f"Aligned {len(master_df)} videos across selected modalities.")

    # 2. VECTOR EXPANSION (Flattening)
    print("Expanding embedding vectors into feature columns...")
    
    dfs_to_concat = [master_df[['video_id', 'clickbait_label']]]
    
    # Add CLIP features
    if selections['use_clip']:
        clip_cols = []
        if selections['clip_max']:
            clip_cols.append('clip_max_similarity')
        if selections['clip_mean']:
            clip_cols.append('clip_mean_similarity')
        dfs_to_concat.append(master_df[clip_cols])
    
    # Add incongruence score
    if selections['use_incong']:
        dfs_to_concat.append(master_df[['incongruence_score']])
    
    # Expand CNN vectors
    if selections['use_cnn']:
        cnn_vecs = pd.DataFrame(master_df['v_cnn'].tolist(), index=master_df.index)
        cnn_vecs.columns = [f'cnn_{i}' for i in range(cnn_vecs.shape[1])]
        dfs_to_concat.append(cnn_vecs)
    
    # Expand VLLM vectors
    if selections['use_vllm']:
        vllm_vecs = pd.DataFrame(master_df['T_vllm'].tolist(), index=master_df.index)
        vllm_vecs.columns = [f'vllm_{i}' for i in range(vllm_vecs.shape[1])]
        dfs_to_concat.append(vllm_vecs)

    # 3. FINAL CONCATENATION
    final_df = pd.concat(dfs_to_concat, axis=1)

    # 4. CREATE OUTPUT DIRECTORY
    folder_name = generate_folder_name(selections)
    output_dir = os.path.join(output_base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'master_vector_dataset.parquet')

    # 5. SAVE FOR TRAINING
    final_df.to_parquet(output_path, index=False)
    
    print(f"\nSuccess! Master file created with shape: {final_df.shape}")
    print(f"Saved to: {output_path}")
    
    # 6. SPLIT DATASET
    split_info = split_dataset(final_df, output_dir)
    
    # 7. GENERATE AND SAVE SUMMARY
    summary = generate_dataset_summary(final_df, selections, output_dir, feature_dims, split_info)


if __name__ == "__main__":

    # Define file paths
    input_dir = 'features/'
    output_base_dir = 'datasets/'

    cnn_path = input_dir + 'visual_features.parquet'
    vllm_path = input_dir + 'vllm_embeddings.parquet'
    clip_path = input_dir + 'clip_features.csv'
    incong_path = input_dir + 'incongruence_scores.csv'
    
    # Get user feature selection
    selections = get_user_feature_selection()
    
    if selections is None:
        print("Exiting due to invalid selection.")
    else:
        # Print selection summary
        print("\n" + "-"*40)
        print("Selected features:")
        if selections['use_cnn']:
            print("  ✓ CNN visual features")
        if selections['use_vllm']:
            print("  ✓ VLLM text embeddings")
        if selections['use_clip']:
            clip_info = []
            if selections['clip_max']:
                clip_info.append("max")
            if selections['clip_mean']:
                clip_info.append("mean")
            print(f"  ✓ CLIP similarity ({', '.join(clip_info)})")
        if selections['use_incong']:
            print("  ✓ Incongruence scores")
        print("-"*40)
        
        # Run the merging process
        load_and_merge_features(cnn_path, vllm_path, clip_path, incong_path, output_base_dir, selections)