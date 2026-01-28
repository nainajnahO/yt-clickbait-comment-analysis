"""
Visual Feature Extraction Script

This script extracts ResNet50 visual features from all thumbnail images and saves them to a parquet file.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# IMPORT v_cnn_extraction
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v_cnn_extraction import create_feature_extractor, preprocess_image


def extract_visual_features_batch(thumbnail_dirs, video_ids, labels, batch_size=32):
    """
    Extract ResNet50 visual features from all thumbnails in batch mode.

    Args:
        thumbnail_dirs (dict): Dictionary mapping labels to thumbnail directory paths.
                               e.g., {0: 'path/to/NMTV_Thumbnails', 1: 'path/to/MTV_Thumbnails'}
        video_ids (dict): Dictionary mapping labels to lists of video IDs.
        labels (list): List of label values (0 for NMTV, 1 for MTV).
        batch_size (int): Number of images to process in each batch.

    Returns:
        df (pd.DataFrame): DataFrame with columns: video_id, v_cnn, label
    """
    # INITIALIZE ResNet50
    weights = ResNet50_Weights.DEFAULT
    resnet = resnet50(weights=weights)
    feature_extractor = create_feature_extractor(resnet)
    feature_extractor.eval()

    # CHECK CUDA
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    feature_extractor = feature_extractor.to(device)

    all_features = []
    all_video_ids = []

    # PROCESS EACH CATEGORY (MTV AND NMTV)
    for label in labels:
        thumbnail_dir = thumbnail_dirs[label]
        video_id_list = video_ids[label]

        print(f"\nProcessing label {label} ({len(video_id_list)} images) from {thumbnail_dir}...")

        # PROCESS IN BATCHES
        for i in tqdm(range(0, len(video_id_list), batch_size), desc=f"Label {label}"):
            batch_video_ids = video_id_list[i:i + batch_size]
            batch_tensors = []
            batch_valid_ids = []

            # LOAD AND PROCESS BATCHES
            for video_id in batch_video_ids:
                image_path = os.path.join(thumbnail_dir, f"{video_id}.jpg")

                # PREPROCESS IMAGE
                img_tensor = preprocess_image(image_path, weights)
                batch_tensors.append(img_tensor)
                batch_valid_ids.append(video_id)

            if len(batch_tensors) == 0:
                continue

            # STACK TENSORS INTO BATCH
            batch = torch.cat(batch_tensors, dim=0).to(device)

            # EXTRACT FEATURES OF BATCH
            with torch.no_grad():
                features = feature_extractor(batch)
                features = features.view(features.size(0), -1)  # Flatten

            # CONVERT TO NUMPY ARRAY
            features_np = features.cpu().numpy()

            for j, video_id in enumerate(batch_valid_ids):
                all_features.append(features_np[j])
                all_video_ids.append(video_id)

    # CREATE DATAFRAME
    df = pd.DataFrame({
        'video_id': all_video_ids,
        'v_cnn': all_features
    })

    return df


def main():
    """Main function to extract visual features and save to parquet."""
    # DEFINE PATHS
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), 'data', 'ThumbnailTruthData')

    mtv_csv = os.path.join(data_dir, 'mtv_cleaned.csv')
    nmtv_csv = os.path.join(data_dir, 'nmtv_cleaned.csv')

    mtv_thumbnails = os.path.join(data_dir, 'MTV_Thumbnails')
    nmtv_thumbnails = os.path.join(data_dir, 'NMTV_Thumbnails')

    output_dir = os.path.join(base_dir, 'features')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'visual_features.parquet')

    # LOAD VIDEO ID'S
    mtv_df = pd.read_csv(mtv_csv)
    nmtv_df = pd.read_csv(nmtv_csv)

    mtv_video_ids = mtv_df['video_id'].tolist()
    nmtv_video_ids = nmtv_df['video_id'].tolist()

    # SET UP DIRECTORIES AND LABELS
    thumbnail_dirs = {
        0: nmtv_thumbnails,  # NMTV = label 0
        1: mtv_thumbnails  # MTV = label 1
    }

    video_ids = {
        0: nmtv_video_ids,
        1: mtv_video_ids
    }

    labels = [0, 1]

    # EXTRACT FEATURES
    df = extract_visual_features_batch(thumbnail_dirs, video_ids, labels, batch_size=32)

    # SAVE TO PARQUET
    df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)


if __name__ == "__main__":
    main()
