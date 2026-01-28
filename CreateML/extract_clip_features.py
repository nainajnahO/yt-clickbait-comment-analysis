"""
CLIP Feature Extraction Script

This script extracts semantic alignment scores between thumbnails and transcripts using CLIP.
The alignment score (Sclip) measures how well the thumbnail matches the video content.
"""

# TODO: CLIP has a strict limit of 77 tokens (roughly 50-75 words)!!!
# Since video transcripts are long, the code cuts the text to the first 500 characters

import os
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


def extract_clip_similarity(image_path, transcript, model, processor, device):
    """
    Calculate semantic alignment by checking the thumbnail against
    the entire transcript using a sliding window.
    """
    image = Image.open(image_path).convert('RGB')

    # Preprocess Image only once
    image_input = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_emb = model.get_image_features(**image_input)
        image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True) # Normalize

    # Define window size (CLIP's sweet spot is around 60-70 words)
    words = transcript.split()
    window_size = 60
    stride = 30

    # CREATE CHUNKS
    chunks = [" ".join(words[i: i + window_size]) for i in range(0, len(words), stride)]

    max_similarity = -1.0

    # Batch process text chunks
    text_inputs = processor(
        text=chunks,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True) # Normalize

    # Compute similarities
    # image_emb is (1, D), text_embs is (N, D)
    similarities = torch.matmul(image_emb, text_emb.T).squeeze(0)

    max_similarity = torch.max(similarities).item()
    mean_similarity = torch.mean(similarities).item()

    return (max_similarity, mean_similarity)


def process_videos(video_data, thumbnail_dir, model, processor, device, label):
    """
    Process a batch of videos to extract CLIP features.

    Args:
        video_data (pd.DataFrame): DataFrame with video_id and transcript columns.
        thumbnail_dir (str): Directory containing thumbnail images.
        model: CLIP model.
        processor: CLIP processor.
        device: Torch device.
        label (int): Label for these videos (0 or 1).

    Returns:
        list: List of dictionaries with video_id, clip_similarity, and label.
    """
    results = []
    failed_count = 0

    # TQDM FOR VISUAL FEEDBACK (PROGRESS BAR)
    for _, row in tqdm(video_data.iterrows(), total=len(video_data), desc=f"Label {label}"):
        # FOR EACH VIDEO
        video_id = row['video_id']
        thumb_path = os.path.join(thumbnail_dir, f"{video_id}.jpg")

        # FOR EACH TRANSCRIPT
        transcript = row['transcript']

        # DETERMINE SIMILARITY
        sclip = extract_clip_similarity(thumb_path, transcript, model, processor, device)

        # APPEND RESULT
        results.append({
            'video_id': video_id,
            'clip_max_similarity': sclip[0],
            'clip_mean_similarity': sclip[1],
            'clickbait_label': label
        })

    # RETURN
    return results


def main():
    # DEFINE PATHS
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), 'data', 'ThumbnailTruthData')

    mtv_transcripts_path = os.path.join(data_dir, 'MTV_transcripts.tsv')
    nmtv_transcripts_path = os.path.join(data_dir, 'NMTV_transcripts.tsv')

    mtv_thumb_dir = os.path.join(data_dir, 'MTV_Thumbnails')
    nmtv_thumb_dir = os.path.join(data_dir, 'NMTV_Thumbnails')

    output_dir = os.path.join(base_dir, 'features')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'clip_features.csv')

    # LOAD CLIP MODEL
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # SET DEVICE
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # LOAD
    mtv_transcripts = pd.read_csv(mtv_transcripts_path, sep='\t')
    nmtv_transcripts = pd.read_csv(nmtv_transcripts_path, sep='\t')

    # PROCESS
    all_results = []

    # MTV
    mtv_results = process_videos(mtv_transcripts, mtv_thumb_dir, model, processor, device, label=1)
    all_results.extend(mtv_results)

    # NMTV
    nmtv_results = process_videos(nmtv_transcripts, nmtv_thumb_dir, model, processor, device, label=0)
    all_results.extend(nmtv_results)

    # CREATE DATAFRAME AND SAVE
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
