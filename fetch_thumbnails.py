import csv
import os
import urllib.request
from urllib.parse import urlparse, parse_qs

def get_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    """
    try:
        query = urlparse(url)
        if query.hostname == 'youtu.be':
            return query.path[1:]
        if query.hostname in ('www.youtube.com', 'youtube.com'):
            if query.path == '/watch':
                p = parse_qs(query.query)
                return p.get('v', [None])[0]
            if query.path[:7] == '/embed/':
                return query.path.split('/')[2]
            if query.path[:3] == '/v/':
                return query.path.split('/')[2]
    except Exception:
        return None
    return None

def fetch_thumbnails(csv_files, output_dir):
    """
    Reads URLs from CSV files and downloads their thumbnails to the output directory.
    Uses the public YouTube thumbnail URL (hqdefault.jpg).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"File {csv_file} not found.")
            continue

        print(f"Processing {csv_file}...")
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get('url', '').strip()
                if not url:
                    continue
                
                video_id = get_video_id(url)
                if video_id:
                    # Construct the thumbnail URL
                    # hqdefault.jpg is a high quality thumbnail that is generally available
                    img_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
                    
                    filename = f"{video_id}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    
                    if not os.path.exists(filepath):
                        try:
                            print(f"Downloading thumbnail for {video_id}...")
                            urllib.request.urlretrieve(img_url, filepath)
                        except Exception as e:
                            print(f"Failed to download {img_url}: {e}")
                    else:
                        # print(f"Thumbnail for {video_id} already exists.")
                        pass
                else:
                    print(f"Could not extract video ID from {url}")

if __name__ == "__main__":
    # Configuration
    CSV_FILES = ['mtv.csv', 'nmtv.csv']
    OUTPUT_DIR = 'media'
    
    fetch_thumbnails(CSV_FILES, OUTPUT_DIR)
