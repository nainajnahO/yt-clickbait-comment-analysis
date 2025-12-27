import csv
import json
import os
import urllib.request
import urllib.parse
import urllib.error
from fetch_thumbnails import get_video_id

def load_env_variable(key):
    """
    Simple function to load a variable from a .env file in the current directory.
    """
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == key:
                        return v.strip()
    return None

# Load API Key from .env
API_KEY = load_env_variable("YOUTUBE_KEY")

def fetch_video_comments(video_id):
    """
    Fetches top-level comments for a given video ID using the YouTube Data API.
    Returns a list of dictionaries containing author, text, and published date.
    """
    if not API_KEY:
        print("Error: YOUTUBE_KEY not found in .env file.")
        return []

    base_url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": API_KEY,
        "maxResults": 100,  # Max allowed per page
        "textFormat": "plainText"
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        with urllib.request.urlopen(url) as response:
            if response.status != 200:
                print(f"Error: Received status code {response.status}")
                return []
            
            data = json.loads(response.read().decode('utf-8'))
            comments = []
            
            for item in data.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'video_id': video_id,
                    'author': snippet.get('authorDisplayName', 'Unknown'),
                    'text': snippet.get('textDisplay', ''),
                    'published_at': snippet.get('publishedAt', '')
                })
            
            return comments

    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("Error: 403 Forbidden. Please check your API Key and Quota.")
        elif e.code == 404:
            print(f"Video {video_id} not found or comments disabled.")
        else:
            print(f"HTTP Error fetching comments for {video_id}: {e}")
        return []
    except Exception as e:
        print(f"Error fetching comments for {video_id}: {e}")
        return []

def main():
    if not API_KEY:
        print("Please create a .env file with YOUTUBE_KEY=your_api_key")
        return

    # Updated paths to reflect the move to 'media' folder
    csv_files = [
        os.path.join('media', 'mtv.csv'),
        os.path.join('media', 'nmtv.csv')
    ]
    
    # Ensure media/comments directory exists
    output_dir = os.path.join("media", "comments")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, 'comments.csv')
    
    all_comments = []

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"File {csv_file} not found.")
            continue
            
        print(f"Reading {csv_file}...")
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get('url', '').strip()
                if not url:
                    continue
                
                video_id = get_video_id(url)
                if video_id:
                    print(f"Fetching comments for video {video_id}...")
                    video_comments = fetch_video_comments(video_id)
                    all_comments.extend(video_comments)
                else:
                    print(f"Skipping invalid URL: {url}")

    # Write results to CSV
    if all_comments:
        print(f"Writing {len(all_comments)} comments to {output_file}...")
        keys = ['video_id', 'author', 'published_at', 'text']
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(all_comments)
            print("Done.")
        except IOError as e:
            print(f"Error writing to file {output_file}: {e}")
    else:
        print("No comments found or fetched.")

if __name__ == "__main__":
    main()
