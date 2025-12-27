import csv
import os
import subprocess
import sys
from fetch_thumbnails import get_video_id

def fetch_transcripts(csv_files, output_dir):
    """
    Reads URLs from CSV files and fetches their transcripts by calling 
    youtube-transcript-api as a command-line tool via subprocess.
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
                    output_path = os.path.join(output_dir, f"{video_id}.txt")
                    
                    if os.path.exists(output_path):
                        # print(f"Transcript for {video_id} already exists.")
                        continue

                    print(f"Fetching transcript for {video_id}...")
                    
                    try:
                        # Run youtube_transcript_api as a module via the current python executable
                        # This avoids import issues within this script
                        cmd = [
                            sys.executable, 
                            "-m", "youtube_transcript_api", 
                            video_id, 
                            "--format", "text"
                        ]
                        
                        # Capture the output
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            # Success: write stdout to file
                            with open(output_path, "w", encoding="utf-8") as f_out:
                                f_out.write(result.stdout)
                            print(f"  Saved transcript to {output_path}")
                        else:
                            # Failure: check stderr for common reasons
                            error_msg = result.stderr
                            if "TranscriptsDisabled" in error_msg:
                                print(f"  Transcripts are disabled for this video.")
                            elif "NoTranscriptFound" in error_msg:
                                print(f"  No transcript found for this video.")
                            else:
                                # Print the first line of the error to avoid clutter
                                first_line = error_msg.strip().split('\n')[0] if error_msg else "Unknown error"
                                print(f"  Failed: {first_line}")

                    except Exception as e:
                        print(f"  Error executing subprocess: {e}")
                else:
                    print(f"Could not extract video ID from {url}")

if __name__ == "__main__":
    # Configuration
    CSV_FILES = [
        os.path.join('media', 'mtv.csv'),
        os.path.join('media', 'nmtv.csv')
    ]
    OUTPUT_DIR = os.path.join('media', 'transcripts')
    
    fetch_transcripts(CSV_FILES, OUTPUT_DIR)
