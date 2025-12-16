import pandas as pd
from itertools import islice
from downloader import *
from time import perf_counter


# Variables
output_file_path = 'C:/Users/abarg/Documents/Uppsala Universitet/Mining-Of-Social-Data/project/data/comments.tsv'


def get_urls():
    df = pd.read_csv('C:/Users/abarg/Documents/Uppsala Universitet/Mining-Of-Social-Data/project/data/transcript_data.csv')
    urls = df['URL'].tolist()
    return urls


def scrape_all_comments(urls, max_comments_per_video=50):
    downloader = YoutubeCommentDownloader()
    all_comments = []
    
    for n, url in enumerate(urls):
        print(f"Scraping comments from ({n+1}/{len(urls)}): {url}")
        try:
            comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
            for comment in islice(comments, max_comments_per_video):
                comment['video_url'] = url  # Add the video URL to each comment
                all_comments.append(comment)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    
    return all_comments


def main():
    start_time = perf_counter()
    urls = get_urls()
    all_comments = scrape_all_comments(urls, max_comments_per_video=50)
    end_time = perf_counter()
    print(f"Scraping completed in {end_time - start_time:.2f} seconds.")
    
    # Save to CSV
    df = pd.DataFrame(all_comments)
    df.to_csv(output_file_path, index=False, sep='\t')
    print(f"Saved {len(all_comments)} comments to {output_file_path}")


if __name__ == "__main__":
    main()
