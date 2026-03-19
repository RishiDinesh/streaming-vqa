#!/usr/bin/env python3
"""
Download diverse CC0 videos from Pexels for use as haystack backgrounds.

Usage:
    1. Get a FREE API key at https://www.pexels.com/api/ (instant, no credit card)
    2. Run:
       python3 download_pexels_videos.py --api_key YOUR_KEY --num_videos 1500

All Pexels videos are CC0 (Creative Commons Zero) — free for any use.
"""

import os
import json
import time
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Diverse search queries to ensure visual variety in the haystack
SEARCH_QUERIES = [
    # Nature & Landscapes
    "ocean waves", "mountain landscape", "forest aerial", "desert sand",
    "waterfall nature", "sunset sky", "aurora borealis", "underwater coral",
    "snow blizzard", "rain drops", "thunderstorm", "spring flowers",
    "autumn leaves", "river flowing", "lake reflection", "jungle canopy",
    # Urban & Architecture
    "city skyline", "busy street", "traffic timelapse", "neon lights",
    "skyscraper view", "subway train", "bridge aerial", "market crowd",
    "cafe interior", "library reading", "museum gallery", "office workspace",
    "construction site", "airport terminal", "harbor boats", "rooftop view",
    # People & Activities
    "cooking kitchen", "people walking", "yoga exercise", "dance performance",
    "sports running", "swimming pool", "cycling road", "hiking trail",
    "painting art", "writing desk", "reading book", "playing guitar",
    "workshop crafts", "gardening plants", "fishing lake", "camping fire",
    # Animals
    "birds flying", "fish aquarium", "cat playing", "dog running",
    "horse galloping", "butterfly garden", "elephant safari", "deer forest",
    # Technology & Science
    "computer coding", "laboratory science", "robot technology", "space stars",
    "drone aerial", "factory machines", "3d printing", "microscope lab",
    # Abstract & Texture
    "abstract motion", "smoke texture", "light bokeh", "water bubbles",
    "fire flames", "ink water", "geometric pattern", "particle effect",
    # Food
    "food preparation", "bakery bread", "coffee pouring", "fruit market",
]


def download_video(url, output_path, timeout=60):
    """Download a single video file."""
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def search_pexels(api_key, query, per_page=40, page=1, min_duration=15):
    """Search Pexels for videos matching a query."""
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": api_key}
    params = {
        "query": query,
        "per_page": per_page,
        "page": page,
        "min_duration": min_duration,
        "orientation": "landscape",
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  API error for '{query}': {e}")
        return None


def get_best_video_file(video_data, max_width=640):
    """Pick the best quality video file closest to max_width (SD by default)."""
    files = video_data.get("video_files", [])
    # Filter to mp4 files
    mp4_files = [f for f in files if f.get("file_type") == "video/mp4"]
    if not mp4_files:
        mp4_files = files

    # Prefer the file closest to (but not exceeding) max_width
    # This avoids accidentally picking a 4K file
    eligible = [f for f in mp4_files if f.get("width", 9999) <= max_width]
    if eligible:
        # Pick the largest among eligible (closest to max_width)
        eligible.sort(key=lambda f: f.get("width", 0), reverse=True)
        return eligible[0].get("link")

    # Fallback: pick the smallest available if nothing is under max_width
    mp4_files.sort(key=lambda f: f.get("width", 9999))
    if mp4_files:
        return mp4_files[0].get("link")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download diverse CC0 videos from Pexels."
    )
    parser.add_argument("--api_key", type=str, required=True,
                        help="Pexels API key (free at pexels.com/api)")
    parser.add_argument("--num_videos", type=int, default=500,
                        help="Target number of videos to download")
    parser.add_argument("--output_dir", type=str,
                        default="datasets/extracted/pexels",
                        help="Output directory for downloaded videos")
    parser.add_argument("--min_duration", type=int, default=15,
                        help="Minimum video duration in seconds")
    parser.add_argument("--max_width", type=int, default=640,
                        help="Maximum video width in pixels (640 = SD, matches VNBench)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel download workers")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Track already-downloaded video IDs to avoid duplicates
    downloaded_ids = set()
    existing_files = os.listdir(args.output_dir)
    for f in existing_files:
        if f.endswith(".mp4"):
            if f.startswith("pexels_"):
                vid_id = f.replace("pexels_", "").replace(".mp4", "")
            else:
                vid_id = "renamed_" + f
            downloaded_ids.add(vid_id)

    print(f"Found {len(downloaded_ids)} already downloaded. Target: {args.num_videos}")

    download_queue = []  # list of (video_id, download_url)

    # Phase 1: Collect video URLs from Pexels API
    print(f"\nSearching Pexels across {len(SEARCH_QUERIES)} categories...")
    videos_per_query = max(1, (args.num_videos - len(downloaded_ids)) // len(SEARCH_QUERIES) + 2)

    for qi, query in enumerate(SEARCH_QUERIES):
        if len(downloaded_ids) >= args.num_videos:
            break

        # Fetch videos for this query until we got enough or run out of pages
        collected_for_query = 0
        page = 1
        while collected_for_query < videos_per_query and page <= 5: # max 5 pages per query
            if len(downloaded_ids) >= args.num_videos:
                break

            data = search_pexels(args.api_key, query, per_page=40, page=page,
                                 min_duration=args.min_duration)
            if not data or "videos" not in data or len(data["videos"]) == 0:
                break # no more videos for this query

            for video in data["videos"]:
                if len(downloaded_ids) >= args.num_videos:
                    break
                if collected_for_query >= videos_per_query:
                    break

                vid_id = str(video["id"])
                if vid_id in downloaded_ids:
                    continue

                url = get_best_video_file(video, args.max_width)
                if url:
                    download_queue.append((vid_id, url))
                    downloaded_ids.add(vid_id)
                    collected_for_query += 1

            page += 1
            # Respect rate limits (200 req/hr for free tier)
            time.sleep(0.5)

        if (qi + 1) % 10 == 0:
            print(f"  Queried {qi + 1}/{len(SEARCH_QUERIES)} categories, "
                  f"{len(download_queue)} new videos queued")

    print(f"\nTotal new videos to download: {len(download_queue)}")

    # Phase 2: Download videos in parallel
    success = 0
    fail = 0

    def _download_one(item):
        vid_id, url = item
        output_path = os.path.join(args.output_dir, f"pexels_{vid_id}.mp4")
        if os.path.exists(output_path):
            return True
        return download_video(url, output_path)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_download_one, item): item
                   for item in download_queue}

        for future in as_completed(futures):
            if future.result():
                success += 1
            else:
                fail += 1

            total = success + fail
            if total % 50 == 0:
                print(f"  Downloaded {success}/{total} ({fail} failed)")

    total_videos = len(existing_files) + success
    print(f"\nDone! Downloaded {success} new videos ({fail} failed)")
    print(f"Total videos in {args.output_dir}: {total_videos}")


if __name__ == "__main__":
    main()
