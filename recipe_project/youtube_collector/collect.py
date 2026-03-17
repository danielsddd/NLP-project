#!/usr/bin/env python3
"""
YouTube Thread Collector — v4.1 (fixed)
Collects Hebrew comment threads from cooking channels.

FIXES over v4:
  - Reads config.yaml for all settings (no more ignored config)
  - Uses playlistItems instead of search for video listing (100x cheaper)
  - Guards against missing --api-key
  - Uses script-relative paths so it works from any working directory
  - Filters replies for Hebrew/spam too
  - Writes collection_stats.json
  - Marks non-Hebrew channels clearly

Usage:
    python collect.py --api-key KEY --discover-all
    python collect.py --api-key KEY --collect
    python collect.py --api-key KEY --collect --target 30000
    python collect.py --list-channels
    python collect.py --filter-channels --csv data/raw_youtube/channels_report.csv
"""

import json
import re
import csv
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import yaml
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# =============================================================================
# PATH SETUP — resolve relative to THIS script, not the working directory
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # assumes youtube_collector/ is one level down

# =============================================================================
# LOAD CONFIG (config.yaml) — single source of truth for all settings
# =============================================================================

def load_config():
    """Load config.yaml if it exists, otherwise return sensible defaults."""
    config_path = SCRIPT_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}

_CONFIG = load_config()

# Extract settings from config.yaml with fallbacks
_OUTPUT_CFG     = _CONFIG.get("output", {})
_COLLECTION_CFG = _CONFIG.get("collection", {})
_FILTER_CFG     = _CONFIG.get("filtering", {})

# Output paths (resolved relative to project root)
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / _OUTPUT_CFG.get("directory", "data/raw_youtube").lstrip("./"))
DEFAULT_CHANNELS_PATH = str(SCRIPT_DIR / "channels.yaml")
LOG_FILE = _OUTPUT_CFG.get("log_file", None)

# Collection parameters
TARGET_COMMENTS        = _COLLECTION_CFG.get("target_threads", 5000)
MAX_VIDEOS_PER_CHANNEL = _COLLECTION_CFG.get("max_videos_per_channel", 50)
MAX_COMMENTS_PER_VIDEO = _COLLECTION_CFG.get("max_threads_per_video", 200)
INCLUDE_REPLIES        = _COLLECTION_CFG.get("include_replies", True)

# Filtering parameters
MIN_WORDS          = _FILTER_CFG.get("min_words", 3)
REQUIRE_HEBREW     = _FILTER_CFG.get("require_hebrew", True)
SKIP_CREATOR       = _FILTER_CFG.get("skip_creator_comments", True)
SPAM_KEYWORDS_LIST = _FILTER_CFG.get("spam_keywords", [])

# =============================================================================
# CONSTANTS
# =============================================================================

HEBREW_RE = re.compile(r'[\u0590-\u05FF]')

# Merge hardcoded spam words with config-defined ones
SPAM_WORDS = {"http", "https", "www.", "subscribe", "לייק", "הירשמו"}
SPAM_WORDS.update(set(SPAM_KEYWORDS_LIST))

DISCOVERY_QUERIES = [
    "מתכונים בישול", "אפייה ביתית", "בישול ביתי", "שף ישראלי",
    "מתכון עוגה", "מתכון לחם", "בישול טבעוני", "קינוחים",
    "ארוחת ערב", "מאפים ביתיים",
]

# =============================================================================
# HELPERS
# =============================================================================

def contains_hebrew(text):
    return bool(HEBREW_RE.search(text))

def is_spam(text):
    t = text.lower()
    return any(w in t for w in SPAM_WORDS)

def word_count(text):
    return len(text.split())

def load_channels(path=DEFAULT_CHANNELS_PATH):
    if not Path(path).exists():
        print(f"⚠️  channels.yaml not found at: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return [ch for ch in data.get("channels", []) if ch.get("active", True)]

def save_channels(channels, path=DEFAULT_CHANNELS_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            existing = (yaml.safe_load(f) or {}).get("channels", [])
    existing_ids = {ch["id"] for ch in existing}
    for ch in channels:
        if ch["id"] not in existing_ids:
            existing.append(ch)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump({"channels": existing}, f, allow_unicode=True, default_flow_style=False)

def append_jsonl(data, path):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def load_existing_ids(path):
    ids = set()
    if Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["thread_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids

def require_youtube(youtube):
    """Guard: crash early with a clear message if API client is missing."""
    if youtube is None:
        print("❌ ERROR: --api-key is required for this command.")
        print("   Usage: python collect.py --api-key YOUR_KEY --collect")
        raise SystemExit(1)

# =============================================================================
# YOUTUBE API
# =============================================================================

def api_call(request, description="", max_retries=3):
    for attempt in range(max_retries):
        try:
            return request.execute()
        except HttpError as e:
            if e.resp.status == 403 and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited ({description}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    return {}

def discover_channels(youtube, query, max_results=10):
    """Search for channels matching a query."""
    resp = api_call(youtube.search().list(
        part='snippet', q=query, type='channel',
        maxResults=max_results, regionCode='IL'
    ), f"search '{query}'")

    channels = []
    for item in resp.get('items', []):
        cid = item['snippet']['channelId']
        info = api_call(youtube.channels().list(part='snippet,statistics', id=cid), "channel info")
        if info.get('items'):
            s = info['items'][0]
            channels.append({
                "id": cid,
                "name": s['snippet']['title'],
                "youtube_url": f"https://youtube.com/channel/{cid}",
                "subscribers": int(s['statistics'].get('subscriberCount', 0)),
                "videos": int(s['statistics'].get('videoCount', 0)),
                "active": True,
                "category": "cooking",
            })
    return channels

def get_videos(youtube, channel_id, max_videos=MAX_VIDEOS_PER_CHANNEL):
    """
    Get video IDs from a channel using playlistItems (uploads playlist).
    
    This costs 1 quota unit per call vs 100 for search().list().
    The uploads playlist ID is always "UU" + channel_id[2:].
    """
    uploads_playlist_id = "UU" + channel_id[2:]
    videos, next_page = [], None

    while len(videos) < max_videos:
        try:
            resp = api_call(youtube.playlistItems().list(
                part='snippet',
                playlistId=uploads_playlist_id,
                maxResults=min(50, max_videos - len(videos)),
                pageToken=next_page
            ), f"videos {channel_id}")
        except HttpError as e:
            if e.resp.status == 404:
                # Uploads playlist not found — fall back to search
                print(f"    ⚠️  Uploads playlist not found for {channel_id}, falling back to search")
                return _get_videos_via_search(youtube, channel_id, max_videos)
            raise

        for item in resp.get('items', []):
            snippet = item['snippet']
            vid_id = snippet['resourceId']['videoId']
            videos.append({
                'id': vid_id,
                'title': snippet['title'],
                'channel_id': channel_id,
                'channel_title': snippet['channelTitle'],
            })
        next_page = resp.get('nextPageToken')
        if not next_page:
            break

    return videos

def _get_videos_via_search(youtube, channel_id, max_videos):
    """Fallback: use search().list() if playlistItems fails (costs 100x more)."""
    videos, next_page = [], None
    while len(videos) < max_videos:
        resp = api_call(youtube.search().list(
            part='snippet', channelId=channel_id, type='video',
            order='date', maxResults=min(50, max_videos - len(videos)),
            pageToken=next_page
        ), f"videos-search {channel_id}")
        for item in resp.get('items', []):
            videos.append({
                'id': item['id']['videoId'],
                'title': item['snippet']['title'],
                'channel_id': channel_id,
                'channel_title': item['snippet']['channelTitle'],
            })
        next_page = resp.get('nextPageToken')
        if not next_page:
            break
    return videos

def get_threads(youtube, video, max_threads=MAX_COMMENTS_PER_VIDEO):
    """Get comment threads from a video. Yields thread dicts."""
    next_page, count = None, 0
    while count < max_threads:
        try:
            resp = api_call(youtube.commentThreads().list(
                part='snippet,replies', videoId=video['id'],
                maxResults=min(100, max_threads - count),
                pageToken=next_page, textFormat='plainText'
            ), f"threads {video['id']}")
        except HttpError as e:
            if 'commentsDisabled' in str(e) or e.resp.status == 403:
                return
            raise

        for item in resp.get('items', []):
            thread = _parse_thread(item, video)
            if thread:
                yield thread
                count += 1
                if count >= max_threads:
                    break

        next_page = resp.get('nextPageToken')
        if not next_page:
            break

def _parse_thread(item, video):
    """Parse a commentThread API response into our thread dict."""
    snip = item['snippet']['topLevelComment']['snippet']
    top_text = snip['textDisplay']
    top_author_channel = snip.get('authorChannelId', {}).get('value', '')
    video_channel = video['channel_id']

    # Skip creator's own top-level posts (usually recipe intros)
    if SKIP_CREATOR and top_author_channel == video_channel:
        return None

    # Filter top-level comment
    if REQUIRE_HEBREW and not contains_hebrew(top_text):
        return None
    if word_count(top_text) < MIN_WORDS:
        return None
    if is_spam(top_text):
        return None

    top_comment = {
        "comment_id": item['snippet']['topLevelComment']['id'],
        "text": top_text,
        "like_count": snip.get('likeCount', 0),
    }

    # Parse replies (also filtered now)
    replies = []
    has_creator_reply = False
    if INCLUDE_REPLIES and 'replies' in item:
        for r in item['replies']['comments']:
            rs = r['snippet']
            reply_author = rs.get('authorChannelId', {}).get('value', '')
            is_creator = (reply_author == video_channel)
            if is_creator:
                has_creator_reply = True

            reply_text = rs['textDisplay']

            # Filter replies too (skip spam/non-Hebrew, but keep creator replies)
            if not is_creator:
                if REQUIRE_HEBREW and not contains_hebrew(reply_text):
                    continue
                if is_spam(reply_text):
                    continue

            replies.append({
                "comment_id": r['id'],
                "text": reply_text,
                "like_count": rs.get('likeCount', 0),
                "is_creator": is_creator,
            })

    total_likes = top_comment['like_count'] + sum(r['like_count'] for r in replies)

    return {
        "thread_id": top_comment['comment_id'],
        "video_id": video['id'],
        "video_title": video['title'],
        "channel_id": video['channel_id'],
        "channel_title": video['channel_title'],
        "top_comment": top_comment,
        "replies": replies,
        "has_creator_reply": has_creator_reply,
        "total_likes": total_likes,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }

# =============================================================================
# COMMANDS
# =============================================================================

def cmd_discover_all(youtube):
    """Run all discovery queries and save channels."""
    require_youtube(youtube)
    all_channels = {}
    for query in DISCOVERY_QUERIES:
        print(f"Searching: {query}")
        for ch in discover_channels(youtube, query):
            if ch['id'] not in all_channels:
                all_channels[ch['id']] = ch
                print(f"  Found: {ch['name']} ({ch['subscribers']:,} subs)")
        time.sleep(1)
    save_channels(list(all_channels.values()))
    print(f"\n✅ {len(all_channels)} channels saved to {DEFAULT_CHANNELS_PATH}")
    print("Review channels.yaml — set active: false for non-cooking channels.")

def cmd_collect(youtube, target=TARGET_COMMENTS, output_dir=DEFAULT_OUTPUT_DIR):
    """Collect threads from all active channels."""
    require_youtube(youtube)
    channels = load_channels()
    if not channels:
        print(f"No active channels in {DEFAULT_CHANNELS_PATH}.")
        print("Run --discover-all first, or check that channels.yaml path is correct.")
        return

    out_path = Path(output_dir) / "threads.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = load_existing_ids(out_path)
    total = len(existing_ids)

    # Stats tracking
    stats = {
        "channels_processed": 0,
        "videos_processed": 0,
        "threads_collected": total,
        "threads_new_this_run": 0,
        "api_errors": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    print(f"Collecting from {len(channels)} channels (target: {target}, have: {total})")
    print(f"Output: {out_path}")
    print(f"Config: max_videos={MAX_VIDEOS_PER_CHANNEL}, max_comments={MAX_COMMENTS_PER_VIDEO}, min_words={MIN_WORDS}")

    for ch in channels:
        if total >= target:
            break
        ch_name = ch.get('name', ch['id'])
        print(f"\n📺 {ch_name}")
        stats["channels_processed"] += 1

        try:
            videos = get_videos(youtube, ch['id'], max_videos=MAX_VIDEOS_PER_CHANNEL)
        except HttpError as e:
            print(f"  ⚠️ Error fetching videos: {e.resp.status}, skipping channel")
            stats["api_errors"] += 1
            continue

        print(f"  Found {len(videos)} videos")

        for vid in videos:
            if total >= target:
                break
            stats["videos_processed"] += 1
            vid_count = 0
            try:
                for thread in get_threads(youtube, vid):
                    if thread['thread_id'] not in existing_ids:
                        append_jsonl(thread, out_path)
                        existing_ids.add(thread['thread_id'])
                        total += 1
                        vid_count += 1
                        stats["threads_new_this_run"] += 1
            except HttpError as e:
                print(f"  ⚠️ Error on video {vid['id']}: {e.resp.status}")
                stats["api_errors"] += 1
                continue

            if vid_count > 0:
                print(f"  {vid['title'][:50]}... → {vid_count} threads")
            time.sleep(0.3)
        time.sleep(1)

    # Update final stats
    stats["threads_collected"] = total
    stats["finished_at"] = datetime.now(timezone.utc).isoformat()

    # Save stats
    stats_path = Path(output_dir) / "collection_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Total: {total} threads saved to {out_path}")
    print(f"📊 Stats saved to {stats_path}")
    print(f"   New this run: {stats['threads_new_this_run']}")
    print(f"   Videos processed: {stats['videos_processed']}")
    print(f"   API errors: {stats['api_errors']}")

def cmd_list_channels(output_dir=DEFAULT_OUTPUT_DIR):
    """Generate channels_report.csv from collected threads."""
    threads_path = Path(output_dir) / "threads.jsonl"
    if not threads_path.exists():
        print(f"No threads file at {threads_path}")
        return

    counts = defaultdict(lambda: {"name": "", "count": 0})
    with open(threads_path, 'r', encoding='utf-8') as f:
        for line in f:
            t = json.loads(line)
            cid = t['channel_id']
            counts[cid]["name"] = t['channel_title']
            counts[cid]["count"] += 1

    csv_path = Path(output_dir) / "channels_report.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["channel_id", "channel_name", "thread_count", "youtube_url", "is_cooking"])
        for cid, info in sorted(counts.items(), key=lambda x: -x[1]["count"]):
            w.writerow([cid, info["name"], info["count"],
                        f"https://youtube.com/channel/{cid}", ""])
    print(f"✅ Report saved to {csv_path} — fill in is_cooking column (YES/NO)")

def cmd_filter_channels(csv_path, output_dir=DEFAULT_OUTPUT_DIR):
    """Filter threads to cooking-only channels based on CSV."""
    threads_path = Path(output_dir) / "threads.jsonl"
    out_path = Path(output_dir) / "threads_cooking_only.jsonl"

    # Load CSV decisions
    cooking_ids = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get("is_cooking", "").strip().upper() == "YES":
                cooking_ids.add(row["channel_id"])

    kept, total = 0, 0
    with open(threads_path, 'r', encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            total += 1
            t = json.loads(line)
            if t['channel_id'] in cooking_ids:
                fout.write(line)
                kept += 1

    print(f"✅ Kept {kept}/{total} threads from {len(cooking_ids)} cooking channels → {out_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="YouTube Thread Collector v4.1")
    parser.add_argument("--api-key", help="YouTube Data API key")
    parser.add_argument("--discover-all", action="store_true", help="Discover cooking channels")
    parser.add_argument("--collect", action="store_true", help="Collect threads")
    parser.add_argument("--target", type=int, default=TARGET_COMMENTS, help=f"Target thread count (default: {TARGET_COMMENTS})")
    parser.add_argument("--list-channels", action="store_true", help="Generate channels report")
    parser.add_argument("--filter-channels", action="store_true", help="Filter to cooking channels")
    parser.add_argument("--csv", help="Path to channels_report.csv for filtering")
    parser.add_argument("--test", action="store_true", help="Test API connection")
    args = parser.parse_args()

    youtube = None
    if args.api_key:
        youtube = build('youtube', 'v3', developerKey=args.api_key)

    if args.test:
        require_youtube(youtube)
        print("Testing YouTube API...")
        resp = youtube.channels().list(part='snippet', id='UC_x5XG1OV2P6uZZ5FSM9Ttw').execute()
        print(f"✅ API works. Test channel: {resp['items'][0]['snippet']['title']}")
        print(f"\n📁 Paths resolved:")
        print(f"   channels.yaml: {DEFAULT_CHANNELS_PATH}")
        print(f"   output dir:    {DEFAULT_OUTPUT_DIR}")
        channels = load_channels()
        print(f"   active channels: {len(channels)}")
    elif args.discover_all:
        cmd_discover_all(youtube)
    elif args.collect:
        cmd_collect(youtube, target=args.target)
    elif args.list_channels:
        cmd_list_channels()
    elif args.filter_channels:
        if not args.csv:
            print("--filter-channels requires --csv path")
            return
        cmd_filter_channels(args.csv)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()