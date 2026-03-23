#!/usr/bin/env python3
"""
Collect threads likely to contain recipe modifications, with emphasis on
question‑answer conversational chains. Saves to a separate file for later
merging.

Fixes:
  1. Requires at least one linguistic signal (question or modification keyword)
     to avoid collecting "thank you" threads.
  2. Uses regex with Hebrew prefix handling to catch attached question words.
  3. Limits pages per video to avoid wasting API quota on viral videos with few
     modifications.
"""

import json
import re
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

import yaml
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# =============================================================================
# Reuse paths and config loading from original collector
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

def load_config():
    config_path = SCRIPT_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}

_CONFIG = load_config()
_OUTPUT_CFG = _CONFIG.get("output", {})
_COLLECTION_CFG = _CONFIG.get("collection", {})
_FILTER_CFG = _CONFIG.get("filtering", {})

# Defaults (same as original)
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / _OUTPUT_CFG.get("directory", "data/raw_youtube").lstrip("./"))
DEFAULT_CHANNELS_PATH = str(SCRIPT_DIR / "channels.yaml")

TARGET_POSITIVES = 5000
MAX_VIDEOS_PER_CHANNEL = _COLLECTION_CFG.get("max_videos_per_channel", 50)
MAX_THREADS_PER_VIDEO = _COLLECTION_CFG.get("max_threads_per_video", 200)
INCLUDE_REPLIES = _COLLECTION_CFG.get("include_replies", True)
MIN_WORDS = _FILTER_CFG.get("min_words", 3)
REQUIRE_HEBREW = _FILTER_CFG.get("require_hebrew", True)
SKIP_CREATOR = _FILTER_CFG.get("skip_creator_comments", True)
SPAM_KEYWORDS_LIST = _FILTER_CFG.get("spam_keywords", [])

# Modification keywords (from config) for scoring
MOD_KEYWORDS = _CONFIG.get("modification_keywords", {})
ALL_MOD_KEYWORDS = set()
for cat, kwlist in MOD_KEYWORDS.items():
    ALL_MOD_KEYWORDS.update(kwlist)

# Heuristic weights
SCORE_WEIGHTS = {
    "question": 10,
    "has_reply": 5,
    "creator_reply": 15,
    "mod_keyword_in_top": 8,
    "mod_keyword_in_reply": 12,
}

# Hebrew text markers (question words)
TEXT_MARKERS = ["אפשר", "האם", "כדאי", "איך", "מה", "למה", "מתי", "איפה", "מי"]

# Regex that allows for common Hebrew prefixes attached to the question word.
PREFIXES = r'(?:^|\s|[.,!?״"\'\-+=])(?:[והכלבמש]{1,2})?'
SUFFIXES = r'(?=\s|[.,!?״"\'\-+=]|$)'
QUESTION_RE = re.compile(PREFIXES + '(' + '|'.join(re.escape(w) for w in TEXT_MARKERS) + ')' + SUFFIXES, re.IGNORECASE)

HEBREW_RE = re.compile(r'[\u0590-\u05FF]')

def contains_hebrew(text):
    return bool(HEBREW_RE.search(text))

def is_spam(text):
    t = text.lower()
    spam = {"http", "https", "www.", "subscribe", "לייק", "הירשמו"}
    spam.update(SPAM_KEYWORDS_LIST)
    return any(w in t for w in spam)

def word_count(text):
    return len(text.split())

def load_existing_ids(path):
    """Load thread IDs already collected to avoid duplicates (from original file too)."""
    ids = set()
    if Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["thread_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids

def append_jsonl(data, path):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def load_channels(path=DEFAULT_CHANNELS_PATH):
    if not Path(path).exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return [ch for ch in data.get("channels", []) if ch.get("active", True)]

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

def get_videos(youtube, channel_id, max_videos=MAX_VIDEOS_PER_CHANNEL):
    """Get video IDs from channel using playlistItems (cheaper than search)."""
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
                # Fallback to search
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

def compute_modification_score(top_text, replies, video_channel_id):
    """
    Heuristic score for a thread. Higher score = more likely to contain a recipe modification.
    Returns 0 if no linguistic signal (question or modification keyword) is present.
    """
    score = 0
    top_lower = top_text.lower()

    # Check for question marker: literal "?" or regex match of Hebrew question words (with prefix handling)
    has_question = "?" in top_lower or bool(QUESTION_RE.search(top_lower))
    has_top_kw = any(kw in top_lower for kw in ALL_MOD_KEYWORDS)
    has_reply_kw = False

    if has_question:
        score += SCORE_WEIGHTS["question"]
    if has_top_kw:
        score += SCORE_WEIGHTS["mod_keyword_in_top"]

    if replies:
        score += SCORE_WEIGHTS["has_reply"]
        for r in replies:
            reply_lower = r['text'].lower()
            if r.get('is_creator', False):
                score += SCORE_WEIGHTS["creator_reply"]
            if any(kw in reply_lower for kw in ALL_MOD_KEYWORDS):
                has_reply_kw = True
                score += SCORE_WEIGHTS["mod_keyword_in_reply"]

    # THE FIX: Force score to 0 if there is no linguistic signal
    if not (has_question or has_top_kw or has_reply_kw):
        return 0

    return score

def get_threads_with_score(youtube, video, max_threads=MAX_THREADS_PER_VIDEO, min_score=0):
    """
    Yields thread dicts if they pass basic filters and have a score >= min_score.
    Limits the number of comment pages fetched to avoid wasting quota on viral videos.
    """
    next_page = None
    count = 0
    pages_fetched = 0
    MAX_PAGES_PER_VIDEO = 5   # Stop after fetching 5 pages (500 comments) if no hits

    while count < max_threads and pages_fetched < MAX_PAGES_PER_VIDEO:
        try:
            resp = api_call(youtube.commentThreads().list(
                part='snippet,replies', videoId=video['id'],
                maxResults=100,   # Always get 100 per page to minimize calls
                pageToken=next_page, textFormat='plainText'
            ), f"threads {video['id']}")
        except HttpError as e:
            if 'commentsDisabled' in str(e) or e.resp.status == 403:
                return
            raise

        pages_fetched += 1

        for item in resp.get('items', []):
            thread = _parse_thread_with_score(item, video)
            if thread and thread['score'] >= min_score:
                yield thread
                count += 1
                if count >= max_threads:
                    break

        next_page = resp.get('nextPageToken')
        if not next_page:
            break

def _parse_thread_with_score(item, video):
    """Parse a commentThread item, apply basic filters, compute score."""
    snip = item['snippet']['topLevelComment']['snippet']
    top_text = snip['textDisplay']
    top_author_channel = snip.get('authorChannelId', {}).get('value', '')
    video_channel = video['channel_id']

    # Skip creator's own top-level posts
    if SKIP_CREATOR and top_author_channel == video_channel:
        return None

    # Basic filtering
    if REQUIRE_HEBREW and not contains_hebrew(top_text):
        return None
    if word_count(top_text) < MIN_WORDS:
        return None
    if is_spam(top_text):
        return None

    # Build top comment dict
    top_comment = {
        "comment_id": item['snippet']['topLevelComment']['id'],
        "text": top_text,
        "like_count": snip.get('likeCount', 0),
    }

    # Parse replies
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
            # Filter replies (spam, non-Hebrew) but keep creator replies
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

    # Compute heuristic score
    score = compute_modification_score(top_text, replies, video_channel)

    # If score is 0, discard immediately (no linguistic signal)
    if score == 0:
        return None

    total_likes = top_comment['like_count'] + sum(r['like_count'] for r in replies)

    thread = {
        "thread_id": top_comment['comment_id'],
        "video_id": video['id'],
        "video_title": video['title'],
        "channel_id": video['channel_id'],
        "channel_title": video['channel_title'],
        "top_comment": top_comment,
        "replies": replies,
        "has_creator_reply": has_creator_reply,
        "total_likes": total_likes,
        "score": score,                     # extra field for analysis
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    return thread

def cmd_collect_positives(youtube, target=TARGET_POSITIVES, output_dir=DEFAULT_OUTPUT_DIR,
                          min_score=20, channels_file=None):
    """
    Collect positive‑focused threads, save to a separate file.
    """
    require_youtube(youtube)

    # Use custom channels file if provided
    channels_path = channels_file or DEFAULT_CHANNELS_PATH
    channels = load_channels(channels_path)
    if not channels:
        print(f"No active channels in {channels_path}. Run --discover-all first.")
        return

    # Output path
    out_path = Path(output_dir) / "threads_positives_focus.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already collected thread IDs (from original collection too)
    existing_ids = load_existing_ids(Path(output_dir) / "threads.jsonl")
    existing_ids.update(load_existing_ids(out_path))   # also avoid duplicates in new file

    collected = 0
    stats = {
        "target": target,
        "collected": 0,
        "channels_processed": 0,
        "videos_processed": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    print(f"Collecting positive‑focused threads (min_score={min_score})")
    print(f"Target: {target} threads, output: {out_path}")

    for ch in channels:
        if collected >= target:
            break
        ch_name = ch.get('name', ch['id'])
        print(f"\n📺 {ch_name}")
        stats["channels_processed"] += 1

        try:
            videos = get_videos(youtube, ch['id'])
        except HttpError as e:
            print(f"  ⚠️ Error fetching videos: {e.resp.status}, skipping channel")
            continue

        print(f"  Found {len(videos)} videos")

        for vid in videos:
            if collected >= target:
                break
            stats["videos_processed"] += 1
            vid_count = 0
            try:
                for thread in get_threads_with_score(youtube, vid, max_threads=MAX_THREADS_PER_VIDEO,
                                                     min_score=min_score):
                    if thread['thread_id'] not in existing_ids:
                        append_jsonl(thread, out_path)
                        existing_ids.add(thread['thread_id'])
                        collected += 1
                        vid_count += 1
                        stats["collected"] += 1
            except HttpError as e:
                print(f"  ⚠️ Error on video {vid['id']}: {e.resp.status}")
                continue

            if vid_count > 0:
                print(f"  {vid['title'][:50]}... → {vid_count} threads (score >= {min_score})")
            time.sleep(0.3)
        time.sleep(1)

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    stats_path = Path(output_dir) / "collection_positives_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Collected {collected} positive‑focused threads → {out_path}")
    print(f"📊 Stats saved to {stats_path}")

def require_youtube(youtube):
    if youtube is None:
        print("❌ ERROR: --api-key is required.")
        raise SystemExit(1)

def main():
    parser = argparse.ArgumentParser(description="Collect positive‑focused threads")
    parser.add_argument("--api-key", help="YouTube Data API key")
    parser.add_argument("--collect", action="store_true", help="Run collection")
    parser.add_argument("--target", type=int, default=TARGET_POSITIVES,
                        help=f"Number of threads to collect (default: {TARGET_POSITIVES})")
    parser.add_argument("--min-score", type=int, default=20,
                        help="Minimum heuristic score to keep thread (higher = more likely positive)")
    parser.add_argument("--channels", help="Optional custom channels.yaml file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory (default: data/raw_youtube)")
    args = parser.parse_args()

    youtube = None
    if args.api_key:
        youtube = build('youtube', 'v3', developerKey=args.api_key)

    if args.collect:
        cmd_collect_positives(youtube, target=args.target, output_dir=args.output_dir,
                              min_score=args.min_score, channels_file=args.channels)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()