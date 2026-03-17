"""
YouTube Comment Collector - Main Script
========================================
Group 11: Daniel Simanovsky & Roei Ben Artzi
NLP 2025a - Recipe Modification Extraction

This script collects Hebrew comments from YouTube cooking channels.
Configuration is read from config.yaml and channels.yaml.

Usage:
    # Test API connection
    python collect.py --test

    # Discover new channels
    python collect.py --discover "מתכונים בישול"

    # Collect from all active channels
    python collect.py --collect

    # Collect from specific channel
    python collect.py --channel UC_CHANNEL_ID

    # Collect from specific video
    python collect.py --video VIDEO_ID

    # Auto-discover ALL queries from channels.yaml → write results back to channels.yaml
    python collect.py --discover-all
    # Then open channels.yaml, delete channels you don't want, set the rest active: true
    # Then run --collect

    # [NO API KEY NEEDED] Generate channel verification report from existing JSONL
    python collect.py --list-channels
    python collect.py --list-channels --input path/to/comments.jsonl

    # [NO API KEY NEEDED] Filter out non-cooking channels after manual review
    python collect.py --filter-channels --input comments.jsonl --csv channels_report.csv
"""

import os
import sys
import json
import re
import csv
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Generator, Any, Tuple
from collections import defaultdict

import yaml
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Comment:
    """
    Represents a YouTube comment with all fields needed for NLP pipeline.

    FIELDS NEEDED FOR:
      - NLP extraction  : text
      - Ranking module  : like_count
      - Context         : video_title, channel_title
      - Data management : comment_id, video_id
      - Channel audit   : channel_id  ← REQUIRED for YouTube channel URL
      - Filtering/debug : word_count, has_modification_keyword, detected_keywords
    """
    # === ESSENTIAL FOR NLP ===
    text: str                   # The comment content - main input for extraction
    like_count: int             # Social signal for ranking module
    video_title: str            # Context: which recipe
    channel_title: str          # Context: which cooking channel

    # === FOR DATA MANAGEMENT ===
    comment_id: str             # For deduplication
    video_id: str               # To group by recipe; also forms video URL
    channel_id: str = ""        # REQUIRED: forms channel URL for audit/filtering
                                #           youtube.com/channel/{channel_id}

    # === PRE-COMPUTED FOR FILTERING ===
    word_count: int = 0
    has_modification_keyword: bool = False
    detected_keywords: List[str] = field(default_factory=list)


@dataclass
class CollectionStats:
    """Statistics for the collection run."""
    start_time: str = ""
    end_time: str = ""
    channels_processed: int = 0
    videos_processed: int = 0
    comments_found: int = 0
    comments_kept: int = 0
    comments_filtered: int = 0
    api_calls: int = 0
    errors: int = 0
    filter_reasons: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

class Config:
    """Loads and provides access to configuration from YAML files."""

    def __init__(self, config_path: str = "config.yaml", channels_path: str = "channels.yaml"):
        # Load main config
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            print(f"⚠️  Config file not found: {config_path}. Using defaults.")
            self._config = {}

        # Load channels config
        channels_file = Path(channels_path)
        if channels_file.exists():
            with open(channels_file, 'r', encoding='utf-8') as f:
                self._channels = yaml.safe_load(f) or {}
        else:
            print(f"⚠️  Channels file not found: {channels_path}. No channels configured.")
            self._channels = {}

    def get(self, *keys, default=None):
        """Get a nested config value by key path."""
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value if value is not None else default

    def get_active_channels(self) -> List[Dict]:
        """Return channels marked active: true in channels.yaml."""
        channels = self._channels.get('channels', [])
        return [ch for ch in channels if ch.get('active', False)]

    def get_all_modification_keywords(self) -> List[str]:
        """Return flat list of all modification detection keywords."""
        keywords = []
        mod_config = self._config.get('modification_keywords', {})
        for category, words in mod_config.items():
            if isinstance(words, list):
                keywords.extend(words)
        return keywords


# =============================================================================
# HEBREW TEXT UTILITIES
# =============================================================================

class HebrewUtils:
    """Utilities for Hebrew text detection and analysis."""

    HEBREW_PATTERN = re.compile(r'[\u0590-\u05FF]')

    @classmethod
    def contains_hebrew(cls, text: str) -> bool:
        """Check if text contains at least one Hebrew character."""
        return bool(cls.HEBREW_PATTERN.search(text))

    @classmethod
    def word_count(cls, text: str) -> int:
        """Count whitespace-separated words."""
        return len(text.strip().split())

    @classmethod
    def is_spam(cls, text: str, spam_keywords: List[str]) -> bool:
        """Return True if text contains any spam indicator."""
        text_lower = text.lower()
        for keyword in spam_keywords:
            if keyword.lower() in text_lower:
                return True
        return False

    @classmethod
    def find_keywords(cls, text: str, keywords: List[str]) -> List[str]:
        """Return list of modification keywords found in text."""
        return [kw for kw in keywords if kw in text]


# =============================================================================
# YOUTUBE API CLIENT
# =============================================================================

class YouTubeClient:
    """Wrapper for YouTube Data API v3 with retry logic and stats tracking."""

    def __init__(self, api_key: str, config: Config):
        self.api_key = api_key
        self.config = config
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.stats = CollectionStats(start_time=datetime.now().isoformat())
        self.seen_ids: set = set()
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging to console and optionally a file."""
        log_file = self.config.get('output', 'log_file')
        handlers = [logging.StreamHandler()]
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=handlers,
        )
        self.logger = logging.getLogger(__name__)

    def _api_call(self, request, description: str = "") -> Dict:
        """
        Execute an API request with exponential-backoff retry.
        Raises immediately on quota-exceeded (403/429) so the caller can stop.
        """
        max_retries = self.config.get('api', 'max_retries', default=3)
        base_delay  = self.config.get('api', 'retry_delay', default=1.0)

        for attempt in range(max_retries):
            try:
                self.stats.api_calls += 1
                return request.execute()

            except HttpError as e:
                # Quota / auth errors are fatal – stop immediately
                if e.resp.status in (403, 429):
                    if any(x in str(e) for x in ('quotaExceeded', 'rateLimitExceeded', 'forbidden')):
                        self.logger.error(
                            "❌ API quota exceeded or forbidden! "
                            "Stop for today and resume tomorrow."
                        )
                        raise

                if attempt == max_retries - 1:
                    self.stats.errors += 1
                    raise

                delay = base_delay * (2 ** attempt)
                self.logger.warning(
                    f"API error ({description}): {e}. "
                    f"Retry {attempt + 1}/{max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)

    # -------------------------------------------------------------------------
    # CHANNEL / VIDEO DISCOVERY
    # -------------------------------------------------------------------------

    def search_channels(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search for channels matching a query string."""
        self.logger.info(f"🔍 Searching channels: '{query}'")

        region = self.config.get('api', 'region_code', default='IL')
        request = self.youtube.search().list(
            part='snippet',
            q=query,
            type='channel',
            maxResults=max_results,
            regionCode=region,
        )
        response = self._api_call(request, "channel search")

        channels = []
        for item in response.get('items', []):
            channel_id = item['snippet']['channelId']

            # Fetch subscriber + video counts
            info_request = self.youtube.channels().list(
                part='snippet,statistics',
                id=channel_id,
            )
            info_response = self._api_call(info_request, "channel info")

            if info_response.get('items'):
                info = info_response['items'][0]
                channels.append({
                    'id':          channel_id,
                    'name':        info['snippet']['title'],
                    'description': info['snippet'].get('description', '')[:200],
                    'subscribers': int(info['statistics'].get('subscriberCount', 0)),
                    'videos':      int(info['statistics'].get('videoCount', 0)),
                })

        return channels

    def get_channel_videos(
        self, channel_id: str, max_videos: int = 50
    ) -> Generator[Dict, None, None]:
        """Yield video dicts from a channel, newest first."""
        next_page = None
        retrieved = 0

        while retrieved < max_videos:
            request = self.youtube.search().list(
                part='snippet',
                channelId=channel_id,
                type='video',
                order='date',
                maxResults=min(50, max_videos - retrieved),
                pageToken=next_page,
            )
            response = self._api_call(request, f"videos for {channel_id}")

            for item in response.get('items', []):
                yield {
                    'id':            item['id']['videoId'],
                    'title':         item['snippet']['title'],
                    'channel_id':    channel_id,
                    'channel_title': item['snippet']['channelTitle'],
                    'published':     item['snippet']['publishedAt'],
                }
                retrieved += 1
                if retrieved >= max_videos:
                    break

            next_page = response.get('nextPageToken')
            if not next_page:
                break

    # -------------------------------------------------------------------------
    # COMMENT COLLECTION
    # -------------------------------------------------------------------------

    def get_video_comments(
        self, video: Dict, max_comments: int = 100
    ) -> Generator[Comment, None, None]:
        """Yield Comment objects from a video (top-level + replies)."""
        include_replies = self.config.get('collection', 'include_replies', default=True)
        next_page = None
        retrieved = 0

        while retrieved < max_comments:
            try:
                request = self.youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video['id'],
                    maxResults=min(100, max_comments - retrieved),
                    pageToken=next_page,
                    textFormat='plainText',
                )
                response = self._api_call(request, f"comments for {video['id']}")

                for item in response.get('items', []):
                    # Top-level comment
                    comment = self._parse_comment(item, video)
                    if comment:
                        yield comment
                        retrieved += 1

                    # Inline replies (saves an extra API request vs. fetching separately)
                    if include_replies and 'replies' in item:
                        for reply_item in item['replies']['comments']:
                            reply = self._parse_reply(reply_item, item, video)
                            if reply:
                                yield reply

                    if retrieved >= max_comments:
                        break

                next_page = response.get('nextPageToken')
                if not next_page:
                    break

            except HttpError as e:
                if 'commentsDisabled' in str(e):
                    self.logger.info(f"   Comments disabled for: {video['title'][:40]}...")
                else:
                    self.logger.warning(f"   Error getting comments: {e}")
                break

    def _parse_comment(self, item: Dict, video: Dict) -> Optional[Comment]:
        """
        Parse a commentThread item into a Comment.

        BUG FIX: channel_id is now stored so we can later form
                 https://youtube.com/channel/{channel_id}
        """
        snippet    = item['snippet']['topLevelComment']['snippet']
        comment_id = item['snippet']['topLevelComment']['id']

        # Skip duplicates
        if comment_id in self.seen_ids:
            return None
        self.seen_ids.add(comment_id)

        # Skip creator's own comments (usually pinned greetings, not modifications)
        author_channel = snippet.get('authorChannelId', {}).get('value', '')
        if self.config.get('filtering', 'skip_creator_comments', default=True):
            if author_channel == video['channel_id']:
                return None

        text = snippet['textDisplay']

        return Comment(
            text=text,
            like_count=snippet.get('likeCount', 0),
            video_title=video['title'],
            channel_title=video['channel_title'],
            comment_id=comment_id,
            video_id=video['id'],
            channel_id=video['channel_id'],          # ← BUG FIX (was missing)
            word_count=HebrewUtils.word_count(text),
        )

    def _parse_reply(
        self, reply_item: Dict, parent_item: Dict, video: Dict
    ) -> Optional[Comment]:
        """
        Parse a reply item into a Comment.

        BUG FIX: channel_id is now stored.
        """
        snippet  = reply_item['snippet']
        reply_id = reply_item['id']

        if reply_id in self.seen_ids:
            return None
        self.seen_ids.add(reply_id)

        text = snippet['textDisplay']

        return Comment(
            text=text,
            like_count=snippet.get('likeCount', 0),
            video_title=video['title'],
            channel_title=video['channel_title'],
            comment_id=reply_id,
            video_id=video['id'],
            channel_id=video['channel_id'],          # ← BUG FIX (was missing)
            word_count=HebrewUtils.word_count(text),
        )

    # -------------------------------------------------------------------------
    # FILTERING + ENRICHMENT
    # -------------------------------------------------------------------------

    def filter_comment(self, comment: Comment) -> Tuple[bool, str]:
        """
        Decide whether to keep a comment.

        Returns:
            (keep: bool, reason: str)  — reason explains why it was dropped
        """
        if self.config.get('filtering', 'require_hebrew', default=True):
            if not HebrewUtils.contains_hebrew(comment.text):
                return False, "no_hebrew"

        min_words = self.config.get('filtering', 'min_words', default=5)
        if comment.word_count < min_words:
            return False, "too_short"

        spam_keywords = self.config.get('filtering', 'spam_keywords', default=[])
        if HebrewUtils.is_spam(comment.text, spam_keywords):
            return False, "spam"

        return True, ""

    def enrich_comment(self, comment: Comment) -> Comment:
        """Pre-compute modification keyword flags for faster downstream filtering."""
        keywords = self.config.get_all_modification_keywords()
        found = HebrewUtils.find_keywords(comment.text, keywords)
        comment.has_modification_keyword = len(found) > 0
        comment.detected_keywords = found
        return comment


# =============================================================================
# CHANNEL REPORT GENERATOR  (no API key needed)
# =============================================================================

def generate_channel_report(
    comments_file: str,
    output_dir: str = None,
) -> None:
    """
    Read an existing comments.jsonl and produce a channel verification report.

    Outputs two files:
      channels_report.txt   — human-readable with clickable YouTube links
      channels_report.csv   — open in Excel, fill in is_cooking_channel (YES/NO)

    Usage (no API key required):
        python collect.py --list-channels
        python collect.py --list-channels --input data/raw_youtube/comments.jsonl
    """
    input_path = Path(comments_file)
    if not input_path.exists():
        print(f"❌ File not found: {comments_file}")
        print("   Run --collect first to gather comments.")
        return

    if output_dir is None:
        output_dir = input_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Reading: {comments_file}")

    # ── Aggregate by channel ──────────────────────────────────────────────────
    # channel_title → { channel_id, video_ids (ordered), video_titles, comment_count }
    channel_data: Dict[str, Dict] = defaultdict(lambda: {
        'channel_id':    '',
        'video_ids':     [],        # ordered list, for sample links
        'video_titles':  {},        # video_id → title
        'comment_count': 0,
    })

    total_comments = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue

            name = c.get('channel_title', 'Unknown Channel')
            ch   = channel_data[name]

            # Prefer channel_id from the new format (field added in this bug-fix release)
            if c.get('channel_id') and not ch['channel_id']:
                ch['channel_id'] = c['channel_id']

            vid_id    = c.get('video_id', '')
            vid_title = c.get('video_title', '')
            if vid_id and vid_id not in ch['video_titles']:
                ch['video_ids'].append(vid_id)
                ch['video_titles'][vid_id] = vid_title

            ch['comment_count'] += 1
            total_comments += 1

    if not channel_data:
        print("⚠️  No comments found in file.")
        return

    # Sort channels by descending comment count
    sorted_channels = sorted(
        channel_data.items(), key=lambda x: -x[1]['comment_count']
    )

    # ── TXT report ───────────────────────────────────────────────────────────
    txt_path = output_dir / "channels_report.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CHANNEL VERIFICATION REPORT\n")
        f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source    : {comments_file}\n")
        f.write(
            f"Total     : {total_comments:,} comments  |  "
            f"{len(sorted_channels)} unique channels\n"
        )
        f.write("=" * 80 + "\n\n")
        f.write("INSTRUCTIONS:\n")
        f.write("  1. Open each YouTube link below.\n")
        f.write("  2. Check whether it is a COOKING channel.\n")
        f.write("  3. Mark [ ] as [YES] (cooking) or [NO] (not cooking).\n")
        f.write("  4. For faster review, use channels_report.csv in Excel.\n")
        f.write("  5. Run --filter-channels to remove non-cooking comments.\n\n")
        f.write("-" * 80 + "\n\n")

        for rank, (ch_name, data) in enumerate(sorted_channels, 1):
            ch_id     = data['channel_id']
            count     = data['comment_count']
            vid_count = len(data['video_ids'])

            f.write(f"  #{rank:02d}  [ ] {ch_name}\n")
            f.write(f"        Comments : {count:,}  |  Videos seen : {vid_count}\n")

            if ch_id:
                f.write(
                    f"        Channel  : https://www.youtube.com/channel/{ch_id}\n"
                )
            else:
                f.write(
                    f"        Channel  : (channel_id not in data — see video links below;\n"
                    f"                    re-collect with updated script to get channel URLs)\n"
                )

            # Up to 3 sample video links
            f.write(f"        Videos   :\n")
            for vid_id in data['video_ids'][:3]:
                title = data['video_titles'].get(vid_id, 'Unknown Title')
                f.write(
                    f"          • {title[:55]:<55}  "
                    f"https://youtube.com/watch?v={vid_id}\n"
                )
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write(
            f"Total: {len(sorted_channels)} channels  |  {total_comments:,} comments\n"
        )

    print(f"📄 Text report  →  {txt_path}")

    # ── CSV report (for Excel) ────────────────────────────────────────────────
    csv_path = output_dir / "channels_report.csv"
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:  # utf-8-sig = Excel BOM
        writer = csv.writer(f)
        writer.writerow([
            'rank',
            'channel_name',
            'channel_id',
            'channel_url',
            'comment_count',
            'video_count',
            'sample_video_1_title', 'sample_video_1_url',
            'sample_video_2_title', 'sample_video_2_url',
            'sample_video_3_title', 'sample_video_3_url',
            'is_cooking_channel',   # ← Fill in: YES or NO
            'notes',
        ])

        for rank, (ch_name, data) in enumerate(sorted_channels, 1):
            ch_id  = data['channel_id']
            ch_url = (
                f"https://www.youtube.com/channel/{ch_id}" if ch_id else ""
            )

            row = [
                rank,
                ch_name,
                ch_id,
                ch_url,
                data['comment_count'],
                len(data['video_ids']),
            ]

            for i in range(3):
                if i < len(data['video_ids']):
                    vid_id = data['video_ids'][i]
                    title  = data['video_titles'].get(vid_id, '')
                    row.append(title)
                    row.append(f"https://youtube.com/watch?v={vid_id}")
                else:
                    row.append('')
                    row.append('')

            row.append('')   # is_cooking_channel — user fills this in
            row.append('')   # notes
            writer.writerow(row)

    print(f"📊 CSV report   →  {csv_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(
        f"CHANNEL SUMMARY  "
        f"({total_comments:,} comments, {len(sorted_channels)} channels)"
    )
    print(f"{'=' * 72}")
    print(f"  {'#':>3}  {'Channel Name':<40}  {'ID':<24}  {'Comments':>9}")
    print(f"  {'-'*3}  {'-'*40}  {'-'*24}  {'-'*9}")
    for rank, (ch_name, data) in enumerate(sorted_channels, 1):
        ch_id_display = data['channel_id'] or '(unknown — re-collect)'
        print(
            f"  {rank:>3}  {ch_name:<40}  {ch_id_display:<24}  "
            f"{data['comment_count']:>9,}"
        )
    print(f"{'=' * 72}")
    print()
    print("✅ Next steps:")
    print("   1. Open channels_report.csv in Excel")
    print("   2. Fill in is_cooking_channel column: YES or NO")
    print("   3. Run: python collect.py --filter-channels \\")
    print(f"               --input {comments_file} \\")
    print(f"               --csv   {csv_path}")
    print()


# =============================================================================
# CHANNEL FILTER  (no API key needed)
# =============================================================================

def filter_channels_from_jsonl(
    comments_file: str,
    csv_report: str,
    output_file: Optional[str] = None,
) -> None:
    """
    After manually filling is_cooking_channel in channels_report.csv,
    use this to strip non-cooking channel comments from comments.jsonl.

    Channels with an empty is_cooking_channel cell are KEPT (conservative default).

    Usage:
        python collect.py --filter-channels \\
            --input  data/raw_youtube/comments.jsonl \\
            --csv    data/raw_youtube/channels_report.csv \\
            --output data/raw_youtube/comments_cooking_only.jsonl
    """
    csv_path = Path(csv_report)
    if not csv_path.exists():
        print(f"❌ CSV report not found: {csv_report}")
        print("   Run --list-channels first, fill in the CSV, then run --filter-channels.")
        return

    # Parse verdicts
    approved: set = set()
    rejected: set = set()

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name    = row.get('channel_name', '').strip()
            verdict = row.get('is_cooking_channel', '').strip().upper()
            if verdict in ('YES', 'Y', '1', 'TRUE', 'COOKING'):
                approved.add(name)
            elif verdict in ('NO', 'N', '0', 'FALSE', 'NOT-COOKING', 'NOT COOKING'):
                rejected.add(name)
            else:
                # Empty or unknown → conservative: keep the channel
                approved.add(name)

    print(f"\n📋 Channel verdicts:")
    print(f"   Approved (COOKING)     : {len(approved)} channels")
    print(f"   Rejected (NOT-COOKING) : {len(rejected)} channels")

    if rejected:
        print("\n   Channels to be removed:")
        for name in sorted(rejected):
            print(f"     ✗  {name}")

    if not approved and not rejected:
        print("⚠️  No verdicts found in CSV. Fill in 'is_cooking_channel' column first.")
        return

    # Filter JSONL
    input_path = Path(comments_file)
    if not input_path.exists():
        print(f"❌ Input not found: {comments_file}")
        return

    if output_file is None:
        output_path = input_path.parent / (input_path.stem + "_cooking_only.jsonl")
    else:
        output_path = Path(output_file)

    kept    = 0
    removed = 0

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                c       = json.loads(line)
                ch_name = c.get('channel_title', '')
                if ch_name not in rejected:
                    fout.write(line + '\n')
                    kept += 1
                else:
                    removed += 1
            except json.JSONDecodeError:
                continue

    print(f"\n✅ Filtering complete:")
    print(f"   Kept    : {kept:,} comments")
    print(f"   Removed : {removed:,} comments")
    print(f"   Output  : {output_path}")


# =============================================================================
# CHANNELS.YAML WRITER
# =============================================================================

def write_channels_yaml(
    channels: List[Dict],
    channels_path: str,
    existing_data: Dict,
) -> None:
    """
    Write discovered channels back to channels.yaml.

    - All channels get active: true (user deletes the ones they don't want)
    - Preserves discovery_queries and any other top-level keys
    - Adds a youtube_url comment next to each channel for easy review
    """
    channels_file = Path(channels_path)

    # Build the new channels list as structured dicts
    channel_entries = []
    for ch in channels:
        ch_url = f"https://www.youtube.com/channel/{ch['id']}"
        entry = {
            'name':        ch['name'],
            'id':          ch['id'],
            'active':      True,
            'category':    'cooking',
            'subscribers': ch.get('subscribers', 0),
            'youtube_url': ch_url,   # Added so user can click straight from the file
            'notes':       ch.get('description', '')[:100].replace('\n', ' ').strip(),
        }
        channel_entries.append(entry)

    # Merge: keep existing top-level keys (discovery_queries, etc.) but replace channels list
    output_data = dict(existing_data)
    output_data['channels'] = channel_entries

    # Dump with a UTF-8-safe, human-readable format
    # PyYAML doesn't support inline comments, so we write a custom header then yaml.dump
    header = (
        "# =============================================================================\n"
        "# HEBREW COOKING CHANNELS — AUTO-GENERATED BY --discover-all\n"
        "# =============================================================================\n"
        "# HOW TO USE:\n"
        "#   1. Open this file and review each channel.\n"
        "#   2. Click the youtube_url to verify it is a cooking channel.\n"
        "#   3. DELETE the entire block for channels you don't want.\n"
        "#      OR set   active: false   to skip without deleting.\n"
        "#   4. Run:  python collect.py --collect\n"
        "# =============================================================================\n\n"
    )

    yaml_str = yaml.dump(
        output_data,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )

    with open(channels_file, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(yaml_str)

    print(f"💾 Written to: {channels_file}")


# =============================================================================
# MAIN COLLECTOR CLASS
# =============================================================================

class Collector:
    """Orchestrates collection across channels, handles save and reporting."""

    def __init__(
        self,
        api_key: str,
        config_path: str = "config.yaml",
        channels_path: str = "channels.yaml",
    ):
        self.config   = Config(config_path, channels_path)
        self.client   = YouTubeClient(api_key, self.config)
        self.collected: List[Comment] = []

        output_dir = self.config.get('output', 'directory', default='./data/raw_youtube')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def discover_channels(self, query: str) -> None:
        """Search for channels and print a summary with YouTube URLs."""
        print(f"\n🔍 Searching for channels: '{query}'\n")
        print("=" * 70)

        channels = self.client.search_channels(query, max_results=10)

        if not channels:
            print("No channels found.")
            return

        print(f"Found {len(channels)} channels:\n")
        for i, ch in enumerate(channels, 1):
            ch_url = f"https://www.youtube.com/channel/{ch['id']}"
            print(f"{i}. {ch['name']}")
            print(f"   ID          : {ch['id']}")
            print(f"   YouTube URL : {ch_url}")          # ← added for easy verification
            print(f"   Subscribers : {ch['subscribers']:,} | Videos: {ch['videos']}")
            print(f"   Description : {ch['description'][:80]}...")
            print()

        print("=" * 70)
        print("\nTo add a channel, copy its ID to channels.yaml:")
        print("""
  - name: "Channel Name"
    id: "UC_PASTE_ID_HERE"
    active: true
    category: "cooking"
""")

    def discover_all_channels(self, channels_path: str = "channels.yaml") -> None:
        """
        Run every query listed under discovery_queries in channels.yaml,
        deduplicate results, and write them ALL back to channels.yaml
        with active: true so the user can just delete the ones they don't want.

        Usage:
            python collect.py --discover-all
        """
        # Read existing channels.yaml to get the discovery_queries list
        channels_file = Path(channels_path)
        if channels_file.exists():
            with open(channels_file, 'r', encoding='utf-8') as f:
                existing_data = yaml.safe_load(f) or {}
        else:
            existing_data = {}

        queries = existing_data.get('discovery_queries', [
            "מתכונים בישול",
            "אפייה ביתית",
            "בישול ישראלי",
            "עוגות ומאפים",
            "מתכונים קלים",
            "שף ישראלי",
            "בישול בריא",
            "קינוחים",
            "ארוחות משפחתיות",
            "מתכונים לשבת",
        ])

        print(f"\n🔍 Auto-discovering channels from {len(queries)} queries...\n")
        print("=" * 70)

        # Collect all channels, dedup by channel ID
        seen_ids: Dict[str, Dict] = {}

        for query in queries:
            print(f"\n▶  Query: '{query}'")
            try:
                results = self.client.search_channels(query, max_results=10)
                new_count = 0
                for ch in results:
                    if ch['id'] not in seen_ids:
                        seen_ids[ch['id']] = ch
                        new_count += 1
                print(f"   Found {len(results)} channels  ({new_count} new, {len(results)-new_count} duplicates)")
                time.sleep(1.0)   # Be polite between queries
            except Exception as e:
                print(f"   ⚠️  Query failed: {e}")
                continue

        if not seen_ids:
            print("\n⚠️  No channels found. Check your API key.")
            return

        # Sort by subscriber count descending
        all_channels = sorted(seen_ids.values(), key=lambda x: -x.get('subscribers', 0))

        print(f"\n{'=' * 70}")
        print(f"DISCOVERED {len(all_channels)} UNIQUE CHANNELS")
        print(f"{'=' * 70}")
        print(f"  {'#':>3}  {'Channel Name':<40}  {'Subs':>10}  {'Videos':>7}")
        print(f"  {'-'*3}  {'-'*40}  {'-'*10}  {'-'*7}")
        for i, ch in enumerate(all_channels, 1):
            print(
                f"  {i:>3}  {ch['name']:<40}  "
                f"{ch['subscribers']:>10,}  {ch['videos']:>7,}"
            )
        print(f"{'=' * 70}")

        # Write back to channels.yaml
        write_channels_yaml(all_channels, channels_path, existing_data)

        print(f"\n✅ channels.yaml updated with {len(all_channels)} channels (all active: true)")
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Open channels.yaml")
        print(f"   2. Review each channel — click the YouTube URL in the file")
        print(f"   3. DELETE the entire block for any non-cooking channel")
        print(f"      (or set active: false to just disable without deleting)")
        print(f"   4. Run:  python collect.py --collect")

    def collect_from_channel(
        self, channel_id: str, channel_name: str = "Unknown"
    ) -> List[Comment]:
        """Collect and filter comments from one channel."""
        ch_url = f"https://www.youtube.com/channel/{channel_id}"
        print(f"\n📺 Collecting from: {channel_name}")
        print(f"   Channel URL : {ch_url}")

        max_videos   = self.config.get('collection', 'max_videos_per_channel', default=50)
        max_comments = self.config.get('collection', 'max_comments_per_video', default=100)
        delay        = self.config.get('api', 'delay_between_videos', default=0.5)

        channel_comments = []
        videos_processed = 0

        for video in self.client.get_channel_videos(channel_id, max_videos):
            videos_processed += 1
            self.client.stats.videos_processed += 1

            print(f"   [{videos_processed}/{max_videos}] {video['title'][:55]}...")

            video_comments = 0
            for comment in self.client.get_video_comments(video, max_comments):
                self.client.stats.comments_found += 1

                keep, reason = self.client.filter_comment(comment)
                if keep:
                    comment = self.client.enrich_comment(comment)
                    channel_comments.append(comment)
                    self.client.stats.comments_kept += 1
                    video_comments += 1
                else:
                    self.client.stats.comments_filtered += 1
                    self.client.stats.filter_reasons[reason] = (
                        self.client.stats.filter_reasons.get(reason, 0) + 1
                    )

            if video_comments > 0:
                print(f"       → {video_comments} comments kept")

            time.sleep(delay)

        self.client.stats.channels_processed += 1
        print(f"   ✓ Channel total: {len(channel_comments):,} comments")

        return channel_comments

    def collect_from_video(self, video_id: str) -> List[Comment]:
        """Collect comments from a single video (useful for quick tests)."""
        print(f"\n📹 Collecting from video: {video_id}")
        print(f"   URL: https://www.youtube.com/watch?v={video_id}")

        max_comments = self.config.get('collection', 'max_comments_per_video', default=100)

        # Fetch video metadata so channel_id and title are real values
        video = {
            'id':            video_id,
            'title':         f'Video {video_id}',
            'channel_id':    'unknown',
            'channel_title': 'Unknown Channel',
        }
        try:
            request  = self.client.youtube.videos().list(part='snippet', id=video_id)
            response = self.client._api_call(request, "video details")
            if response.get('items'):
                item = response['items'][0]
                video['title']         = item['snippet']['title']
                video['channel_id']    = item['snippet']['channelId']
                video['channel_title'] = item['snippet']['channelTitle']
                print(f"   Title   : {video['title']}")
                print(f"   Channel : {video['channel_title']}")
        except Exception as e:
            print(f"   ⚠️  Could not fetch video details: {e}")

        video_comments = []
        for comment in self.client.get_video_comments(video, max_comments):
            self.client.stats.comments_found += 1
            keep, reason = self.client.filter_comment(comment)
            if keep:
                comment = self.client.enrich_comment(comment)
                video_comments.append(comment)
                self.client.stats.comments_kept += 1
            else:
                self.client.stats.comments_filtered += 1
                self.client.stats.filter_reasons[reason] = (
                    self.client.stats.filter_reasons.get(reason, 0) + 1
                )

        print(f"   ✓ Collected {len(video_comments)} comments")
        return video_comments

    def collect_all(self) -> None:
        """Collect from all channels marked active: true in channels.yaml."""
        channels = self.config.get_active_channels()
        target   = self.config.get('collection', 'target_comments', default=3000)

        if not channels:
            print("⚠️  No active channels in channels.yaml!")
            print("   Add channels (active: true) or use --discover to find them.")
            return

        print(f"\n📋 Collecting from {len(channels)} active channels")
        print(f"   Target : {target:,} comments\n")

        for channel in channels:
            if len(self.collected) >= target:
                print(f"\n✅ Target of {target:,} comments reached. Stopping.")
                break

            ch_id   = channel.get('id', '')
            ch_name = channel.get('name', 'Unknown')

            comments = self.collect_from_channel(ch_id, ch_name)
            self.collected.extend(comments)
            print(f"   Running total: {len(self.collected):,} / {target:,}")

        self.save_results()

    def save_results(self) -> None:
        """
        Save comments.jsonl, collection_stats.json, and auto-generate
        channels_report.txt + channels_report.csv for manual review.
        """
        if not self.collected:
            print("\n⚠️  No comments to save")
            return

        # ── comments.jsonl ────────────────────────────────────────────────────
        comments_file = self.config.get('output', 'comments_file', default='comments.jsonl')
        comments_path = self.output_dir / comments_file

        with open(comments_path, 'w', encoding='utf-8') as f:
            for comment in self.collected:
                f.write(json.dumps(asdict(comment), ensure_ascii=False) + '\n')

        print(f"\n💾 Saved {len(self.collected):,} comments → {comments_path}")

        # ── collection_stats.json ─────────────────────────────────────────────
        self.client.stats.end_time = datetime.now().isoformat()
        stats_file = self.config.get('output', 'stats_file', default='collection_stats.json')
        stats_path = self.output_dir / stats_file

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.client.stats), f, indent=2, ensure_ascii=False)

        print(f"📊 Stats           → {stats_path}")

        # ── auto-generate channel verification report ──────────────────────────
        print(f"\n🔍 Generating channel verification report...")
        generate_channel_report(str(comments_path), str(self.output_dir))

        self.print_summary()

    def print_summary(self) -> None:
        """Print a human-readable collection summary."""
        stats = self.client.stats

        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        print(f"  Channels processed : {stats.channels_processed}")
        print(f"  Videos processed   : {stats.videos_processed}")
        print(f"  Comments found     : {stats.comments_found:,}")
        print(f"  Comments kept      : {stats.comments_kept:,}")
        print(f"  Comments filtered  : {stats.comments_filtered:,}")
        print(f"  API calls made     : {stats.api_calls}")

        if stats.filter_reasons:
            print("\n  Filter breakdown:")
            for reason, count in sorted(
                stats.filter_reasons.items(), key=lambda x: -x[1]
            ):
                print(f"    - {reason}: {count:,}")

        if self.collected:
            with_kw  = sum(1 for c in self.collected if c.has_modification_keyword)
            avg_like = sum(c.like_count for c in self.collected) / len(self.collected)
            print(
                f"\n  With modification keywords : {with_kw:,} "
                f"({100 * with_kw / len(self.collected):.1f}%)"
            )
            print(f"  Average likes/comment      : {avg_like:.1f}")

        print("=" * 60)


# =============================================================================
# API TEST
# =============================================================================

def test_api(api_key: str, config: Config) -> bool:
    """Quick 3-step API connectivity test."""
    print("\n🔧 Testing YouTube API connection...\n")
    client = YouTubeClient(api_key, config)

    # 1 – channel search
    print("1. Channel search...")
    try:
        channels = client.search_channels("מתכונים בישול", max_results=2)
        if not channels:
            print("   ✗ No channels found"); return False
        print(f"   ✓ Found {len(channels)} channels")
    except Exception as e:
        print(f"   ✗ {e}"); return False

    # 2 – video fetch
    print("2. Video retrieval...")
    try:
        videos = list(client.get_channel_videos(channels[0]['id'], max_videos=2))
        if not videos:
            print("   ✗ No videos found"); return False
        print(f"   ✓ Found {len(videos)} videos")
    except Exception as e:
        print(f"   ✗ {e}"); return False

    # 3 – comment fetch
    print("3. Comment retrieval...")
    try:
        comments = list(client.get_video_comments(videos[0], max_comments=10))
        hebrew   = [c for c in comments if HebrewUtils.contains_hebrew(c.text)]
        print(f"   ✓ {len(comments)} comments  ({len(hebrew)} Hebrew)")
        if hebrew:
            s = hebrew[0]
            print(f"\n   Sample [{s.like_count} likes | channel_id: {s.channel_id}]:")
            print(f"   {s.text[:100]}...")
    except Exception as e:
        print(f"   ✗ {e}"); return False

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED — API is working correctly")
    print("=" * 60)
    return True


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='YouTube Comment Collector for Recipe Modifications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Requires API key:
  python collect.py --test
  python collect.py --discover-all                    # Auto-find ALL channels → channels.yaml
  python collect.py --discover "מתכונים בישול"        # Single query (manual)
  python collect.py --channel UCxxxxxxx
  python collect.py --video   VIDEO_ID
  python collect.py --collect

  # No API key needed:
  python collect.py --list-channels
  python collect.py --list-channels --input data/raw_youtube/comments.jsonl
  python collect.py --filter-channels --input comments.jsonl --csv channels_report.csv
        """,
    )

    # API key
    parser.add_argument('--api-key',      type=str, help='YouTube Data API key')
    parser.add_argument('--api-key-file', type=str, help='File containing the API key')

    # Config
    parser.add_argument('--config',   type=str, default='config.yaml')
    parser.add_argument('--channels', type=str, default='channels.yaml')

    # Actions (require API key)
    parser.add_argument('--test',         action='store_true', help='Test API connection')
    parser.add_argument('--discover-all', action='store_true',
                        help='Run all discovery_queries → write all channels to channels.yaml')
    parser.add_argument('--discover',     type=str, metavar='QUERY', help='Search for channels (single query)')
    parser.add_argument('--channel',      type=str, metavar='ID',    help='Collect from one channel')
    parser.add_argument('--video',        type=str, metavar='ID',    help='Collect from one video')
    parser.add_argument('--collect',      action='store_true',       help='Collect from all active channels')

    # Actions (no API key needed)
    parser.add_argument(
        '--list-channels', action='store_true',
        help='Generate channel verification report from existing JSONL (no API key needed)',
    )
    parser.add_argument(
        '--filter-channels', action='store_true',
        help='Remove non-cooking channels from JSONL using filled CSV (no API key needed)',
    )

    # Shared file paths
    parser.add_argument(
        '--input',  type=str, default='data/raw_youtube/comments.jsonl',
        help='Input comments.jsonl (used by --list-channels and --filter-channels)',
    )
    parser.add_argument(
        '--csv',    type=str, default='data/raw_youtube/channels_report.csv',
        help='Filled CSV report (used by --filter-channels)',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path for --filter-channels (default: <input>_cooking_only.jsonl)',
    )

    args = parser.parse_args()

    # ── No-API commands ────────────────────────────────────────────────────────
    if args.list_channels:
        generate_channel_report(args.input)
        return 0

    if args.filter_channels:
        filter_channels_from_jsonl(args.input, args.csv, args.output)
        return 0

    # ── API-key required ───────────────────────────────────────────────────────
    api_key = args.api_key
    if not api_key and args.api_key_file:
        with open(args.api_key_file, 'r') as f:
            api_key = f.read().strip()
    if not api_key:
        api_key = os.environ.get('YOUTUBE_API_KEY')

    if not api_key:
        print("❌ No API key provided!")
        print("   Use: --api-key KEY  |  --api-key-file FILE  |  YOUTUBE_API_KEY env var")
        print("\nNote: --list-channels and --filter-channels work WITHOUT an API key.")
        return 1

    config = Config(args.config, args.channels)

    if args.test:
        return 0 if test_api(api_key, config) else 1

    elif args.discover_all:
        Collector(api_key, args.config, args.channels).discover_all_channels(args.channels)
        return 0

    elif args.discover:
        Collector(api_key, args.config, args.channels).discover_channels(args.discover)
        return 0

    elif args.channel:
        c = Collector(api_key, args.config, args.channels)
        c.collected = c.collect_from_channel(args.channel)
        c.save_results()
        return 0

    elif args.video:
        c = Collector(api_key, args.config, args.channels)
        c.collected = c.collect_from_video(args.video)
        c.save_results()
        return 0

    elif args.collect:
        Collector(api_key, args.config, args.channels).collect_all()
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())