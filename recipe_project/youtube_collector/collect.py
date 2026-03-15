"""
YouTube Comment Collector - Main Script
========================================
Group 11: Daniel Simanovsky & Roei Ben Artzi
NLP 2025a - Recipe Modification Extraction

This script collects Hebrew comments from YouTube cooking channels.
Configuration is read from config.yaml and channels.yaml.

Key improvements over v1:
- APPEND mode: never overwrites existing data
- Duplicate detection: skips already-collected comment IDs
- Better spam filtering: emoji-only, too-short, promotional
- Quality scoring: prioritizes comments with modification keywords
- Progress is saved after every channel (crash-safe)

Usage:
    python collect.py --test
    python collect.py --discover "מתכונים בישול"
    python collect.py --collect
    python collect.py --channel UC_CHANNEL_ID
    python collect.py --video VIDEO_ID
"""

import os
import sys
import json
import re
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Generator, Tuple

import yaml
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Comment:
    # === ESSENTIAL FOR NLP ===
    text: str
    like_count: int
    video_title: str
    channel_title: str

    # === DATA MANAGEMENT ===
    comment_id: str
    video_id: str

    # === PRE-COMPUTED ===
    word_count: int = 0
    has_modification_keyword: bool = False
    detected_keywords: List[str] = field(default_factory=list)


@dataclass
class CollectionStats:
    start_time: str = ""
    end_time: str = ""
    channels_processed: int = 0
    videos_processed: int = 0
    comments_found: int = 0
    comments_kept: int = 0
    comments_filtered: int = 0
    comments_appended: int = 0
    comments_duplicate: int = 0
    api_calls: int = 0
    errors: int = 0
    filter_reasons: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    def __init__(self, config_path: str = "config.yaml", channels_path: str = "channels.yaml"):
        self.config_path = Path(config_path)
        self.channels_path = Path(channels_path)
        self._config = {}
        self._channels = {}
        self._load()

    def _load(self):
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            print(f"⚠️  Config file not found: {self.config_path}. Using defaults.")
            self._config = self._default_config()

        if self.channels_path.exists():
            with open(self.channels_path, 'r', encoding='utf-8') as f:
                self._channels = yaml.safe_load(f) or {}

    def _default_config(self) -> dict:
        return {
            'output': {
                'directory': './data/raw_youtube',
                'comments_file': 'comments.jsonl',
                'stats_file': 'collection_stats.json',
                'log_file': './logs/collection.log',
            },
            'collection': {
                'target_comments': 30000,
                'max_videos_per_channel': 50,
                'max_comments_per_video': 200,
                'include_replies': True,
            },
            'filtering': {
                'min_words': 3,
                'require_hebrew': True,
                'skip_creator_comments': True,
                'spam_keywords': ['http', 'https', 'www.', 'bit.ly'],
            },
            'api': {
                'delay_between_videos': 0.5,
                'delay_between_channels': 2.0,
                'max_retries': 5,
                'base_retry_delay': 1.0,
                'region_code': 'IL',
            }
        }

    def get(self, *keys, default=None):
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value if value is not None else default

    def get_active_channels(self) -> List[Dict]:
        channels = self._channels.get('channels', [])
        return [ch for ch in channels if ch.get('active', False)]

    def get_all_modification_keywords(self) -> List[str]:
        keywords = []
        mod_config = self._config.get('modification_keywords', {})
        for category, words in mod_config.items():
            if isinstance(words, list):
                keywords.extend(words)
        return keywords


# =============================================================================
# HEBREW TEXT UTILITIES + QUALITY FILTERING
# =============================================================================

class HebrewUtils:
    HEBREW_PATTERN  = re.compile(r'[\u0590-\u05FF]')
    EMOJI_PATTERN   = re.compile(
        "[\U0001F300-\U0001FFFF"
        "\U00002600-\U000027BF"
        "\U0000FE00-\U0000FEFF"
        "\U00002000-\U000023FF]+",
        flags=re.UNICODE
    )
    # Patterns that indicate low-quality / irrelevant comments
    SPAM_PATTERNS = [
        re.compile(r'(פולו|עקבו|subscribe|הירשמ)', re.IGNORECASE),
        re.compile(r'(ביקרו|בקרו|בלוג|אתר)', re.IGNORECASE),
        re.compile(r'(http|https|www\.|\.co\.il|\.com|bit\.ly|tinyurl)', re.IGNORECASE),
        re.compile(r'(לחצו|לחץ|קישור|לינק)', re.IGNORECASE),
        re.compile(r'(קנו|לרכוש|מחיר|הזמינו)', re.IGNORECASE),
    ]
    # Comments that are reactions, not content — too generic for NLP
    REACTION_ONLY_PATTERNS = [
        re.compile(r'^(וואו+|וואי+|יאמי+|מממ+|אממ+|אוממ+)[!?.]*$', re.IGNORECASE),
        re.compile(r'^(תודה|תודה רבה|תודה רבה!|תנקיו|תנקס)[!?.]*$', re.IGNORECASE),
        re.compile(r'^(מעולה!*|נהדר!*|יפה!*|כל הכבוד!*|ממש טוב!*)[!?.]*$', re.IGNORECASE),
        re.compile(r'^(❤+|♥+|😍+|👏+|🔥+|✨+)[!?.]*$'),
    ]

    @classmethod
    def contains_hebrew(cls, text: str) -> bool:
        return bool(cls.HEBREW_PATTERN.search(text))

    @classmethod
    def word_count(cls, text: str) -> int:
        # Strip emojis before counting words
        clean = cls.EMOJI_PATTERN.sub('', text).strip()
        return len(clean.split())

    @classmethod
    def hebrew_ratio(cls, text: str) -> float:
        """Ratio of Hebrew characters to total non-space characters."""
        chars = [c for c in text if not c.isspace()]
        if not chars:
            return 0.0
        hebrew = [c for c in chars if cls.HEBREW_PATTERN.match(c)]
        return len(hebrew) / len(chars)

    @classmethod
    def is_spam(cls, text: str, extra_keywords: List[str] = None) -> bool:
        """Check built-in spam patterns + optional extra keywords."""
        for pattern in cls.SPAM_PATTERNS:
            if pattern.search(text):
                return True
        if extra_keywords:
            text_lower = text.lower()
            for kw in extra_keywords:
                if kw.lower() in text_lower:
                    return True
        return False

    @classmethod
    def is_reaction_only(cls, text: str) -> bool:
        """True if comment is just a generic reaction with no content."""
        clean = cls.EMOJI_PATTERN.sub('', text).strip()
        if not clean:
            return True  # emoji-only comment
        for pattern in cls.REACTION_ONLY_PATTERNS:
            if pattern.match(clean):
                return True
        return False

    @classmethod
    def is_mostly_english(cls, text: str) -> bool:
        """True if less than 20% of characters are Hebrew — likely not useful."""
        return cls.hebrew_ratio(text) < 0.2

    @classmethod
    def find_keywords(cls, text: str, keywords: List[str]) -> List[str]:
        return [kw for kw in keywords if kw in text]


# =============================================================================
# YOUTUBE API CLIENT
# =============================================================================

class YouTubeClient:
    def __init__(self, api_key: str, config: Config):
        self.api_key = api_key
        self.config = config
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.stats = CollectionStats(start_time=datetime.now().isoformat())
        self.seen_ids: set = set()
        self._setup_logging()

    def _setup_logging(self):
        log_file = self.config.get('output', 'log_file')
        handlers = [logging.StreamHandler()]
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path, encoding='utf-8'))
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=handlers,
            force=True,
        )
        self.logger = logging.getLogger(__name__)

    def _api_call(self, request, description: str = "API call"):
        max_retries = self.config.get('api', 'max_retries', default=5)
        base_delay  = self.config.get('api', 'base_retry_delay', default=1.0)

        for attempt in range(max_retries):
            try:
                self.stats.api_calls += 1
                return request.execute()
            except HttpError as e:
                if e.resp.status == 403 and 'quotaExceeded' in str(e):
                    self.logger.error("❌ API quota exceeded! Wait until tomorrow.")
                    raise
                elif e.resp.status == 429:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"⏳ Rate limited. Waiting {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    self.stats.errors += 1
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(base_delay)

        raise Exception(f"Max retries exceeded for {description}")

    # -------------------------------------------------------------------------
    # DISCOVERY
    # -------------------------------------------------------------------------

    def search_channels(self, query: str, max_results: int = 10) -> List[Dict]:
        self.logger.info(f"🔍 Searching channels: '{query}'")
        region = self.config.get('api', 'region_code', default='IL')
        request = self.youtube.search().list(
            part='snippet', q=query, type='channel',
            maxResults=max_results, regionCode=region
        )
        response = self._api_call(request, "channel search")

        channels = []
        for item in response.get('items', []):
            channel_id = item['snippet']['channelId']
            info_response = self._api_call(
                self.youtube.channels().list(part='snippet,statistics', id=channel_id),
                "channel info"
            )
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

    def get_channel_videos(self, channel_id: str, max_videos: int = 50) -> Generator[Dict, None, None]:
        next_page = None
        retrieved = 0
        while retrieved < max_videos:
            request = self.youtube.search().list(
                part='snippet', channelId=channel_id, type='video',
                order='viewCount',   # most-viewed first = more comments
                maxResults=min(50, max_videos - retrieved),
                pageToken=next_page
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

    def get_video_comments(self, video: Dict, max_comments: int = 200) -> Generator[Comment, None, None]:
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
                    order='relevance',   # top comments first — higher quality
                    textFormat='plainText'
                )
                response = self._api_call(request, f"comments for {video['id']}")

                for item in response.get('items', []):
                    comment = self._parse_comment(item, video)
                    if comment:
                        yield comment
                        retrieved += 1

                    if include_replies and 'replies' in item:
                        for reply_item in item['replies']['comments']:
                            reply = self._parse_reply(reply_item, video)
                            if reply:
                                yield reply

                    if retrieved >= max_comments:
                        break

                next_page = response.get('nextPageToken')
                if not next_page:
                    break

            except HttpError as e:
                if 'commentsDisabled' in str(e):
                    self.logger.info(f"   Comments disabled: {video['title'][:40]}")
                else:
                    self.logger.warning(f"   Comment error: {e}")
                break

    def _parse_comment(self, item: Dict, video: Dict) -> Optional[Comment]:
        snippet    = item['snippet']['topLevelComment']['snippet']
        comment_id = item['snippet']['topLevelComment']['id']

        if comment_id in self.seen_ids:
            return None
        self.seen_ids.add(comment_id)

        # Skip creator comments
        if self.config.get('filtering', 'skip_creator_comments', default=True):
            author_channel = snippet.get('authorChannelId', {}).get('value', '')
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
            word_count=HebrewUtils.word_count(text),
        )

    def _parse_reply(self, reply_item: Dict, video: Dict) -> Optional[Comment]:
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
            word_count=HebrewUtils.word_count(text),
        )

    # -------------------------------------------------------------------------
    # FILTERING
    # -------------------------------------------------------------------------

    def filter_comment(self, comment: Comment) -> Tuple[bool, str]:
        """
        Multi-layer quality filter.
        Returns (keep, reason_if_rejected).
        """
        text = comment.text

        # 1. Hebrew requirement
        if self.config.get('filtering', 'require_hebrew', default=True):
            if not HebrewUtils.contains_hebrew(text):
                return False, "no_hebrew"

        # 2. Mostly English (< 20% Hebrew chars) — not useful for Hebrew NLP
        if HebrewUtils.is_mostly_english(text):
            return False, "mostly_english"

        # 3. Emoji-only or reaction-only
        if HebrewUtils.is_reaction_only(text):
            return False, "reaction_only"

        # 4. Minimum word count (after stripping emojis)
        min_words = self.config.get('filtering', 'min_words', default=3)
        if comment.word_count < min_words:
            return False, "too_short"

        # 5. Spam
        extra_spam = self.config.get('filtering', 'spam_keywords', default=[])
        if HebrewUtils.is_spam(text, extra_spam):
            return False, "spam"

        return True, ""

    def enrich_comment(self, comment: Comment) -> Comment:
        keywords = self.config.get_all_modification_keywords()
        found = HebrewUtils.find_keywords(comment.text, keywords)
        comment.has_modification_keyword = len(found) > 0
        comment.detected_keywords = found
        return comment


# =============================================================================
# MAIN COLLECTOR
# =============================================================================

class Collector:
    def __init__(self, api_key: str, config_path: str = "config.yaml", channels_path: str = "channels.yaml"):
        self.config     = Config(config_path, channels_path)
        self.client     = YouTubeClient(api_key, self.config)
        self.collected: List[Comment] = []

        output_dir = self.config.get('output', 'directory', default='./data/raw_youtube')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing comment IDs for deduplication (APPEND mode support)
        self.existing_ids = self._load_existing_ids()
        print(f"📂 Existing comments in file: {len(self.existing_ids):,} (will skip duplicates)")

    def _load_existing_ids(self) -> set:
        """Load all comment IDs already saved to disk."""
        comments_file = self.config.get('output', 'comments_file', default='comments.jsonl')
        comments_path = self.output_dir / comments_file
        ids = set()
        if comments_path.exists():
            with open(comments_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        ids.add(json.loads(line.strip())['comment_id'])
                    except Exception:
                        pass
        return ids

    # -------------------------------------------------------------------------
    # DISCOVERY
    # -------------------------------------------------------------------------

    def discover_channels(self, query: str):
        print(f"\n🔍 Searching for channels: '{query}'\n")
        print("=" * 70)
        channels = self.client.search_channels(query, max_results=10)

        if not channels:
            print("No channels found.")
            return

        print(f"Found {len(channels)} channels:\n")
        for i, ch in enumerate(channels, 1):
            print(f"{i}. {ch['name']}")
            print(f"   ID: {ch['id']}")
            print(f"   Subscribers: {ch['subscribers']:,} | Videos: {ch['videos']}")
            print(f"   Description: {ch['description'][:80]}...")
            print()

        print("=" * 70)
        print("\nTo add a channel, copy its ID to channels.yaml:")
        print("""
  - name: "Channel Name"
    id: "UC_PASTE_ID_HERE"
    active: true
    category: "cooking"
""")

    # -------------------------------------------------------------------------
    # COLLECTION
    # -------------------------------------------------------------------------

    def collect_from_channel(self, channel_id: str, channel_name: str = "Unknown") -> List[Comment]:
        print(f"\n📺 Collecting from: {channel_name}  [{channel_id}]")

        max_videos  = self.config.get('collection', 'max_videos_per_channel', default=50)
        max_comments = self.config.get('collection', 'max_comments_per_video', default=200)
        delay       = self.config.get('api', 'delay_between_videos', default=0.5)

        channel_comments = []
        videos_processed = 0

        for video in self.client.get_channel_videos(channel_id, max_videos):
            videos_processed += 1
            self.client.stats.videos_processed += 1
            print(f"   [{videos_processed}/{max_videos}] {video['title'][:55]}...", end=' ')

            video_kept = 0
            for comment in self.client.get_video_comments(video, max_comments):
                self.client.stats.comments_found += 1

                # Skip already collected
                if comment.comment_id in self.existing_ids:
                    self.client.stats.comments_duplicate += 1
                    continue

                keep, reason = self.client.filter_comment(comment)
                if keep:
                    comment = self.client.enrich_comment(comment)
                    channel_comments.append(comment)
                    self.client.stats.comments_kept += 1
                    video_kept += 1
                else:
                    self.client.stats.comments_filtered += 1
                    self.client.stats.filter_reasons[reason] = \
                        self.client.stats.filter_reasons.get(reason, 0) + 1

            print(f"→ {video_kept} kept")
            time.sleep(delay)

        self.client.stats.channels_processed += 1
        print(f"   ✓ Channel total: {len(channel_comments)} new comments")
        return channel_comments

    def collect_from_video(self, video_id: str) -> List[Comment]:
        print(f"\n🎬 Collecting from video: {video_id}")

        response = self.client._api_call(
            self.client.youtube.videos().list(part='snippet', id=video_id),
            "video info"
        )
        if not response.get('items'):
            print("❌ Video not found")
            return []

        info  = response['items'][0]['snippet']
        video = {
            'id':            video_id,
            'title':         info['title'],
            'channel_id':    info['channelId'],
            'channel_title': info['channelTitle'],
        }
        print(f"   Title:   {video['title'][:60]}")
        print(f"   Channel: {video['channel_title']}")

        max_comments = self.config.get('collection', 'max_comments_per_video', default=200)
        video_comments = []

        for comment in self.client.get_video_comments(video, max_comments):
            self.client.stats.comments_found += 1
            if comment.comment_id in self.existing_ids:
                self.client.stats.comments_duplicate += 1
                continue
            keep, reason = self.client.filter_comment(comment)
            if keep:
                comment = self.client.enrich_comment(comment)
                video_comments.append(comment)
                self.client.stats.comments_kept += 1
            else:
                self.client.stats.comments_filtered += 1

        print(f"   ✓ Collected: {len(video_comments)} new comments")
        return video_comments

    def collect_all(self):
        channels = self.config.get_active_channels()
        if not channels:
            print("\n⚠️  No active channels configured!")
            return

        target = self.config.get('collection', 'target_comments', default=30000)
        delay  = self.config.get('api', 'delay_between_channels', default=2.0)

        # Count already on disk
        already_on_disk = len(self.existing_ids)
        print(f"\n🚀 Starting collection from {len(channels)} channels")
        print(f"   Already on disk: {already_on_disk:,}")
        print(f"   Target total:    {target:,}")
        print(f"   Still needed:    {max(0, target - already_on_disk):,}\n")

        for channel in channels:
            # Check if we've already hit target (counting disk + newly collected)
            total_so_far = already_on_disk + len(self.collected)
            if total_so_far >= target:
                print(f"\n✅ Target of {target:,} reached!")
                break

            comments = self.collect_from_channel(
                channel_id=channel['id'],
                channel_name=channel.get('name', 'Unknown')
            )
            self.collected.extend(comments)

            # Save after EVERY channel (crash-safe)
            self._append_to_disk(comments)

            total_so_far = already_on_disk + len(self.collected)
            print(f"\n   📊 Total on disk: {total_so_far:,} / {target:,}")
            time.sleep(delay)

        self._save_stats()
        self.print_summary()

    # -------------------------------------------------------------------------
    # SAVING — APPEND MODE
    # -------------------------------------------------------------------------

    def _append_to_disk(self, comments: List[Comment]):
        """Append new comments to disk immediately. Never overwrites."""
        if not comments:
            return

        comments_file = self.config.get('output', 'comments_file', default='comments.jsonl')
        comments_path = self.output_dir / comments_file

        appended = 0
        with open(comments_path, 'a', encoding='utf-8') as f:
            for comment in comments:
                if comment.comment_id not in self.existing_ids:
                    f.write(json.dumps(asdict(comment), ensure_ascii=False) + '\n')
                    self.existing_ids.add(comment.comment_id)
                    appended += 1

        self.client.stats.comments_appended += appended

    def save_results(self):
        """Save collected comments (used by --video and --channel modes)."""
        if not self.collected:
            print("\n⚠️  No comments to save")
            return
        self._append_to_disk(self.collected)
        self._save_stats()
        self.print_summary()

    def _save_stats(self):
        self.client.stats.end_time = datetime.now().isoformat()
        stats_file = self.config.get('output', 'stats_file', default='collection_stats.json')
        stats_path = self.output_dir / stats_file
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.client.stats), f, indent=2, ensure_ascii=False)

    def print_summary(self):
        stats = self.client.stats
        total_on_disk = len(self.existing_ids)

        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        print(f"  Channels processed:   {stats.channels_processed}")
        print(f"  Videos processed:     {stats.videos_processed}")
        print(f"  Comments found:       {stats.comments_found}")
        print(f"  Comments kept:        {stats.comments_kept}")
        print(f"  Comments filtered:    {stats.comments_filtered}")
        print(f"  Duplicates skipped:   {stats.comments_duplicate}")
        print(f"  Appended to file:     {stats.comments_appended}")
        print(f"  Total on disk:        {total_on_disk:,}")
        print(f"  API calls made:       {stats.api_calls}")

        if stats.filter_reasons:
            print("\n  Filter breakdown:")
            for reason, count in sorted(stats.filter_reasons.items(), key=lambda x: -x[1]):
                print(f"    - {reason}: {count}")

        if self.collected:
            with_kw  = sum(1 for c in self.collected if c.has_modification_keyword)
            avg_like = sum(c.like_count for c in self.collected) / len(self.collected)
            pct      = 100 * with_kw / len(self.collected)
            print(f"\n  With modification keywords: {with_kw} ({pct:.1f}%)")
            print(f"  Average likes per comment:  {avg_like:.1f}")

        print("=" * 60)


# =============================================================================
# API TEST
# =============================================================================

def test_api(api_key: str, config: Config):
    print("\n" + "=" * 60)
    print("API CONNECTION TEST")
    print("=" * 60)

    client = YouTubeClient(api_key, config)

    print("\n1. Testing channel search...")
    channels = client.search_channels("מתכונים", max_results=2)
    if not channels:
        print("   ✗ No channels found"); return False
    print(f"   ✓ Found {len(channels)} channels")
    for ch in channels:
        print(f"     - {ch['name']} ({ch['videos']} videos)")

    print("\n2. Testing video retrieval...")
    videos = list(client.get_channel_videos(channels[0]['id'], max_videos=2))
    if not videos:
        print("   ✗ No videos found"); return False
    print(f"   ✓ Found {len(videos)} videos")
    for v in videos:
        print(f"     - {v['title'][:50]}...")

    print("\n3. Testing comment retrieval...")
    comments = list(client.get_video_comments(videos[0], max_comments=10))
    print(f"   ✓ Found {len(comments)} comments")
    hebrew = [c for c in comments if HebrewUtils.contains_hebrew(c.text)]
    print(f"   ✓ Hebrew comments: {len(hebrew)}/{len(comments)}")
    if hebrew:
        s = hebrew[0]
        print(f"\n   Sample — Likes: {s.like_count} | Words: {s.word_count}")
        print(f"   Text: {s.text[:100]}...")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='YouTube Comment Collector for Recipe Modifications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect.py --test
  python collect.py --discover "מתכונים בישול"
  python collect.py --channel UCxxxxxxx
  python collect.py --video VIDEO_ID
  python collect.py --collect
        """
    )
    parser.add_argument('--api-key',      type=str)
    parser.add_argument('--api-key-file', type=str)
    parser.add_argument('--config',       type=str, default='config.yaml')
    parser.add_argument('--channels',     type=str, default='channels.yaml')
    parser.add_argument('--test',         action='store_true')
    parser.add_argument('--discover',     type=str, metavar='QUERY')
    parser.add_argument('--channel',      type=str, metavar='ID')
    parser.add_argument('--video',        type=str, metavar='ID')
    parser.add_argument('--collect',      action='store_true')
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key and args.api_key_file:
        with open(args.api_key_file) as f:
            api_key = f.read().strip()
    if not api_key:
        api_key = os.environ.get('YOUTUBE_API_KEY')

    if not api_key:
        print("❌ No API key!\n  --api-key YOUR_KEY\n  YOUTUBE_API_KEY env var")
        return 1

    config = Config(args.config, args.channels)

    if args.test:
        return 0 if test_api(api_key, config) else 1

    elif args.discover:
        Collector(api_key, args.config, args.channels).discover_channels(args.discover)

    elif args.channel:
        c = Collector(api_key, args.config, args.channels)
        c.collected = c.collect_from_channel(args.channel)
        c.save_results()

    elif args.video:
        c = Collector(api_key, args.config, args.channels)
        c.collected = c.collect_from_video(args.video)
        c.save_results()

    elif args.collect:
        Collector(api_key, args.config, args.channels).collect_all()

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())