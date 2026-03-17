"""
YouTube Comment Collector v3 — Thread-Aware
=============================================
Group 11: Daniel Simanovsky & Roei Ben Artzi
NLP 2025a - Recipe Modification Extraction

MAJOR CHANGE IN v3:
  Comments are now collected as THREADS (top comment + all replies together),
  not as flat individual records. This preserves the question→answer relationship
  which is critical for correct teacher labeling.

  A thread = one training example.
  Target: 30,000 threads ≈ 85,000 individual comments.

KEY DESIGN RULE:
  A question comment ("אפשר במקום X?") is NEVER the signal.
  The REPLY to that question is the signal.
  The question is context. Store threads together, label replies.

BUG FIXES IN v3:
  - channel_id now stored in every thread (was missing in v1/v2)
  - Creator comments flagged (is_creator=True) instead of silently dropped
  - Thread structure replaces flat comment list
  - source field added for future multi-source support
  - appearance_count for deduplication

Usage:
    python collect.py --test
    python collect.py --discover-all
    python collect.py --collect
    python collect.py --channel UC_CHANNEL_ID
    python collect.py --video VIDEO_ID
    python collect.py --list-channels   # no API key needed
    python collect.py --filter-channels --input threads.jsonl --csv channels_report.csv
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
from typing import List, Dict, Optional, Generator, Tuple, Any
from collections import defaultdict

import yaml
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ReplyRecord:
    """A single reply within a thread."""
    comment_id: str
    text: str
    like_count: int = 0
    is_creator: bool = False       # True if this reply is from the video creator


@dataclass
class TopComment:
    """The top-level comment that starts a thread."""
    comment_id: str
    text: str
    like_count: int = 0


@dataclass
class Thread:
    """
    A full comment thread: one top-level comment + all its replies.

    This is the primary unit of collection and the primary training example.

    Design rationale:
      - A question comment is meaningless without its replies
      - A creator answer is the strongest modification signal
      - Storing threads preserves this context for teacher labeling

    JSONL format (one line per thread):
    {
      "thread_id":        "UgwXXX",
      "video_id":         "abc123",
      "video_title":      "לחם כוסמין",
      "channel_title":    "חן במטבח",
      "channel_id":       "UCc0z...",         ← was missing in v1/v2
      "source":           "youtube",
      "appearance_count": 1,
      "top_comment":      {"comment_id": ..., "text": ..., "like_count": ...},
      "replies":          [{"comment_id": ..., "text": ..., "like_count": ...,
                            "is_creator": true/false}, ...],
      "has_creator_reply": false,
      "total_likes":      4
    }
    """
    # === IDENTIFIERS ===
    thread_id:   str              # = top comment's comment_id
    video_id:    str
    channel_id:  str              # BUG FIX: was missing in v1/v2
    
    # === CONTEXT ===
    video_title:   str
    channel_title: str
    source:        str = "youtube"

    # === CONTENT ===
    top_comment:      TopComment = None
    replies:          List[ReplyRecord] = field(default_factory=list)
    
    # === COMPUTED ===
    has_creator_reply: bool = False
    total_likes:       int  = 0
    appearance_count:  int  = 1   # for deduplication — how many times exact text seen


@dataclass
class CollectionStats:
    start_time:          str = ""
    end_time:            str = ""
    channels_processed:  int = 0
    videos_processed:    int = 0
    threads_found:       int = 0
    threads_kept:        int = 0
    threads_filtered:    int = 0
    api_calls:           int = 0
    errors:              int = 0
    filter_reasons:      Dict[str, int] = field(default_factory=dict)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    def __init__(self, config_path: str = "config.yaml", channels_path: str = "channels.yaml"):
        cfg_file = Path(config_path)
        self._config = yaml.safe_load(cfg_file.read_text(encoding='utf-8')) or {} \
            if cfg_file.exists() else self._defaults()

        ch_file = Path(channels_path)
        self._channels = yaml.safe_load(ch_file.read_text(encoding='utf-8')) or {} \
            if ch_file.exists() else {}

    def _defaults(self) -> dict:
        return {
            'output':     {'directory': './data/raw_youtube', 'threads_file': 'threads.jsonl',
                           'stats_file': 'collection_stats.json', 'log_file': None},
            'collection': {'target_threads': 30000, 'max_videos_per_channel': 100,
                           'max_threads_per_video': 200, 'include_replies': True},
            'filtering':  {'min_words': 5, 'require_hebrew': True,
                           'skip_creator_top_comments': True, 'spam_keywords': ['http','https','www.']},
            'api':        {'delay_between_videos': 0.5, 'delay_between_channels': 2.0,
                           'max_retries': 3, 'retry_delay': 1.0, 'region_code': 'IL'},
        }

    def get(self, *keys, default=None):
        val = self._config
        for k in keys:
            val = val.get(k, default) if isinstance(val, dict) else default
        return val if val is not None else default

    def get_active_channels(self) -> List[Dict]:
        return [c for c in self._channels.get('channels', []) if c.get('active', False)]

    def get_discovery_queries(self) -> List[str]:
        return self._channels.get('discovery_queries', [
            "מתכונים בישול", "אפייה ביתית", "בישול ישראלי",
            "עוגות ומאפים", "מתכונים קלים", "שף ישראלי",
            "בישול בריא", "קינוחים", "ארוחות משפחתיות", "מתכונים לשבת",
        ])

    def get_modification_keywords(self) -> List[str]:
        kws = []
        for cat, words in self._config.get('modification_keywords', {}).items():
            if isinstance(words, list):
                kws.extend(words)
        return kws


# =============================================================================
# HEBREW UTILITIES
# =============================================================================

class HebrewUtils:
    HEBREW = re.compile(r'[\u0590-\u05FF]')

    @classmethod
    def contains_hebrew(cls, text: str) -> bool:
        return bool(cls.HEBREW.search(text))

    @classmethod
    def word_count(cls, text: str) -> int:
        return len(text.strip().split())

    @classmethod
    def is_spam(cls, text: str, keywords: List[str]) -> bool:
        t = text.lower()
        return any(k.lower() in t for k in keywords)

    @classmethod
    def find_keywords(cls, text: str, keywords: List[str]) -> List[str]:
        return [k for k in keywords if k in text]

    @classmethod
    def is_question(cls, text: str) -> bool:
        """Detect if text is a question (not a modification statement)."""
        stripped = text.strip()
        if stripped.endswith('?'):
            return True
        question_words = ['האם', 'אפשר', 'כדאי', 'אפשרי', 'מה אם', 'מה לגבי']
        return any(stripped.startswith(qw) or f' {qw} ' in stripped for qw in question_words)

    @classmethod
    def is_unhelpful_reply(cls, text: str) -> bool:
        """Detect replies that don't answer the question (e.g., 'me too', emojis only)."""
        stripped = text.strip()
        if not stripped:
            return True
        # Only emojis / very short
        no_hebrew = cls.HEBREW.sub('', stripped)
        if len(stripped) < 4 and not cls.contains_hebrew(stripped):
            return True
        unhelpful = ['גם אני', 'גם אני רוצה', 'תשאל', 'לא יודע', 'מעניין', 'וואו']
        return any(u in stripped for u in unhelpful)


# =============================================================================
# YOUTUBE API CLIENT
# =============================================================================

class YouTubeClient:
    def __init__(self, api_key: str, config: Config):
        self.config  = config
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.stats   = CollectionStats(start_time=datetime.now().isoformat())
        self.seen_thread_ids: set = set()
        self._setup_logging()

    def _setup_logging(self):
        handlers = [logging.StreamHandler()]
        log_file = self.config.get('output', 'log_file')
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
        logging.basicConfig(level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s', handlers=handlers)
        self.logger = logging.getLogger(__name__)

    def _api_call(self, request, desc: str = "") -> Dict:
        retries   = self.config.get('api', 'max_retries', default=3)
        base_delay = self.config.get('api', 'retry_delay', default=1.0)
        for attempt in range(retries):
            try:
                self.stats.api_calls += 1
                return request.execute()
            except HttpError as e:
                if e.resp.status in (403, 429) and any(
                        x in str(e) for x in ('quotaExceeded', 'rateLimitExceeded', 'forbidden')):
                    self.logger.error("❌ API quota exceeded. Stop for today.")
                    raise
                if attempt == retries - 1:
                    self.stats.errors += 1
                    raise
                delay = base_delay * (2 ** attempt)
                self.logger.warning(f"API error ({desc}): {e}. Retry in {delay:.1f}s")
                time.sleep(delay)

    # ─────────────────────────────────────────────────────────────────────────
    # CHANNEL / VIDEO DISCOVERY
    # ─────────────────────────────────────────────────────────────────────────

    def search_channels(self, query: str, max_results: int = 10) -> List[Dict]:
        region = self.config.get('api', 'region_code', default='IL')
        resp = self._api_call(
            self.youtube.search().list(part='snippet', q=query, type='channel',
                maxResults=max_results, regionCode=region),
            "channel search")
        channels = []
        for item in resp.get('items', []):
            cid = item['snippet']['channelId']
            info = self._api_call(
                self.youtube.channels().list(part='snippet,statistics', id=cid),
                "channel info")
            if info.get('items'):
                d = info['items'][0]
                channels.append({
                    'id':          cid,
                    'name':        d['snippet']['title'],
                    'description': d['snippet'].get('description', '')[:200],
                    'subscribers': int(d['statistics'].get('subscriberCount', 0)),
                    'videos':      int(d['statistics'].get('videoCount', 0)),
                })
        return channels

    def get_channel_videos(self, channel_id: str, max_videos: int = 100) -> Generator[Dict, None, None]:
        next_page, retrieved = None, 0
        while retrieved < max_videos:
            resp = self._api_call(
                self.youtube.search().list(
                    part='snippet', channelId=channel_id, type='video',
                    order='date', maxResults=min(50, max_videos - retrieved),
                    pageToken=next_page),
                f"videos for {channel_id}")
            for item in resp.get('items', []):
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
            next_page = resp.get('nextPageToken')
            if not next_page:
                break

    # ─────────────────────────────────────────────────────────────────────────
    # THREAD COLLECTION  (core new logic)
    # ─────────────────────────────────────────────────────────────────────────

    def get_video_threads(
        self, video: Dict, max_threads: int = 200
    ) -> Generator[Thread, None, None]:
        """
        Yield Thread objects from a video.

        Each thread contains the top-level comment + ALL its replies.
        Replies include a flag for whether the commenter is the video creator.

        This replaces the old get_video_comments() which lost reply context.
        """
        include_replies = self.config.get('collection', 'include_replies', default=True)
        skip_creator_top = self.config.get('filtering', 'skip_creator_top_comments', default=True)
        next_page = None
        retrieved = 0

        while retrieved < max_threads:
            try:
                request = self.youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video['id'],
                    maxResults=min(100, max_threads - retrieved),
                    pageToken=next_page,
                    textFormat='plainText',
                )
                response = self._api_call(request, f"threads for {video['id']}")

                for item in response.get('items', []):
                    thread = self._build_thread(item, video,
                                                skip_creator_top, include_replies)
                    if thread is not None:
                        yield thread
                        retrieved += 1

                    if retrieved >= max_threads:
                        break

                next_page = response.get('nextPageToken')
                if not next_page:
                    break

            except HttpError as e:
                if 'commentsDisabled' in str(e):
                    self.logger.info(f"   Comments disabled: {video['title'][:40]}...")
                else:
                    self.logger.warning(f"   Thread fetch error: {e}")
                break

    def _build_thread(
        self,
        item: Dict,
        video: Dict,
        skip_creator_top: bool,
        include_replies: bool,
    ) -> Optional['Thread']:
        """
        Build a Thread from a commentThread API item.

        BUG FIX (v3): channel_id is now stored.
        BUG FIX (v3): Creator comments are flagged (is_creator=True) instead of dropped.
        NEW (v3):     All replies included with is_creator flag.
        """
        top_snippet  = item['snippet']['topLevelComment']['snippet']
        top_id       = item['snippet']['topLevelComment']['id']

        # Skip duplicate threads
        if top_id in self.seen_thread_ids:
            return None
        self.seen_thread_ids.add(top_id)

        creator_channel = video['channel_id']
        top_author_ch   = top_snippet.get('authorChannelId', {}).get('value', '')
        top_is_creator  = (top_author_ch == creator_channel)

        # Skip creator's own top-level comments (usually pinned recipe intro)
        # BUT: we still collect them as replies if they answer user questions
        if skip_creator_top and top_is_creator:
            return None

        top_text = top_snippet['textDisplay']

        top_comment = TopComment(
            comment_id=top_id,
            text=top_text,
            like_count=top_snippet.get('likeCount', 0),
        )

        # Build replies list
        replies: List[ReplyRecord] = []
        has_creator_reply = False
        total_likes = top_comment.like_count

        if include_replies and 'replies' in item:
            for reply_item in item['replies']['comments']:
                r_snippet    = reply_item['snippet']
                r_id         = reply_item['id']
                r_author_ch  = r_snippet.get('authorChannelId', {}).get('value', '')
                r_is_creator = (r_author_ch == creator_channel)

                reply = ReplyRecord(
                    comment_id=r_id,
                    text=r_snippet['textDisplay'],
                    like_count=r_snippet.get('likeCount', 0),
                    is_creator=r_is_creator,
                )
                replies.append(reply)
                total_likes += reply.like_count
                if r_is_creator:
                    has_creator_reply = True

        return Thread(
            thread_id=top_id,
            video_id=video['id'],
            channel_id=video['channel_id'],      # BUG FIX: was missing in v1/v2
            video_title=video['title'],
            channel_title=video['channel_title'],
            source="youtube",
            top_comment=top_comment,
            replies=replies,
            has_creator_reply=has_creator_reply,
            total_likes=total_likes,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # FILTERING
    # ─────────────────────────────────────────────────────────────────────────

    def filter_thread(self, thread: Thread) -> Tuple[bool, str]:
        """
        Decide whether to keep a thread.

        Filtering logic:
          1. Top comment must have Hebrew (or at least one reply must)
          2. Minimum word count on top comment
          3. No spam
          4. If top comment is a question and has NO meaningful replies → discard
             (A question with no answer has zero signal value)
        """
        top_text = thread.top_comment.text
        all_texts = [top_text] + [r.text for r in thread.replies]

        # Hebrew check — at least one text in the thread must be Hebrew
        require_hebrew = self.config.get('filtering', 'require_hebrew', default=True)
        if require_hebrew:
            if not any(HebrewUtils.contains_hebrew(t) for t in all_texts):
                return False, "no_hebrew"

        # Min word count on top comment
        min_words = self.config.get('filtering', 'min_words', default=5)
        if HebrewUtils.word_count(top_text) < min_words:
            return False, "too_short"

        # Spam check
        spam_kws = self.config.get('filtering', 'spam_keywords', default=[])
        if HebrewUtils.is_spam(top_text, spam_kws):
            return False, "spam"

        # Question with no meaningful replies = no signal
        if HebrewUtils.is_question(top_text):
            meaningful = [r for r in thread.replies
                         if not HebrewUtils.is_unhelpful_reply(r.text)
                         and HebrewUtils.contains_hebrew(r.text)]
            if not meaningful:
                return False, "question_unanswered"

        return True, ""

    def enrich_thread(self, thread: Thread) -> Thread:
        """Pre-compute modification keyword detection on the thread text."""
        kws = self.config.get_modification_keywords()
        all_text = thread.top_comment.text + " " + " ".join(r.text for r in thread.replies)
        # We don't store keywords on the thread — teacher labeling does the real work
        # This is just a fast pre-filter signal
        return thread


# =============================================================================
# CHANNELS.YAML WRITER
# =============================================================================

def write_channels_yaml(channels: List[Dict], channels_path: str, existing_data: Dict) -> None:
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
    entries = []
    for ch in channels:
        entries.append({
            'name':        ch['name'],
            'id':          ch['id'],
            'active':      True,
            'category':    'cooking',
            'subscribers': ch.get('subscribers', 0),
            'youtube_url': f"https://www.youtube.com/channel/{ch['id']}",
            'notes':       ch.get('description', '')[:100].replace('\n', ' ').strip(),
        })
    out = dict(existing_data)
    out['channels'] = entries
    yaml_str = yaml.dump(out, allow_unicode=True, default_flow_style=False,
                         sort_keys=False, width=120)
    with open(channels_path, 'w', encoding='utf-8') as f:
        f.write(header + yaml_str)
    print(f"💾 Written to: {channels_path}")


# =============================================================================
# CHANNEL REPORT  (no API key needed)
# =============================================================================

def generate_channel_report(threads_file: str, output_dir: str = None) -> None:
    """
    Read existing threads.jsonl → produce channels_report.txt + channels_report.csv
    for manual cooking/not-cooking review. No API key needed.
    """
    input_path = Path(threads_file)
    if not input_path.exists():
        print(f"❌ File not found: {threads_file}")
        print("   Run --collect first.")
        return

    output_dir = Path(output_dir or input_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 Reading: {threads_file}")

    channel_data: Dict[str, Dict] = defaultdict(lambda: {
        'channel_id': '', 'video_ids': [], 'video_titles': {}, 'thread_count': 0
    })
    total = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = t.get('channel_title', 'Unknown')
            ch   = channel_data[name]
            if t.get('channel_id') and not ch['channel_id']:
                ch['channel_id'] = t['channel_id']
            vid_id = t.get('video_id', '')
            if vid_id and vid_id not in ch['video_titles']:
                ch['video_ids'].append(vid_id)
                ch['video_titles'][vid_id] = t.get('video_title', '')
            ch['thread_count'] += 1
            total += 1

    if not channel_data:
        print("⚠️  No threads found.")
        return

    sorted_ch = sorted(channel_data.items(), key=lambda x: -x[1]['thread_count'])

    # TXT
    txt_path = output_dir / "channels_report.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CHANNEL VERIFICATION REPORT\n")
        f.write(f"Source : {threads_file}\n")
        f.write(f"Total  : {total:,} threads  |  {len(sorted_ch)} channels\n")
        f.write("=" * 80 + "\n\n")
        f.write("INSTRUCTIONS: Mark [ ] as [YES] (cooking) or [NO] (not cooking).\n")
        f.write("Then run: python collect.py --filter-channels --input threads.jsonl --csv channels_report.csv\n\n")
        f.write("-" * 80 + "\n\n")
        for i, (name, d) in enumerate(sorted_ch, 1):
            ch_id = d['channel_id']
            f.write(f"  #{i:02d}  [ ] {name}\n")
            f.write(f"        Threads : {d['thread_count']:,}  |  Videos: {len(d['video_ids'])}\n")
            if ch_id:
                f.write(f"        Channel : https://www.youtube.com/channel/{ch_id}\n")
            for vid_id in d['video_ids'][:3]:
                title = d['video_titles'].get(vid_id, '')
                f.write(f"          • {title[:55]:<55}  https://youtube.com/watch?v={vid_id}\n")
            f.write("\n")

    # CSV
    csv_path = output_dir / "channels_report.csv"
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank','channel_name','channel_id','channel_url','thread_count',
                         'video_count','v1_title','v1_url','v2_title','v2_url','v3_title','v3_url',
                         'is_cooking_channel','notes'])
        for i, (name, d) in enumerate(sorted_ch, 1):
            ch_id = d['channel_id']
            ch_url = f"https://www.youtube.com/channel/{ch_id}" if ch_id else ""
            row = [i, name, ch_id, ch_url, d['thread_count'], len(d['video_ids'])]
            for j in range(3):
                if j < len(d['video_ids']):
                    vid = d['video_ids'][j]
                    row += [d['video_titles'].get(vid,''), f"https://youtube.com/watch?v={vid}"]
                else:
                    row += ['', '']
            row += ['', '']
            writer.writerow(row)

    print(f"📄 TXT report → {txt_path}")
    print(f"📊 CSV report → {csv_path}")

    print(f"\n{'='*72}")
    print(f"  {'#':>3}  {'Channel Name':<40}  {'Threads':>9}")
    print(f"  {'-'*3}  {'-'*40}  {'-'*9}")
    for i, (name, d) in enumerate(sorted_ch, 1):
        print(f"  {i:>3}  {name:<40}  {d['thread_count']:>9,}")
    print(f"{'='*72}\n")


# =============================================================================
# CHANNEL FILTER  (no API key needed)
# =============================================================================

def filter_channels_from_jsonl(
    threads_file: str, csv_report: str, output_file: Optional[str] = None
) -> None:
    """Remove non-cooking channel threads after manual CSV review."""
    csv_path = Path(csv_report)
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_report}")
        return

    approved, rejected = set(), set()
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name    = row.get('channel_name', '').strip()
            verdict = row.get('is_cooking_channel', '').strip().upper()
            if verdict in ('YES','Y','1','TRUE','COOKING'):
                approved.add(name)
            elif verdict in ('NO','N','0','FALSE','NOT-COOKING','NOT COOKING'):
                rejected.add(name)
            else:
                approved.add(name)   # conservative default: keep if unsure

    print(f"\n📋 Approved: {len(approved)}  |  Rejected: {len(rejected)}")
    if rejected:
        print("   Removing:")
        for n in sorted(rejected):
            print(f"     ✗ {n}")

    input_path = Path(threads_file)
    output_path = Path(output_file) if output_file else \
        input_path.parent / (input_path.stem + "_cooking_only.jsonl")

    kept = removed = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                if t.get('channel_title', '') not in rejected:
                    fout.write(line + '\n')
                    kept += 1
                else:
                    removed += 1
            except json.JSONDecodeError:
                continue

    print(f"\n✅ Kept: {kept:,}  |  Removed: {removed:,}")
    print(f"   Output: {output_path}")


# =============================================================================
# MAIN COLLECTOR CLASS
# =============================================================================

class Collector:
    def __init__(self, api_key: str, config_path: str = "config.yaml",
                 channels_path: str = "channels.yaml"):
        self.config       = Config(config_path, channels_path)
        self.client       = YouTubeClient(api_key, self.config)
        self.collected:    List[Thread] = []
        self.channels_path = channels_path

        out_dir = self.config.get('output', 'directory', default='./data/raw_youtube')
        self.output_dir = Path(out_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ─── DISCOVER ────────────────────────────────────────────────────────────

    def discover_channels(self, query: str) -> None:
        print(f"\n🔍 Searching: '{query}'\n{'='*70}")
        channels = self.client.search_channels(query, max_results=10)
        if not channels:
            print("No channels found.")
            return
        for i, ch in enumerate(channels, 1):
            print(f"{i}. {ch['name']}")
            print(f"   ID          : {ch['id']}")
            print(f"   YouTube URL : https://www.youtube.com/channel/{ch['id']}")
            print(f"   Subscribers : {ch['subscribers']:,} | Videos: {ch['videos']}")
            print(f"   Description : {ch['description'][:80]}...")
            print()
        print("=" * 70)

    def discover_all_channels(self) -> None:
        """Run all discovery queries → write all results to channels.yaml."""
        ch_file = Path(self.channels_path)
        existing = yaml.safe_load(ch_file.read_text(encoding='utf-8')) or {} \
            if ch_file.exists() else {}

        queries = self.config.get_discovery_queries()
        print(f"\n🔍 Auto-discovering from {len(queries)} queries...\n{'='*70}")

        seen: Dict[str, Dict] = {}
        for query in queries:
            print(f"\n▶  Query: '{query}'")
            try:
                results = self.client.search_channels(query, max_results=10)
                new = sum(1 for r in results if r['id'] not in seen)
                for ch in results:
                    seen.setdefault(ch['id'], ch)
                print(f"   {len(results)} channels ({new} new)")
                time.sleep(1.0)
            except Exception as e:
                print(f"   ⚠️ Failed: {e}")

        if not seen:
            print("\n⚠️  No channels found.")
            return

        all_ch = sorted(seen.values(), key=lambda x: -x.get('subscribers', 0))

        print(f"\n{'='*70}\nDISCOVERED {len(all_ch)} UNIQUE CHANNELS\n{'='*70}")
        print(f"  {'#':>3}  {'Channel Name':<40}  {'Subs':>10}  {'Videos':>7}")
        for i, ch in enumerate(all_ch, 1):
            print(f"  {i:>3}  {ch['name']:<40}  {ch['subscribers']:>10,}  {ch['videos']:>7,}")
        print(f"{'='*70}")

        write_channels_yaml(all_ch, self.channels_path, existing)

        print(f"\n✅ channels.yaml updated with {len(all_ch)} channels (all active: true)")
        print("\n📋 NEXT STEPS:")
        print("   1. Open channels.yaml")
        print("   2. Click youtube_url for each channel to verify it's cooking")
        print("   3. DELETE non-cooking channel blocks (or set active: false)")
        print("   4. Run: python collect.py --collect")

    # ─── COLLECT ─────────────────────────────────────────────────────────────

    def collect_from_channel(self, channel_id: str, channel_name: str = "Unknown") -> List[Thread]:
        print(f"\n📺 {channel_name}")
        print(f"   URL: https://www.youtube.com/channel/{channel_id}")

        max_videos  = self.config.get('collection', 'max_videos_per_channel', default=100)
        max_threads = self.config.get('collection', 'max_threads_per_video',  default=200)
        delay       = self.config.get('api', 'delay_between_videos', default=0.5)

        channel_threads = []
        vp = 0

        for video in self.client.get_channel_videos(channel_id, max_videos):
            vp += 1
            self.client.stats.videos_processed += 1
            print(f"   [{vp}/{max_videos}] {video['title'][:55]}...")

            video_kept = 0
            for thread in self.client.get_video_threads(video, max_threads):
                self.client.stats.threads_found += 1
                keep, reason = self.client.filter_thread(thread)
                if keep:
                    channel_threads.append(thread)
                    self.client.stats.threads_kept += 1
                    video_kept += 1
                else:
                    self.client.stats.threads_filtered += 1
                    self.client.stats.filter_reasons[reason] = \
                        self.client.stats.filter_reasons.get(reason, 0) + 1

            if video_kept:
                print(f"       → {video_kept} threads kept")
            time.sleep(delay)

        self.client.stats.channels_processed += 1
        print(f"   ✓ Channel total: {len(channel_threads):,} threads")
        return channel_threads

    def collect_from_video(self, video_id: str) -> List[Thread]:
        print(f"\n📹 Video: https://www.youtube.com/watch?v={video_id}")
        max_threads = self.config.get('collection', 'max_threads_per_video', default=200)
        video = {'id': video_id, 'title': f'Video {video_id}',
                 'channel_id': 'unknown', 'channel_title': 'Unknown Channel'}
        try:
            resp = self.client._api_call(
                self.client.youtube.videos().list(part='snippet', id=video_id), "video details")
            if resp.get('items'):
                s = resp['items'][0]['snippet']
                video.update({'title': s['title'], 'channel_id': s['channelId'],
                              'channel_title': s['channelTitle']})
                print(f"   Title   : {video['title']}")
                print(f"   Channel : {video['channel_title']}")
        except Exception as e:
            print(f"   ⚠️ Could not fetch video details: {e}")

        threads = []
        for thread in self.client.get_video_threads(video, max_threads):
            keep, _ = self.client.filter_thread(thread)
            if keep:
                threads.append(thread)
        print(f"   ✓ {len(threads)} threads collected")
        return threads

    def collect_all(self) -> None:
        channels = self.config.get_active_channels()
        target   = self.config.get('collection', 'target_threads', default=30000)

        if not channels:
            print("⚠️  No active channels in channels.yaml!")
            return

        print(f"\n📋 Collecting from {len(channels)} channels | Target: {target:,} threads\n")

        for ch in channels:
            if len(self.collected) >= target:
                print(f"\n✅ Target of {target:,} threads reached. Stopping.")
                break
            threads = self.collect_from_channel(ch.get('id',''), ch.get('name','Unknown'))
            self.collected.extend(threads)
            print(f"   Running total: {len(self.collected):,} / {target:,}")
            time.sleep(self.config.get('api', 'delay_between_channels', default=2.0))

        self.save_results()

    # ─── SAVE ────────────────────────────────────────────────────────────────

    def _thread_to_dict(self, t: Thread) -> Dict:
        return {
            'thread_id':        t.thread_id,
            'video_id':         t.video_id,
            'channel_id':       t.channel_id,
            'video_title':      t.video_title,
            'channel_title':    t.channel_title,
            'source':           t.source,
            'appearance_count': t.appearance_count,
            'top_comment': {
                'comment_id': t.top_comment.comment_id,
                'text':       t.top_comment.text,
                'like_count': t.top_comment.like_count,
            },
            'replies': [
                {'comment_id': r.comment_id, 'text': r.text,
                 'like_count': r.like_count, 'is_creator': r.is_creator}
                for r in t.replies
            ],
            'has_creator_reply': t.has_creator_reply,
            'total_likes':       t.total_likes,
        }

    def save_results(self) -> None:
        if not self.collected:
            print("\n⚠️  No threads to save")
            return

        threads_file = self.config.get('output', 'threads_file', default='threads.jsonl')
        threads_path = self.output_dir / threads_file

        with open(threads_path, 'w', encoding='utf-8') as f:
            for t in self.collected:
                f.write(json.dumps(self._thread_to_dict(t), ensure_ascii=False) + '\n')

        print(f"\n💾 Saved {len(self.collected):,} threads → {threads_path}")

        self.client.stats.end_time = datetime.now().isoformat()
        stats_file = self.config.get('output', 'stats_file', default='collection_stats.json')
        with open(self.output_dir / stats_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.client.stats), f, indent=2, ensure_ascii=False)

        print(f"\n🔍 Generating channel report...")
        generate_channel_report(str(threads_path), str(self.output_dir))
        self._print_summary()

    def _print_summary(self) -> None:
        s = self.client.stats
        print(f"\n{'='*60}\nCOLLECTION SUMMARY\n{'='*60}")
        print(f"  Channels processed : {s.channels_processed}")
        print(f"  Videos processed   : {s.videos_processed}")
        print(f"  Threads found      : {s.threads_found:,}")
        print(f"  Threads kept       : {s.threads_kept:,}")
        print(f"  Threads filtered   : {s.threads_filtered:,}")
        print(f"  API calls          : {s.api_calls}")
        if s.filter_reasons:
            print("\n  Filter breakdown:")
            for reason, count in sorted(s.filter_reasons.items(), key=lambda x: -x[1]):
                print(f"    - {reason}: {count:,}")
        if self.collected:
            with_replies = sum(1 for t in self.collected if t.replies)
            with_creator = sum(1 for t in self.collected if t.has_creator_reply)
            questions    = sum(1 for t in self.collected
                               if HebrewUtils.is_question(t.top_comment.text))
            print(f"\n  Threads with replies      : {with_replies:,} "
                  f"({100*with_replies/len(self.collected):.1f}%)")
            print(f"  Threads with creator reply: {with_creator:,}")
            print(f"  Question threads (answered): {questions:,}")
        print("=" * 60)


# =============================================================================
# API TEST
# =============================================================================

def test_api(api_key: str, config: Config) -> bool:
    print("\n🔧 Testing YouTube API...\n")
    client = YouTubeClient(api_key, config)

    print("1. Channel search...")
    try:
        channels = client.search_channels("מתכונים בישול", max_results=2)
        if not channels:
            print("   ✗ No channels found"); return False
        print(f"   ✓ {len(channels)} channels found")
    except Exception as e:
        print(f"   ✗ {e}"); return False

    print("2. Video retrieval...")
    try:
        videos = list(client.get_channel_videos(channels[0]['id'], max_videos=2))
        if not videos:
            print("   ✗ No videos"); return False
        print(f"   ✓ {len(videos)} videos found")
    except Exception as e:
        print(f"   ✗ {e}"); return False

    print("3. Thread retrieval...")
    try:
        threads = list(client.get_video_threads(videos[0], max_threads=5))
        print(f"   ✓ {len(threads)} threads found")
        for t in threads[:2]:
            q = " [QUESTION]" if HebrewUtils.is_question(t.top_comment.text) else ""
            print(f"   Thread{q}: {t.top_comment.text[:60]}...")
            if t.replies:
                print(f"     → {len(t.replies)} replies, "
                      f"creator: {t.has_creator_reply}, "
                      f"channel_id: {t.channel_id}")
    except Exception as e:
        print(f"   ✗ {e}"); return False

    print(f"\n{'='*60}\n✅ ALL TESTS PASSED\n{'='*60}")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='YouTube Thread Collector v3 — Recipe Modifications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (require API key):
  python collect.py --test
  python collect.py --discover-all
  python collect.py --discover "מתכונים בישול"
  python collect.py --channel UCxxxxxxx
  python collect.py --video   VIDEO_ID
  python collect.py --collect

Examples (no API key):
  python collect.py --list-channels
  python collect.py --list-channels --input data/raw_youtube/threads.jsonl
  python collect.py --filter-channels --input threads.jsonl --csv channels_report.csv
        """)

    parser.add_argument('--api-key',      type=str)
    parser.add_argument('--api-key-file', type=str)
    parser.add_argument('--config',       type=str, default='config.yaml')
    parser.add_argument('--channels',     type=str, default='channels.yaml')

    # API actions
    parser.add_argument('--test',         action='store_true')
    parser.add_argument('--discover-all', action='store_true')
    parser.add_argument('--discover',     type=str, metavar='QUERY')
    parser.add_argument('--channel',      type=str, metavar='ID')
    parser.add_argument('--video',        type=str, metavar='ID')
    parser.add_argument('--collect',      action='store_true')

    # No-API actions
    parser.add_argument('--list-channels',    action='store_true')
    parser.add_argument('--filter-channels',  action='store_true')
    parser.add_argument('--input',  type=str, default='data/raw_youtube/threads.jsonl')
    parser.add_argument('--csv',    type=str, default='data/raw_youtube/channels_report.csv')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    # No-API commands
    if args.list_channels:
        generate_channel_report(args.input)
        return 0
    if args.filter_channels:
        filter_channels_from_jsonl(args.input, args.csv, args.output)
        return 0

    # API key
    api_key = args.api_key
    if not api_key and args.api_key_file:
        with open(args.api_key_file, 'r') as f:
            api_key = f.read().strip()
    if not api_key:
        api_key = os.environ.get('YOUTUBE_API_KEY')

    if not api_key:
        print("❌ No API key provided.")
        print("   Use: --api-key KEY  |  --api-key-file FILE  |  YOUTUBE_API_KEY env var")
        print("\nNote: --list-channels and --filter-channels work without an API key.")
        return 1

    config = Config(args.config, args.channels)

    if args.test:
        return 0 if test_api(api_key, config) else 1
    elif args.discover_all:
        Collector(api_key, args.config, args.channels).discover_all_channels()
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