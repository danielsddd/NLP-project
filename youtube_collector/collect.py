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
from typing import List, Dict, Optional, Generator, Any

import yaml
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Comment:
    """
    Represents a YouTube comment - MINIMAL fields for NLP project.
    
    We only collect what's needed for:
    1. Recipe modification extraction (text)
    2. Ranking module (like_count)
    3. Context (video_title, channel_title)
    """
    # === ESSENTIAL FOR NLP ===
    text: str                   # The comment content - THIS IS WHAT WE ANALYZE
    like_count: int             # For ranking module (social signal)
    video_title: str            # Context: which recipe
    channel_title: str          # Context: which cooking channel
    
    # === FOR DATA MANAGEMENT ===
    comment_id: str             # For deduplication only
    video_id: str               # To group by recipe
    
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
    """Loads and provides access to configuration."""
    
    def __init__(self, config_path: str = "config.yaml", channels_path: str = "channels.yaml"):
        self.config_path = Path(config_path)
        self.channels_path = Path(channels_path)
        self._config = {}
        self._channels = {}
        self._load()
    
    def _load(self):
        """Load configuration files."""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            print(f"⚠️  Config file not found: {self.config_path}")
            print("   Using default settings.")
            self._config = self._default_config()
        
        if self.channels_path.exists():
            with open(self.channels_path, 'r', encoding='utf-8') as f:
                self._channels = yaml.safe_load(f) or {}
    
    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'output': {
                'directory': './data/raw_youtube',
                'comments_file': 'comments.jsonl',
                'stats_file': 'collection_stats.json',
            },
            'collection': {
                'target_comments': 3000,
                'max_videos_per_channel': 50,
                'max_comments_per_video': 100,
                'include_replies': True,
            },
            'filtering': {
                'min_words': 5,
                'require_hebrew': True,
                'skip_creator_comments': True,
                'spam_keywords': ['http', 'https', 'www.', 'bit.ly'],
            },
            'api': {
                'delay_between_videos': 0.5,
                'delay_between_channels': 2.0,
                'max_retries': 5,
                'region_code': 'IL',
            }
        }
    
    def get(self, *keys, default=None):
        """Get a nested config value."""
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value if value is not None else default
    
    def get_active_channels(self) -> List[Dict]:
        """Get list of active channels to collect from."""
        channels = self._channels.get('channels', [])
        return [ch for ch in channels if ch.get('active', False)]
    
    def get_all_modification_keywords(self) -> List[str]:
        """Get flat list of all modification keywords."""
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
    """Utilities for Hebrew text processing."""
    
    HEBREW_PATTERN = re.compile(r'[\u0590-\u05FF]')
    
    @classmethod
    def contains_hebrew(cls, text: str) -> bool:
        """Check if text contains Hebrew characters."""
        return bool(cls.HEBREW_PATTERN.search(text))
    
    @classmethod
    def word_count(cls, text: str) -> int:
        """Count words in text."""
        return len(text.strip().split())
    
    @classmethod
    def is_spam(cls, text: str, spam_keywords: List[str]) -> bool:
        """Check if text contains spam indicators."""
        text_lower = text.lower()
        for keyword in spam_keywords:
            if keyword.lower() in text_lower:
                return True
        return False
    
    @classmethod
    def find_keywords(cls, text: str, keywords: List[str]) -> List[str]:
        """Find which keywords appear in text."""
        found = []
        for keyword in keywords:
            if keyword in text:
                found.append(keyword)
        return found


# =============================================================================
# YOUTUBE API CLIENT
# =============================================================================

class YouTubeClient:
    """Wrapper for YouTube Data API v3."""
    
    def __init__(self, api_key: str, config: Config):
        self.api_key = api_key
        self.config = config
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.stats = CollectionStats(start_time=datetime.now().isoformat())
        self.seen_ids: set = set()
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging."""
        log_file = self.config.get('output', 'log_file')
        
        handlers = [logging.StreamHandler()]
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path, encoding='utf-8'))
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=handlers
        )
        self.logger = logging.getLogger(__name__)
    
    def _api_call(self, request, description: str = "API call"):
        """Execute API call with retry logic."""
        max_retries = self.config.get('api', 'max_retries', default=5)
        base_delay = self.config.get('api', 'base_retry_delay', default=1.0)
        
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
    # DISCOVERY METHODS
    # -------------------------------------------------------------------------
    
    def search_channels(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search for channels matching a query."""
        self.logger.info(f"🔍 Searching channels: '{query}'")
        
        region = self.config.get('api', 'region_code', default='IL')
        request = self.youtube.search().list(
            part='snippet',
            q=query,
            type='channel',
            maxResults=max_results,
            regionCode=region
        )
        response = self._api_call(request, "channel search")
        
        channels = []
        for item in response.get('items', []):
            channel_id = item['snippet']['channelId']
            
            # Get detailed channel info
            info_request = self.youtube.channels().list(
                part='snippet,statistics',
                id=channel_id
            )
            info_response = self._api_call(info_request, "channel info")
            
            if info_response.get('items'):
                info = info_response['items'][0]
                channels.append({
                    'id': channel_id,
                    'name': info['snippet']['title'],
                    'description': info['snippet'].get('description', '')[:200],
                    'subscribers': int(info['statistics'].get('subscriberCount', 0)),
                    'videos': int(info['statistics'].get('videoCount', 0)),
                })
        
        return channels
    
    def get_channel_videos(self, channel_id: str, max_videos: int = 50) -> Generator[Dict, None, None]:
        """Get videos from a channel."""
        next_page = None
        retrieved = 0
        
        while retrieved < max_videos:
            request = self.youtube.search().list(
                part='snippet',
                channelId=channel_id,
                type='video',
                order='date',
                maxResults=min(50, max_videos - retrieved),
                pageToken=next_page
            )
            response = self._api_call(request, f"videos for {channel_id}")
            
            for item in response.get('items', []):
                yield {
                    'id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'channel_id': channel_id,
                    'channel_title': item['snippet']['channelTitle'],
                    'published': item['snippet']['publishedAt'],
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
        self, 
        video: Dict,
        max_comments: int = 100
    ) -> Generator[Comment, None, None]:
        """Get comments from a video."""
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
                    textFormat='plainText'
                )
                response = self._api_call(request, f"comments for {video['id']}")
                
                for item in response.get('items', []):
                    # Process top-level comment
                    comment = self._parse_comment(item, video, is_reply=False)
                    if comment:
                        yield comment
                        retrieved += 1
                    
                    # Process replies
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
    
    def _parse_comment(self, item: Dict, video: Dict, is_reply: bool) -> Optional[Comment]:
        """Parse a comment thread item into a Comment object."""
        snippet = item['snippet']['topLevelComment']['snippet']
        comment_id = item['snippet']['topLevelComment']['id']
        
        # Skip if already seen
        if comment_id in self.seen_ids:
            return None
        self.seen_ids.add(comment_id)
        
        # Skip creator comments if configured
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
            word_count=HebrewUtils.word_count(text),
        )
    
    def _parse_reply(self, reply_item: Dict, parent_item: Dict, video: Dict) -> Optional[Comment]:
        """Parse a reply into a Comment object."""
        snippet = reply_item['snippet']
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
    
    def filter_comment(self, comment: Comment) -> tuple[bool, str]:
        """
        Check if comment should be kept.
        
        Returns:
            (keep: bool, reason: str) - reason is why it was filtered (if not kept)
        """
        # Check Hebrew requirement
        if self.config.get('filtering', 'require_hebrew', default=True):
            if not HebrewUtils.contains_hebrew(comment.text):
                return False, "no_hebrew"
        
        # Check minimum word count
        min_words = self.config.get('filtering', 'min_words', default=5)
        if comment.word_count < min_words:
            return False, "too_short"
        
        # Check for spam
        spam_keywords = self.config.get('filtering', 'spam_keywords', default=[])
        if HebrewUtils.is_spam(comment.text, spam_keywords):
            return False, "spam"
        
        return True, ""
    
    def enrich_comment(self, comment: Comment) -> Comment:
        """Add additional fields to comment (keyword detection, etc.)."""
        keywords = self.config.get_all_modification_keywords()
        found = HebrewUtils.find_keywords(comment.text, keywords)
        comment.has_modification_keyword = len(found) > 0
        comment.detected_keywords = found
        return comment


# =============================================================================
# MAIN COLLECTOR
# =============================================================================

class Collector:
    """Main collection orchestrator."""
    
    def __init__(self, api_key: str, config_path: str = "config.yaml", channels_path: str = "channels.yaml"):
        self.config = Config(config_path, channels_path)
        self.client = YouTubeClient(api_key, self.config)
        self.collected: List[Comment] = []
        
        # Create output directory
        output_dir = self.config.get('output', 'directory', default='./data/raw_youtube')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def discover_channels(self, query: str):
        """Search for channels and print results."""
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
    
    def collect_from_channel(self, channel_id: str, channel_name: str = "Unknown"):
        """Collect comments from a single channel."""
        print(f"\n📺 Collecting from: {channel_name}")
        print(f"   Channel ID: {channel_id}")
        
        max_videos = self.config.get('collection', 'max_videos_per_channel', default=50)
        max_comments = self.config.get('collection', 'max_comments_per_video', default=100)
        delay = self.config.get('api', 'delay_between_videos', default=0.5)
        
        channel_comments = []
        videos_processed = 0
        
        for video in self.client.get_channel_videos(channel_id, max_videos):
            videos_processed += 1
            self.client.stats.videos_processed += 1
            
            print(f"   [{videos_processed}/{max_videos}] {video['title'][:50]}...")
            
            video_comments = 0
            for comment in self.client.get_video_comments(video, max_comments):
                self.client.stats.comments_found += 1
                
                # Filter
                keep, reason = self.client.filter_comment(comment)
                
                if keep:
                    # Enrich with keyword detection
                    comment = self.client.enrich_comment(comment)
                    channel_comments.append(comment)
                    self.client.stats.comments_kept += 1
                    video_comments += 1
                else:
                    self.client.stats.comments_filtered += 1
                    self.client.stats.filter_reasons[reason] = \
                        self.client.stats.filter_reasons.get(reason, 0) + 1
            
            if video_comments > 0:
                print(f"       → {video_comments} comments kept")
            
            time.sleep(delay)
        
        self.client.stats.channels_processed += 1
        print(f"   ✓ Channel total: {len(channel_comments)} comments")
        
        return channel_comments
    
    def collect_from_video(self, video_id: str):
        """Collect comments from a single video."""
        print(f"\n🎬 Collecting from video: {video_id}")
        
        # Get video info first
        request = self.client.youtube.videos().list(
            part='snippet',
            id=video_id
        )
        response = self.client._api_call(request, "video info")
        
        if not response.get('items'):
            print("❌ Video not found")
            return []
        
        video_info = response['items'][0]['snippet']
        video = {
            'id': video_id,
            'title': video_info['title'],
            'channel_id': video_info['channelId'],
            'channel_title': video_info['channelTitle'],
        }
        
        print(f"   Title: {video['title'][:60]}...")
        print(f"   Channel: {video['channel_title']}")
        
        max_comments = self.config.get('collection', 'max_comments_per_video', default=100)
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
        
        print(f"   ✓ Collected: {len(video_comments)} comments")
        return video_comments
    
    def collect_all(self):
        """Collect from all active channels."""
        channels = self.config.get_active_channels()
        
        if not channels:
            print("\n⚠️  No active channels configured!")
            print("   1. Run: python collect.py --discover \"מתכונים\"")
            print("   2. Add channels to channels.yaml with 'active: true'")
            return
        
        print(f"\n🚀 Starting collection from {len(channels)} channels")
        target = self.config.get('collection', 'target_comments', default=3000)
        print(f"   Target: {target} comments\n")
        
        delay = self.config.get('api', 'delay_between_channels', default=2.0)
        
        for channel in channels:
            comments = self.collect_from_channel(
                channel_id=channel['id'],
                channel_name=channel.get('name', 'Unknown')
            )
            self.collected.extend(comments)
            
            print(f"\n   📊 Progress: {len(self.collected)}/{target} comments")
            
            if len(self.collected) >= target:
                print("\n✅ Target reached!")
                break
            
            time.sleep(delay)
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save collected comments and statistics."""
        if not self.collected:
            print("\n⚠️  No comments to save")
            return
        
        # Save comments
        comments_file = self.config.get('output', 'comments_file', default='comments.jsonl')
        comments_path = self.output_dir / comments_file
        
        with open(comments_path, 'w', encoding='utf-8') as f:
            for comment in self.collected:
                line = json.dumps(asdict(comment), ensure_ascii=False)
                f.write(line + '\n')
        
        print(f"\n💾 Saved {len(self.collected)} comments to: {comments_path}")
        
        # Save statistics
        self.client.stats.end_time = datetime.now().isoformat()
        stats_file = self.config.get('output', 'stats_file', default='collection_stats.json')
        stats_path = self.output_dir / stats_file
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.client.stats), f, indent=2, ensure_ascii=False)
        
        print(f"📊 Saved statistics to: {stats_path}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print collection summary."""
        stats = self.client.stats
        
        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        print(f"  Channels processed:  {stats.channels_processed}")
        print(f"  Videos processed:    {stats.videos_processed}")
        print(f"  Comments found:      {stats.comments_found}")
        print(f"  Comments kept:       {stats.comments_kept}")
        print(f"  Comments filtered:   {stats.comments_filtered}")
        print(f"  API calls made:      {stats.api_calls}")
        
        if stats.filter_reasons:
            print("\n  Filter breakdown:")
            for reason, count in sorted(stats.filter_reasons.items(), key=lambda x: -x[1]):
                print(f"    - {reason}: {count}")
        
        # Analyze collected comments
        if self.collected:
            with_keywords = sum(1 for c in self.collected if c.has_modification_keyword)
            avg_likes = sum(c.like_count for c in self.collected) / len(self.collected)
            print(f"\n  With modification keywords: {with_keywords} ({100*with_keywords/len(self.collected):.1f}%)")
            print(f"  Average likes per comment:  {avg_likes:.1f}")
        
        print("=" * 60)


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_api(api_key: str, config: Config):
    """Quick API test."""
    print("\n" + "=" * 60)
    print("API CONNECTION TEST")
    print("=" * 60)
    
    client = YouTubeClient(api_key, config)
    
    # Test 1: Search
    print("\n1. Testing channel search...")
    channels = client.search_channels("מתכונים", max_results=2)
    if channels:
        print(f"   ✓ Found {len(channels)} channels")
        for ch in channels:
            print(f"     - {ch['name']} ({ch['videos']} videos)")
    else:
        print("   ✗ No channels found")
        return False
    
    # Test 2: Get videos
    print("\n2. Testing video retrieval...")
    videos = list(client.get_channel_videos(channels[0]['id'], max_videos=2))
    if videos:
        print(f"   ✓ Found {len(videos)} videos")
        for v in videos:
            print(f"     - {v['title'][:50]}...")
    else:
        print("   ✗ No videos found")
        return False
    
    # Test 3: Get comments
    print("\n3. Testing comment retrieval...")
    comments = list(client.get_video_comments(videos[0], max_comments=10))
    print(f"   ✓ Found {len(comments)} comments")
    
    # Show Hebrew comments
    hebrew = [c for c in comments if HebrewUtils.contains_hebrew(c.text)]
    print(f"   ✓ Hebrew comments: {len(hebrew)}/{len(comments)}")
    
    if hebrew:
        print("\n   Sample comment:")
        sample = hebrew[0]
        print(f"   Likes: {sample.like_count} | Words: {sample.word_count}")
        print(f"   Text: {sample.text[:100]}...")
    
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
  python collect.py --test                          # Test API connection
  python collect.py --discover "מתכונים בישול"      # Find channels
  python collect.py --channel UCxxxxxxx             # Collect from channel
  python collect.py --video dQw4w9WgXcQ             # Collect from video
  python collect.py --collect                       # Collect from all active channels
        """
    )
    
    # API key options
    parser.add_argument('--api-key', type=str, help='YouTube API key')
    parser.add_argument('--api-key-file', type=str, help='File containing API key')
    
    # Config options
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--channels', type=str, default='channels.yaml', help='Channels file path')
    
    # Actions
    parser.add_argument('--test', action='store_true', help='Test API connection')
    parser.add_argument('--discover', type=str, metavar='QUERY', help='Search for channels')
    parser.add_argument('--channel', type=str, metavar='ID', help='Collect from specific channel')
    parser.add_argument('--video', type=str, metavar='ID', help='Collect from specific video')
    parser.add_argument('--collect', action='store_true', help='Collect from all active channels')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key
    if not api_key and args.api_key_file:
        with open(args.api_key_file, 'r') as f:
            api_key = f.read().strip()
    if not api_key:
        api_key = os.environ.get('YOUTUBE_API_KEY')
    
    if not api_key:
        print("❌ No API key provided!")
        print("\nProvide API key via:")
        print("  --api-key YOUR_KEY")
        print("  --api-key-file path/to/key.txt")
        print("  YOUTUBE_API_KEY environment variable")
        return 1
    
    # Load config
    config = Config(args.config, args.channels)
    
    # Execute action
    if args.test:
        success = test_api(api_key, config)
        return 0 if success else 1
    
    elif args.discover:
        collector = Collector(api_key, args.config, args.channels)
        collector.discover_channels(args.discover)
        return 0
    
    elif args.channel:
        collector = Collector(api_key, args.config, args.channels)
        comments = collector.collect_from_channel(args.channel)
        collector.collected = comments
        collector.save_results()
        return 0
    
    elif args.video:
        collector = Collector(api_key, args.config, args.channels)
        comments = collector.collect_from_video(args.video)
        collector.collected = comments
        collector.save_results()
        return 0
    
    elif args.collect:
        collector = Collector(api_key, args.config, args.channels)
        collector.collect_all()
        return 0
    
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
