"""
YouTube Comment Collector for Hebrew Recipe Modification Extraction
===================================================================
Group 11: Daniel Simanovsky & Roei Ben Artzi
NLP 2025a, Tel Aviv University

This script collects Hebrew comments from cooking channels on YouTube
for the purpose of extracting recipe modifications.
"""

import json
import re
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, asdict
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Comment:
    """Data class representing a YouTube comment."""
    comment_id: str
    video_id: str
    video_title: str
    channel_id: str
    channel_title: str
    text: str
    author: str
    author_channel_id: str
    like_count: int
    published_at: str
    updated_at: str
    reply_count: int
    is_reply: bool = False
    parent_id: Optional[str] = None


class HebrewFilter:
    """Utility class for filtering Hebrew content."""
    
    # Hebrew Unicode range
    HEBREW_PATTERN = re.compile(r'[\u0590-\u05FF]')
    
    # Spam indicators (Hebrew and English)
    SPAM_INDICATORS = [
        'הירשמו',       # Subscribe
        'לינק',         # Link  
        'http',
        'https',
        'www.',
        'לחצו כאן',     # Click here
        'משחק',         # Game (often spam)
        'פרסומת',       # Advertisement
        'קישור',        # Link
        '.com',
        '.co.il',
        'bit.ly',
        'tinyurl',
        '💰',           # Money emoji (often spam)
        '🎁',           # Gift emoji (often spam)
        'free',
        'giveaway',
    ]
    
    @classmethod
    def contains_hebrew(cls, text: str) -> bool:
        """Check if text contains Hebrew characters."""
        return bool(cls.HEBREW_PATTERN.search(text))
    
    @classmethod
    def word_count(cls, text: str) -> int:
        """Count words in text."""
        return len(text.strip().split())
    
    @classmethod
    def is_spam(cls, text: str) -> bool:
        """Check if text appears to be spam."""
        text_lower = text.lower()
        for indicator in cls.SPAM_INDICATORS:
            if indicator.lower() in text_lower:
                return True
        return False
    
    @classmethod
    def is_valid_comment(
        cls, 
        text: str, 
        min_words: int = 5,
        require_hebrew: bool = True
    ) -> bool:
        """
        Check if a comment passes all filters.
        
        Args:
            text: Comment text
            min_words: Minimum word count
            require_hebrew: Whether Hebrew content is required
            
        Returns:
            True if comment passes all filters
        """
        # Filter 1: Hebrew content check
        if require_hebrew and not cls.contains_hebrew(text):
            return False
        
        # Filter 2: Minimum length
        if cls.word_count(text) < min_words:
            return False
        
        # Filter 3: Spam check
        if cls.is_spam(text):
            return False
        
        return True


class YouTubeCollector:
    """
    Collects comments from YouTube cooking channels.
    
    Handles rate limiting, pagination, and filtering for Hebrew content.
    """
    
    def __init__(
        self, 
        api_key: str,
        output_dir: str = "data/raw_youtube",
        max_retries: int = 5,
        base_delay: float = 1.0
    ):
        """
        Initialize the YouTube collector.
        
        Args:
            api_key: YouTube Data API key
            output_dir: Directory to save collected comments
            max_retries: Maximum retry attempts for API calls
            base_delay: Base delay for exponential backoff (seconds)
        """
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        # Build YouTube API client
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        
        # Track seen comment IDs to avoid duplicates
        self.seen_comment_ids: set = set()
        
        # Statistics
        self.stats = {
            'total_videos_processed': 0,
            'total_comments_collected': 0,
            'total_comments_filtered': 0,
            'api_calls': 0,
            'errors': 0
        }
    
    def _api_call_with_retry(self, request):
        """
        Execute an API request with exponential backoff retry.
        
        Args:
            request: YouTube API request object
            
        Returns:
            API response
            
        Raises:
            HttpError: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                self.stats['api_calls'] += 1
                response = request.execute()
                return response
            except HttpError as e:
                if e.resp.status == 403:
                    # Quota exceeded
                    logger.error(f"API quota exceeded: {e}")
                    raise
                elif e.resp.status == 429:
                    # Rate limited
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"API error: {e}")
                    self.stats['errors'] += 1
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(self.base_delay)
        
        raise Exception("Max retries exceeded")
    
    def get_channel_info(self, channel_id: str) -> Optional[Dict]:
        """
        Get information about a YouTube channel.
        
        Args:
            channel_id: YouTube channel ID
            
        Returns:
            Channel information dict or None if not found
        """
        try:
            request = self.youtube.channels().list(
                part='snippet,statistics',
                id=channel_id
            )
            response = self._api_call_with_retry(request)
            
            if response.get('items'):
                item = response['items'][0]
                return {
                    'channel_id': channel_id,
                    'title': item['snippet']['title'],
                    'description': item['snippet'].get('description', ''),
                    'subscriber_count': int(item['statistics'].get('subscriberCount', 0)),
                    'video_count': int(item['statistics'].get('videoCount', 0))
                }
        except Exception as e:
            logger.error(f"Error getting channel info for {channel_id}: {e}")
        
        return None
    
    def search_channels(
        self, 
        query: str, 
        max_results: int = 10,
        region_code: str = 'IL'
    ) -> List[Dict]:
        """
        Search for YouTube channels by query.
        
        Args:
            query: Search query (e.g., "בישול" for cooking)
            max_results: Maximum number of results
            region_code: Region code for localized results
            
        Returns:
            List of channel information dicts
        """
        channels = []
        try:
            request = self.youtube.search().list(
                part='snippet',
                q=query,
                type='channel',
                maxResults=max_results,
                regionCode=region_code
            )
            response = self._api_call_with_retry(request)
            
            for item in response.get('items', []):
                channel_id = item['snippet']['channelId']
                channel_info = self.get_channel_info(channel_id)
                if channel_info:
                    channels.append(channel_info)
                    logger.info(f"Found channel: {channel_info['title']} ({channel_info['video_count']} videos)")
        
        except Exception as e:
            logger.error(f"Error searching channels: {e}")
        
        return channels
    
    def get_channel_videos(
        self, 
        channel_id: str,
        max_videos: int = 50
    ) -> Generator[Dict, None, None]:
        """
        Get videos from a channel.
        
        Args:
            channel_id: YouTube channel ID
            max_videos: Maximum number of videos to retrieve
            
        Yields:
            Video information dicts
        """
        next_page_token = None
        videos_retrieved = 0
        
        while videos_retrieved < max_videos:
            try:
                request = self.youtube.search().list(
                    part='snippet',
                    channelId=channel_id,
                    type='video',
                    order='date',
                    maxResults=min(50, max_videos - videos_retrieved),
                    pageToken=next_page_token
                )
                response = self._api_call_with_retry(request)
                
                for item in response.get('items', []):
                    video_info = {
                        'video_id': item['id']['videoId'],
                        'title': item['snippet']['title'],
                        'description': item['snippet'].get('description', ''),
                        'channel_id': channel_id,
                        'channel_title': item['snippet']['channelTitle'],
                        'published_at': item['snippet']['publishedAt']
                    }
                    yield video_info
                    videos_retrieved += 1
                    
                    if videos_retrieved >= max_videos:
                        break
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except Exception as e:
                logger.error(f"Error getting videos for channel {channel_id}: {e}")
                break
    
    def get_video_comments(
        self, 
        video_id: str,
        video_title: str,
        channel_id: str,
        channel_title: str,
        max_comments: int = 100,
        include_replies: bool = True
    ) -> Generator[Comment, None, None]:
        """
        Get comments from a video.
        
        Args:
            video_id: YouTube video ID
            video_title: Video title (for metadata)
            channel_id: Channel ID (for filtering creator comments)
            channel_title: Channel title (for metadata)
            max_comments: Maximum comments to retrieve
            include_replies: Whether to include reply comments
            
        Yields:
            Comment objects
        """
        next_page_token = None
        comments_retrieved = 0
        
        while comments_retrieved < max_comments:
            try:
                request = self.youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    maxResults=min(100, max_comments - comments_retrieved),
                    pageToken=next_page_token,
                    textFormat='plainText'
                )
                response = self._api_call_with_retry(request)
                
                for item in response.get('items', []):
                    # Top-level comment
                    top_comment = item['snippet']['topLevelComment']['snippet']
                    comment_id = item['snippet']['topLevelComment']['id']
                    
                    # Skip if already seen
                    if comment_id in self.seen_comment_ids:
                        continue
                    self.seen_comment_ids.add(comment_id)
                    
                    # Get author channel ID
                    author_channel_id = top_comment.get('authorChannelId', {}).get('value', '')
                    
                    # Skip creator's own comments (often pinned, not modifications)
                    if author_channel_id == channel_id:
                        continue
                    
                    comment = Comment(
                        comment_id=comment_id,
                        video_id=video_id,
                        video_title=video_title,
                        channel_id=channel_id,
                        channel_title=channel_title,
                        text=top_comment['textDisplay'],
                        author=top_comment['authorDisplayName'],
                        author_channel_id=author_channel_id,
                        like_count=top_comment.get('likeCount', 0),
                        published_at=top_comment['publishedAt'],
                        updated_at=top_comment.get('updatedAt', top_comment['publishedAt']),
                        reply_count=item['snippet'].get('totalReplyCount', 0),
                        is_reply=False,
                        parent_id=None
                    )
                    
                    yield comment
                    comments_retrieved += 1
                    
                    # Process replies
                    if include_replies and 'replies' in item:
                        for reply_item in item['replies']['comments']:
                            reply_snippet = reply_item['snippet']
                            reply_id = reply_item['id']
                            
                            if reply_id in self.seen_comment_ids:
                                continue
                            self.seen_comment_ids.add(reply_id)
                            
                            reply_author_channel_id = reply_snippet.get('authorChannelId', {}).get('value', '')
                            if reply_author_channel_id == channel_id:
                                continue
                            
                            reply_comment = Comment(
                                comment_id=reply_id,
                                video_id=video_id,
                                video_title=video_title,
                                channel_id=channel_id,
                                channel_title=channel_title,
                                text=reply_snippet['textDisplay'],
                                author=reply_snippet['authorDisplayName'],
                                author_channel_id=reply_author_channel_id,
                                like_count=reply_snippet.get('likeCount', 0),
                                published_at=reply_snippet['publishedAt'],
                                updated_at=reply_snippet.get('updatedAt', reply_snippet['publishedAt']),
                                reply_count=0,
                                is_reply=True,
                                parent_id=comment_id
                            )
                            yield reply_comment
                    
                    if comments_retrieved >= max_comments:
                        break
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except HttpError as e:
                if 'commentsDisabled' in str(e):
                    logger.info(f"Comments disabled for video {video_id}")
                else:
                    logger.error(f"Error getting comments for video {video_id}: {e}")
                break
            except Exception as e:
                logger.error(f"Error getting comments for video {video_id}: {e}")
                break
    
    def collect_from_channel(
        self,
        channel_id: str,
        max_videos: int = 50,
        max_comments_per_video: int = 100,
        min_words: int = 5,
        require_hebrew: bool = True
    ) -> List[Comment]:
        """
        Collect filtered comments from a channel.
        
        Args:
            channel_id: YouTube channel ID
            max_videos: Maximum videos to process
            max_comments_per_video: Maximum comments per video
            min_words: Minimum word count for comments
            require_hebrew: Whether to require Hebrew content
            
        Returns:
            List of filtered Comment objects
        """
        collected_comments = []
        
        # Get channel info
        channel_info = self.get_channel_info(channel_id)
        if not channel_info:
            logger.error(f"Could not get info for channel {channel_id}")
            return collected_comments
        
        logger.info(f"Collecting from channel: {channel_info['title']}")
        
        # Process videos
        for video in self.get_channel_videos(channel_id, max_videos):
            self.stats['total_videos_processed'] += 1
            logger.info(f"Processing video: {video['title'][:50]}...")
            
            # Get comments
            for comment in self.get_video_comments(
                video_id=video['video_id'],
                video_title=video['title'],
                channel_id=channel_id,
                channel_title=channel_info['title'],
                max_comments=max_comments_per_video
            ):
                # Apply filters
                if HebrewFilter.is_valid_comment(
                    comment.text, 
                    min_words=min_words,
                    require_hebrew=require_hebrew
                ):
                    collected_comments.append(comment)
                    self.stats['total_comments_collected'] += 1
                else:
                    self.stats['total_comments_filtered'] += 1
            
            # Small delay between videos to be nice to the API
            time.sleep(0.5)
        
        return collected_comments
    
    def collect_from_channels(
        self,
        channel_ids: List[str],
        max_videos_per_channel: int = 50,
        max_comments_per_video: int = 100,
        min_words: int = 5,
        require_hebrew: bool = True,
        target_count: Optional[int] = None
    ) -> List[Comment]:
        """
        Collect comments from multiple channels.
        
        Args:
            channel_ids: List of YouTube channel IDs
            max_videos_per_channel: Maximum videos per channel
            max_comments_per_video: Maximum comments per video
            min_words: Minimum word count
            require_hebrew: Whether to require Hebrew content
            target_count: Stop after collecting this many comments (optional)
            
        Returns:
            List of collected Comment objects
        """
        all_comments = []
        
        for channel_id in channel_ids:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing channel: {channel_id}")
            logger.info(f"{'='*60}")
            
            comments = self.collect_from_channel(
                channel_id=channel_id,
                max_videos=max_videos_per_channel,
                max_comments_per_video=max_comments_per_video,
                min_words=min_words,
                require_hebrew=require_hebrew
            )
            all_comments.extend(comments)
            
            logger.info(f"Collected {len(comments)} comments from this channel")
            logger.info(f"Total so far: {len(all_comments)}")
            
            # Check if we've reached target
            if target_count and len(all_comments) >= target_count:
                logger.info(f"Reached target count of {target_count}")
                break
        
        return all_comments
    
    def save_comments(
        self, 
        comments: List[Comment], 
        filename: str = "comments.jsonl"
    ) -> Path:
        """
        Save comments to a JSONL file.
        
        Args:
            comments: List of Comment objects
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for comment in comments:
                json_line = json.dumps(asdict(comment), ensure_ascii=False)
                f.write(json_line + '\n')
        
        logger.info(f"Saved {len(comments)} comments to {output_path}")
        return output_path
    
    def save_stats(self, filename: str = "collection_stats.json") -> Path:
        """
        Save collection statistics.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        stats_with_timestamp = {
            **self.stats,
            'timestamp': datetime.now().isoformat(),
            'seen_comment_ids_count': len(self.seen_comment_ids)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_with_timestamp, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved stats to {output_path}")
        return output_path
    
    def print_stats(self):
        """Print collection statistics."""
        print("\n" + "="*60)
        print("COLLECTION STATISTICS")
        print("="*60)
        print(f"Videos processed:     {self.stats['total_videos_processed']}")
        print(f"Comments collected:   {self.stats['total_comments_collected']}")
        print(f"Comments filtered:    {self.stats['total_comments_filtered']}")
        print(f"API calls made:       {self.stats['api_calls']}")
        print(f"Errors encountered:   {self.stats['errors']}")
        print(f"Unique comment IDs:   {len(self.seen_comment_ids)}")
        print("="*60 + "\n")


# =============================================================================
# KNOWN HEBREW COOKING CHANNELS
# =============================================================================
# These are some popular Hebrew cooking channels you can use.
# To find more, use the search_channels() method with queries like:
# "בישול", "מתכונים", "אפייה", "שף"

HEBREW_COOKING_CHANNELS = {
    # Format: 'channel_name': 'channel_id'
    # You need to find the actual channel IDs - see instructions below
    
    # Example (replace with real IDs):
    # 'דנילה גרמון': 'UCxxxxxxxxxx',
    # 'השף הלבן': 'UCyyyyyyyyyy',
}


def find_channel_id(channel_url: str) -> str:
    """
    Helper to extract channel ID from different URL formats.
    
    YouTube channel URLs can be:
    - https://www.youtube.com/channel/UCxxxxxxx (direct ID)
    - https://www.youtube.com/@username (handle - need API to resolve)
    - https://www.youtube.com/c/ChannelName (custom URL - need API to resolve)
    
    For handles (@username), you need to use the API to search for the channel.
    """
    if '/channel/' in channel_url:
        return channel_url.split('/channel/')[-1].split('/')[0].split('?')[0]
    else:
        # For @handles and custom URLs, use the search API
        return None


# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

def main():
    """
    Main function to demonstrate and test the YouTube collector.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='YouTube Comment Collector for Recipe Modifications')
    parser.add_argument('--api-key', type=str, help='YouTube Data API key')
    parser.add_argument('--api-key-file', type=str, help='Path to file containing API key')
    parser.add_argument('--output-dir', type=str, default='data/raw_youtube', help='Output directory')
    parser.add_argument('--search-query', type=str, help='Search for channels with this query')
    parser.add_argument('--channel-id', type=str, help='Collect from specific channel ID')
    parser.add_argument('--video-id', type=str, help='Collect from specific video ID')
    parser.add_argument('--max-videos', type=int, default=5, help='Max videos per channel')
    parser.add_argument('--max-comments', type=int, default=50, help='Max comments per video')
    parser.add_argument('--test', action='store_true', help='Run in test mode (minimal collection)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key
    if not api_key and args.api_key_file:
        with open(args.api_key_file, 'r') as f:
            api_key = f.read().strip()
    
    if not api_key:
        api_key = os.environ.get('YOUTUBE_API_KEY')
    
    if not api_key:
        print("ERROR: No API key provided!")
        print("Please provide an API key via:")
        print("  --api-key YOUR_KEY")
        print("  --api-key-file path/to/key.txt")
        print("  YOUTUBE_API_KEY environment variable")
        return 1
    
    # Initialize collector
    collector = YouTubeCollector(
        api_key=api_key,
        output_dir=args.output_dir
    )
    
    # Test mode
    if args.test:
        print("\n" + "="*60)
        print("RUNNING IN TEST MODE")
        print("="*60 + "\n")
        
        # Test 1: Search for channels
        print("Test 1: Searching for Hebrew cooking channels...")
        channels = collector.search_channels("מתכונים בישול", max_results=3)
        print(f"Found {len(channels)} channels")
        for ch in channels:
            print(f"  - {ch['title']} (ID: {ch['channel_id']}, Videos: {ch['video_count']})")
        
        if channels:
            # Test 2: Get videos from first channel
            test_channel = channels[0]
            print(f"\nTest 2: Getting videos from '{test_channel['title']}'...")
            videos = list(collector.get_channel_videos(test_channel['channel_id'], max_videos=3))
            print(f"Found {len(videos)} videos")
            for v in videos:
                print(f"  - {v['title'][:50]}...")
            
            if videos:
                # Test 3: Get comments from first video
                test_video = videos[0]
                print(f"\nTest 3: Getting comments from '{test_video['title'][:30]}...'")
                comments = list(collector.get_video_comments(
                    video_id=test_video['video_id'],
                    video_title=test_video['title'],
                    channel_id=test_channel['channel_id'],
                    channel_title=test_channel['title'],
                    max_comments=10
                ))
                print(f"Found {len(comments)} comments")
                
                # Filter and show Hebrew comments
                hebrew_comments = [c for c in comments if HebrewFilter.is_valid_comment(c.text, min_words=3)]
                print(f"Hebrew comments (min 3 words): {len(hebrew_comments)}")
                
                for i, c in enumerate(hebrew_comments[:5]):
                    print(f"\n  Comment {i+1} (likes: {c.like_count}):")
                    print(f"    {c.text[:100]}{'...' if len(c.text) > 100 else ''}")
                
                # Save test results
                if hebrew_comments:
                    collector.save_comments(hebrew_comments, "test_comments.jsonl")
        
        collector.print_stats()
        print("\nTEST COMPLETED SUCCESSFULLY!")
        return 0
    
    # Search mode
    if args.search_query:
        print(f"\nSearching for channels matching: {args.search_query}")
        channels = collector.search_channels(args.search_query, max_results=10)
        print(f"\nFound {len(channels)} channels:")
        for ch in channels:
            print(f"  - {ch['title']}")
            print(f"    ID: {ch['channel_id']}")
            print(f"    Videos: {ch['video_count']}, Subscribers: {ch['subscriber_count']:,}")
            print()
        return 0
    
    # Single video mode
    if args.video_id:
        print(f"\nCollecting comments from video: {args.video_id}")
        comments = list(collector.get_video_comments(
            video_id=args.video_id,
            video_title="Unknown",
            channel_id="Unknown",
            channel_title="Unknown",
            max_comments=args.max_comments
        ))
        hebrew_comments = [c for c in comments if HebrewFilter.is_valid_comment(c.text)]
        print(f"Collected {len(hebrew_comments)} Hebrew comments")
        collector.save_comments(hebrew_comments, f"video_{args.video_id}_comments.jsonl")
        collector.print_stats()
        return 0
    
    # Channel mode
    if args.channel_id:
        print(f"\nCollecting from channel: {args.channel_id}")
        comments = collector.collect_from_channel(
            channel_id=args.channel_id,
            max_videos=args.max_videos,
            max_comments_per_video=args.max_comments
        )
        collector.save_comments(comments, f"channel_{args.channel_id}_comments.jsonl")
        collector.save_stats()
        collector.print_stats()
        return 0
    
    print("No action specified. Use --test, --search-query, --video-id, or --channel-id")
    return 1


if __name__ == "__main__":
    exit(main())
