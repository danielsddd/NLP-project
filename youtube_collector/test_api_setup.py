#!/usr/bin/env python3
"""
Quick Test Script for YouTube API Setup
=======================================
Group 11: Daniel Simanovsky & Roei Ben Artzi

Run this script to verify your YouTube API key is working correctly.
This uses minimal API quota (just a few calls).

Usage:
    python test_api_setup.py YOUR_API_KEY
    
    OR
    
    export YOUTUBE_API_KEY=YOUR_API_KEY
    python test_api_setup.py
"""

import sys
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def test_api_connection(api_key: str) -> bool:
    """Test basic API connectivity."""
    print("\n" + "="*60)
    print("TEST 1: API Connection")
    print("="*60)
    
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        print("✓ Successfully connected to YouTube API")
        return True
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False


def test_search_api(api_key: str) -> bool:
    """Test search API (costs 100 units)."""
    print("\n" + "="*60)
    print("TEST 2: Search API (Hebrew cooking channels)")
    print("="*60)
    
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # Search for Hebrew cooking content
        request = youtube.search().list(
            part='snippet',
            q='מתכונים בישול',  # "Recipes cooking" in Hebrew
            type='channel',
            maxResults=3,
            regionCode='IL'
        )
        response = request.execute()
        
        if response.get('items'):
            print(f"✓ Found {len(response['items'])} channels:")
            for item in response['items']:
                title = item['snippet']['title']
                channel_id = item['snippet']['channelId']
                print(f"  - {title}")
                print(f"    Channel ID: {channel_id}")
            return True
        else:
            print("✗ No results found (unusual, but API worked)")
            return True
            
    except HttpError as e:
        if 'quotaExceeded' in str(e):
            print("✗ API quota exceeded - wait until tomorrow or use a different key")
        elif 'forbidden' in str(e).lower():
            print("✗ API key invalid or YouTube Data API not enabled")
            print("  Go to: https://console.cloud.google.com/apis/library/youtube.googleapis.com")
        else:
            print(f"✗ API error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_video_search(api_key: str) -> dict:
    """Test video search and return a video ID for further testing."""
    print("\n" + "="*60)
    print("TEST 3: Video Search")
    print("="*60)
    
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # Search for a Hebrew cooking video
        request = youtube.search().list(
            part='snippet',
            q='מתכון עוגה',  # "Cake recipe" in Hebrew
            type='video',
            maxResults=1,
            regionCode='IL'
        )
        response = request.execute()
        
        if response.get('items'):
            item = response['items'][0]
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            channel = item['snippet']['channelTitle']
            channel_id = item['snippet']['channelId']
            
            print(f"✓ Found video:")
            print(f"  Title: {title}")
            print(f"  Video ID: {video_id}")
            print(f"  Channel: {channel}")
            
            return {
                'video_id': video_id,
                'title': title,
                'channel_id': channel_id,
                'channel_title': channel
            }
        else:
            print("✗ No videos found")
            return None
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_comments_api(api_key: str, video_id: str) -> bool:
    """Test comments API (costs 1 unit per call)."""
    print("\n" + "="*60)
    print(f"TEST 4: Comments API (video: {video_id})")
    print("="*60)
    
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=5,
            textFormat='plainText'
        )
        response = request.execute()
        
        if response.get('items'):
            print(f"✓ Found {len(response['items'])} comment threads:")
            
            hebrew_count = 0
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                text = comment['textDisplay']
                likes = comment.get('likeCount', 0)
                
                # Check for Hebrew
                import re
                has_hebrew = bool(re.search(r'[\u0590-\u05FF]', text))
                if has_hebrew:
                    hebrew_count += 1
                
                # Show first 80 chars
                display_text = text[:80] + ('...' if len(text) > 80 else '')
                print(f"\n  [{likes} likes] {'[HE]' if has_hebrew else '[--]'}")
                print(f"  {display_text}")
            
            print(f"\n✓ Hebrew comments found: {hebrew_count}/{len(response['items'])}")
            return True
        else:
            print("✓ No comments found (video may have comments disabled)")
            return True
            
    except HttpError as e:
        if 'commentsDisabled' in str(e):
            print("✓ Comments are disabled for this video (API works, just no comments)")
            return True
        else:
            print(f"✗ API error: {e}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_channel_api(api_key: str, channel_id: str) -> bool:
    """Test channel API."""
    print("\n" + "="*60)
    print(f"TEST 5: Channel API (channel: {channel_id})")
    print("="*60)
    
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        request = youtube.channels().list(
            part='snippet,statistics',
            id=channel_id
        )
        response = request.execute()
        
        if response.get('items'):
            item = response['items'][0]
            title = item['snippet']['title']
            subs = int(item['statistics'].get('subscriberCount', 0))
            videos = int(item['statistics'].get('videoCount', 0))
            
            print(f"✓ Channel info:")
            print(f"  Name: {title}")
            print(f"  Subscribers: {subs:,}")
            print(f"  Videos: {videos}")
            return True
        else:
            print("✗ Channel not found")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def print_quota_info():
    """Print information about API quota."""
    print("\n" + "="*60)
    print("API QUOTA INFORMATION")
    print("="*60)
    print("""
YouTube Data API v3 has a daily quota of 10,000 units (free tier).

Quota costs:
  - search.list:          100 units per call
  - commentThreads.list:  1 unit per call
  - channels.list:        1 unit per call
  - videos.list:          1 unit per call

This test script uses approximately:
  - Test 2 (search channels): 100 units
  - Test 3 (search videos):   100 units
  - Test 4 (get comments):    1 unit
  - Test 5 (channel info):    1 unit
  - TOTAL: ~202 units

For collecting 3,000 comments, you'll need:
  - ~30-50 channel searches (3,000-5,000 units)
  - ~100-200 video comment fetches (100-200 units)
  - TOTAL: ~3,200-5,200 units per day

Strategy: Collect over 1-2 days to stay within quota.
    """)


def main():
    """Run all tests."""
    # Get API key
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = os.environ.get('YOUTUBE_API_KEY')
    
    if not api_key:
        print("ERROR: No API key provided!")
        print("\nUsage:")
        print("  python test_api_setup.py YOUR_API_KEY")
        print("\nOr set environment variable:")
        print("  export YOUTUBE_API_KEY=YOUR_API_KEY")
        print("  python test_api_setup.py")
        return 1
    
    print("\n" + "="*60)
    print("YOUTUBE API SETUP TEST")
    print("Group 11: Daniel Simanovsky & Roei Ben Artzi")
    print("="*60)
    print(f"\nAPI Key: {api_key[:10]}...{api_key[-4:]}")
    
    # Run tests
    results = {}
    
    # Test 1: Connection
    results['connection'] = test_api_connection(api_key)
    if not results['connection']:
        print("\n✗ Cannot proceed - API connection failed")
        return 1
    
    # Test 2: Search channels
    results['search'] = test_search_api(api_key)
    
    # Test 3: Search videos
    video_info = test_video_search(api_key)
    results['video_search'] = video_info is not None
    
    # Test 4: Comments (if we found a video)
    if video_info:
        results['comments'] = test_comments_api(api_key, video_info['video_id'])
        
        # Test 5: Channel info
        results['channel'] = test_channel_api(api_key, video_info['channel_id'])
    else:
        results['comments'] = False
        results['channel'] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:15} {status}")
        if not passed:
            all_passed = False
    
    # Print quota info
    print_quota_info()
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED - API IS READY TO USE!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run the full collector with --test flag:")
        print("     python src/youtube_collector.py --api-key YOUR_KEY --test")
        print("\n  2. Search for cooking channels:")
        print("     python src/youtube_collector.py --api-key YOUR_KEY --search-query 'מתכונים'")
        print("\n  3. Collect from a specific channel:")
        print("     python src/youtube_collector.py --api-key YOUR_KEY --channel-id UC... --max-videos 10")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Check errors above")
        return 1


if __name__ == "__main__":
    exit(main())
