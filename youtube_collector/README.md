# YouTube Comment Collector for Recipe Modification Extraction

**Group 11: Daniel Simanovsky & Roei Ben Artzi**  
**NLP 2025a, Tel Aviv University**

This module collects Hebrew comments from YouTube cooking channels for the purpose of extracting recipe modifications.

## Prerequisites

1. **Python 3.8+**
2. **YouTube Data API v3 key** (free)

## Quick Start

### Step 1: Get YouTube API Key (One-Time Setup)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Go to **APIs & Services** → **Library**
4. Search for **"YouTube Data API v3"** and click **Enable**
5. Go to **APIs & Services** → **Credentials**
6. Click **Create Credentials** → **API Key**
7. Copy your API key

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Test Your Setup

```bash
# Replace YOUR_API_KEY with your actual key
python test_api_setup.py YOUR_API_KEY
```

You should see output like:
```
============================================================
YOUTUBE API SETUP TEST
============================================================
✓ Successfully connected to YouTube API
✓ Found 3 channels
✓ Found video: עוגת שוקולד...
✓ Found 5 comment threads
✓ ALL TESTS PASSED - API IS READY TO USE!
```

### Step 4: Collect Comments

```bash
# Test mode (minimal collection, ~5 comments)
python src/youtube_collector.py --api-key YOUR_KEY --test

# Search for Hebrew cooking channels
python src/youtube_collector.py --api-key YOUR_KEY --search-query "מתכונים בישול"

# Collect from a specific channel
python src/youtube_collector.py --api-key YOUR_KEY --channel-id UC... --max-videos 10

# Collect from a specific video
python src/youtube_collector.py --api-key YOUR_KEY --video-id dQw4w9WgXcQ --max-comments 50
```

## API Quota Management

## Output Format

Comments are saved to `data/raw_youtube/comments.jsonl`:

```json
{"comment_id": "Ugw123...", "video_id": "abc123", "video_title": "עוגת שוקולד מושלמת", "channel_id": "UC...", "channel_title": "השף הביתי", "text": "הוספתי קצת יותר סוכר והיה מעולה!", "author": "משתמש", "author_channel_id": "UC...", "like_count": 15, "published_at": "2024-03-15T10:30:00Z", "updated_at": "2024-03-15T10:30:00Z", "reply_count": 2, "is_reply": false, "parent_id": null}
```

## Finding Channel IDs

Channel IDs are needed to collect comments. Here's how to find them:

### Method 1: Use the Search Feature
```bash
python src/youtube_collector.py --api-key YOUR_KEY --search-query "בישול ישראלי"
```

### Method 2: From Channel URL
- If URL is `youtube.com/channel/UCxxxxxx`, the ID is `UCxxxxxx`
- If URL is `youtube.com/@username`, use the search feature to find the ID

### Method 3: Browser Developer Tools
1. Go to the channel page
2. Press F12 (Developer Tools)
3. Search for `channelId` in the page source

## Known Hebrew Cooking Channels

Here are some channels to get you started (you'll need to find their IDs):

- דנילה גרמון (baking)
- השף הלבן (professional chef)
- מתכונים לכל מצב (home cooking)
- אפייה בריאה (healthy baking)

## Troubleshooting

### "API quota exceeded"
- Wait until midnight Pacific Time (10 AM Israel time)
- Or create a new Google Cloud project with a fresh quota

### "Comments disabled"
- Some videos have comments disabled
- The collector handles this gracefully and moves on

### "No Hebrew comments found"
- Try different channels/videos
- Lower the `min_words` filter
- Set `require_hebrew=False` for testing

### "Connection error"
- Check your internet connection
- Verify the API key is correct
- Ensure YouTube Data API is enabled in Google Cloud Console

## Files

```
youtube_collector/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── test_api_setup.py            # Quick API test script
├── src/
│   └── youtube_collector.py     # Main collector module
└── data/
    └── raw_youtube/             # Output directory (created automatically)
        ├── comments.jsonl       # Collected comments
        └── collection_stats.json # Collection statistics
```

## Support

For issues specific to this project, contact the group members.
