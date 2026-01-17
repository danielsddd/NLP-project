# YouTube Comment Collector v2

**Group 11: Daniel Simanovsky & Roei Ben Artzi**  
**NLP 2025a - Recipe Modification Extraction**

A clean, modular YouTube comment collector with configuration files for easy customization.

## 📁 Files Structure

```
youtube_collector_v2/
├── collect.py          # Main script (the only code you run)
├── config.yaml         # Settings: filtering, API limits, output paths
├── channels.yaml       # Channel list: add your channels here
├── requirements.txt    # Python dependencies
└── data/
    └── raw_youtube/    # Output directory (created automatically)
        ├── comments.jsonl
        └── collection_stats.json
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Test your API key

```bash
python collect.py --api-key YOUR_KEY --test
```

### 3. Find Hebrew cooking channels

```bash
python collect.py --api-key YOUR_KEY --discover "מתכונים בישול"
```

### 4. Add channels to `channels.yaml`

```yaml
channels:
  - name: "שם הערוץ"
    id: "UC_PASTE_ID_HERE"      # From discover output
    active: true
    category: "cooking"
```

### 5. Collect comments

```bash
python collect.py --api-key YOUR_KEY --collect
```

## 📝 What Data Is Collected?

**Only what's needed for the NLP project:**

| Field | Example | Why We Need It |
|-------|---------|----------------|
| `text` | `"הוספתי יותר סוכר והיה מעולה"` | **Main input for extraction** |
| `like_count` | `15` | **Ranking module** (social signal) |
| `video_title` | `"עוגת שוקולד מושלמת"` | Context: which recipe |
| `channel_title` | `"השף הביתי"` | Context: which channel |
| `video_id` | `"abc123"` | Group comments by recipe |
| `comment_id` | `"Ugw..."` | Deduplication only |
| `word_count` | `5` | Pre-computed for filtering |
| `has_modification_keyword` | `true` | Pre-detected keywords |
| `detected_keywords` | `["הוספתי", "יותר"]` | Which keywords found |

**NOT collected (unnecessary for NLP):** Author info, timestamps, reply relationships

## ⚙️ Configuration Files

### `config.yaml` - Settings

Edit this to change:
- **Output paths**: Where files are saved
- **Collection limits**: Max videos, comments per video
- **Filtering**: Minimum words, Hebrew requirement, spam keywords
- **API settings**: Retry limits, delays

### `channels.yaml` - Channel List

Edit this to:
- Add new channels (set `active: true`)
- Disable channels (set `active: false`)
- Organize by category

## 📋 Commands Reference

| Command | Description |
|---------|-------------|
| `--test` | Test API connection |
| `--discover "query"` | Search for channels |
| `--channel UC...` | Collect from one channel |
| `--video VIDEO_ID` | Collect from one video |
| `--collect` | Collect from all active channels |

### Examples

```bash
# Test API
python collect.py --api-key YOUR_KEY --test

# Search for baking channels
python collect.py --api-key YOUR_KEY --discover "אפייה ביתית"

# Collect from a specific video (for testing)
python collect.py --api-key YOUR_KEY --video dQw4w9WgXcQ

# Collect from a specific channel
python collect.py --api-key YOUR_KEY --channel UCxxxxxxxxxxxxxxx

# Full collection from all configured channels
python collect.py --api-key YOUR_KEY --collect
```

## 📊 Output Files

### `comments.jsonl`

One JSON object per line:

```json
{"comment_id": "Ugw...", "video_id": "abc", "text": "הוספתי יותר סוכר והיה מעולה", "like_count": 15, "reply_count": 2, "has_modification_keyword": true, "detected_keywords": ["הוספתי", "יותר"], ...}
```

### `collection_stats.json`

Statistics from the collection run:

```json
{
  "channels_processed": 5,
  "videos_processed": 127,
  "comments_found": 4521,
  "comments_kept": 2847,
  "comments_filtered": 1674,
  "api_calls": 312,
  "filter_reasons": {
    "no_hebrew": 892,
    "too_short": 523,
    "spam": 241,
    "creator_comment": 18
  }
}
```

## 💡 Tips for Efficient Collection

1. **Test with one video first**
   ```bash
   python collect.py --api-key YOUR_KEY --video VIDEO_ID
   ```

2. **Check API quota usage**
   - Go to: Google Cloud Console → APIs & Services → YouTube Data API v3
   - Check "Quotas" tab

3. **Start with fewer videos/comments**
   - Edit `config.yaml`:
   ```yaml
   collection:
     max_videos_per_channel: 10  # Start small
     max_comments_per_video: 50
   ```

4. **Add channels incrementally**
   - Discover → Add 1-2 channels → Test → Add more

## 🔧 Troubleshooting

### "No comments found" / Empty output
- Check if video has comments enabled
- Try a different video/channel
- Reduce `min_words` in config.yaml

### "API quota exceeded"
- Wait until midnight Pacific Time
- Or create a new Google Cloud project

### Comments not in Hebrew
- Ensure `require_hebrew: true` in config.yaml
- Try channels with more Hebrew speakers

## 📈 Quota Estimation

| Operation | Cost | For 3,000 comments |
|-----------|------|-------------------|
| Search channels | 100 | ~500 (one-time) |
| Get videos | 100 | ~2,000 |
| Get comments | 1 | ~500 |

Daily quota: 10,000 units → Easily achievable in one day!

---