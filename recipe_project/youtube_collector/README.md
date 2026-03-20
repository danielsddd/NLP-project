# YouTube Thread Collector v4.1

**Group 11: Daniel Simanovsky & Roei Ben Artzi**  
**NLP 2025a - Recipe Modification Extraction**

Collects Hebrew comment **threads** (top comment + replies) from YouTube cooking channels. Thread-level collection enables the teacher model to see question→answer context for better labeling.

## Files Structure

```
youtube_collector/
├── collect.py          # Main script (the only code you run)
├── config.yaml         # Settings: filtering, API limits, output paths
├── channels.yaml       # Channel list: add your channels here
└── README.md           # This file
```

Output goes to:
```
data/raw_youtube/
├── threads.jsonl              # All collected threads
├── threads_cooking_only.jsonl # After channel verification filter
├── channels_report.csv        # Channel verification spreadsheet
└── collection_stats.json      # Run statistics
```

## Quick Start

### 1. Install dependencies

```bash
pip install google-api-python-client pyyaml python-dotenv
```

### 2. Test your API key

```bash
python collect.py --api-key YOUR_KEY --test
```

### 3. Discover Hebrew cooking channels

```bash
python collect.py --api-key YOUR_KEY --discover-all
```

This runs 10 Hebrew cooking queries and writes results to `channels.yaml`.

### 4. Verify channels in `channels.yaml`

Open each channel URL, confirm it's a cooking channel, set `active: false` for non-cooking:

```yaml
channels:
  - name: "שם הערוץ"
    id: "UC_PASTE_ID_HERE"
    active: true
    category: "cooking"
```

### 5. Collect threads

```bash
python collect.py --api-key YOUR_KEY --collect
```

### 6. (Optional) Post-collection channel verification

```bash
# Generate channel report CSV
python collect.py --list-channels

# Filter to cooking-only after reviewing CSV
python collect.py --filter-channels --csv data/raw_youtube/channels_report.csv
```

## Thread JSONL Format

Each line in `threads.jsonl` is one JSON object:

```json
{
  "thread_id": "UgwXXX",
  "video_id": "abc123",
  "video_title": "לחם כוסמין",
  "channel_id": "UCc0z...",
  "channel_title": "חן במטבח",
  "top_comment": {
    "comment_id": "UgwXXX",
    "text": "אפשר במקום קמח רגיל להשתמש בכוסמין?",
    "like_count": 3
  },
  "replies": [
    {
      "comment_id": "UgwYYY",
      "text": "כן בטח, אותה כמות",
      "like_count": 1,
      "is_creator": false
    }
  ],
  "has_creator_reply": false,
  "total_likes": 4,
  "collected_at": "2026-03-18T14:30:00Z"
}
```

### Key Fields

| Field | Purpose |
|-------|---------|
| `thread_id` | Deduplication + matching to teacher labels |
| `top_comment.text` | Main comment text (input for teacher) |
| `replies[].text` | Reply texts (teacher extracts mods from replies to questions) |
| `replies[].is_creator` | Creator replies have highest credibility |
| `has_creator_reply` | Quick filter for high-value threads |
| `total_likes` | Social signal for ranking module |
| `channel_id` | Per-channel analysis in paper |

## Why Threads (Not Flat Comments)

Many recipe modifications appear as question→answer pairs:

```
Comment: "אפשר במקום חרדל גרגירי את הסוג החלק?"   ← QUESTION
Reply:   "כן בטח, אותה כמות"                       ← ANSWER (the actual mod)
```

The teacher model sees the full thread and extracts from the **reply**, not the question. A question with no meaningful reply is discarded (`has_modification: false`).

## Filtering Rules

- At least one Hebrew character in thread
- Top comment >= 3 words
- No spam keywords (http, www, subscribe, etc.)
- Skips creator's own top-level posts (usually recipe intro/pin)
- Keeps creator replies (valuable confirmation signal)
- Replies also filtered for Hebrew and spam

## API Rate Limits

YouTube API free tier: 10,000 units/day. Uses `playlistItems` (cheaper) instead of `search` for video listing. The script auto-resumes if interrupted (tracks seen `thread_id`s).

## Commands Reference

| Command | Description |
|---------|-------------|
| `--test` | Test API connection |
| `--discover-all` | Search for channels using 10 Hebrew cooking queries |
| `--collect` | Collect threads from all active channels |
| `--collect --target N` | Collect up to N threads |
| `--list-channels` | Generate channels_report.csv |
| `--filter-channels --csv FILE` | Filter threads to cooking-only channels |

### Examples

```bash
# Test API
python collect.py --api-key YOUR_KEY --test

# Full discovery
python collect.py --api-key YOUR_KEY --discover-all

# Collect 5000 threads
python collect.py --api-key YOUR_KEY --collect --target 5000

# Generate channel report for verification
python collect.py --list-channels

# Filter to cooking-only after reviewing CSV
python collect.py --filter-channels --csv data/raw_youtube/channels_report.csv
```

## Collection Stats

After collection, `collection_stats.json` contains:

```json
{
  "channels_processed": 35,
  "videos_processed": 1247,
  "threads_found": 5231,
  "threads_kept": 5012,
  "threads_filtered": 219,
  "api_calls": 1589
}
```