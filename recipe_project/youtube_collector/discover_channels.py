"""
Automated Hebrew Cooking Channel Discovery
==========================================
Group 11: Daniel Simanovsky & Roei Ben Artzi
NLP 2025a - Recipe Modification Extraction

Runs all discovery queries automatically, filters for Hebrew channels only,
and writes a ready-to-use channels.yaml.

Usage:
    python discover_channels.py --api-key YOUR_KEY
    python discover_channels.py --api-key YOUR_KEY --min-subs 5000
    python discover_channels.py --api-key YOUR_KEY --output my_channels.yaml
"""

import os
import sys
import re
import time
import argparse
import yaml
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# =============================================================================
# DISCOVERY QUERIES
# Dish-specific queries surface channels whose comment sections contain
# exactly the modification language we need for the NLP task.
# =============================================================================

DISCOVERY_QUERIES = [
    # Meat & chicken (highest modification comment rate)
    "אסאדו מתכון",
    "שיפודי פרגיות מתכון",
    "שווארמה ביתית מתכון",
    "שניצל ביתי מתכון",
    "קציצות בשר מתכון",
    "נאגטס ביתי מתכון",
    "לזניה ביתית מתכון",

    # Bread & dough (bakers always adjust quantities/techniques)
    "לחמניות באן מתכון",
    "לחם ביתי מתכון",
    "בצק שמרים מתכון",
    "סינבון ביתי מתכון",
    "פיתה ביתית מתכון",
    "בייגלה פרצל מתכון",
    "לחם מחמצת מתכון",

    # Desserts & pastry (precise = people always report what they changed)
    "סופגניות ביתיות מתכון",
    "פחזניות מתכון",
    "עוגת שוקולד מתכון",
    "טירמיסו ביתי מתכון",
    "מלבי מתכון",
    "קראמבל תפוחים מתכון",
    "עוגת גבינה מתכון",

    # First courses & sides
    "חומוס ביתי מתכון",
    "פנקייק ביתי מתכון",
    "כדורי פירה מתכון",
    "צ'יפס ביתי מתכון",

    # Generic sweep
    "מתכונים ביתיים ישראלי",
    "אפייה ביתית מתכונים",
    "בישול ישראלי מתכונים",
    "מתכונים לשבת",
    "שף ביתי ישראלי",
]

# =============================================================================
# FILTERS
# =============================================================================

MIN_SUBSCRIBERS = 10_000   # Skip tiny channels
MIN_VIDEOS      = 30       # Must have enough content
MAX_PER_QUERY   = 10       # Results per query

HEBREW_RE = re.compile(r'[\u0590-\u05FF]')


def contains_hebrew(text: str) -> bool:
    return bool(HEBREW_RE.search(text or ""))


def is_hebrew_channel(channel: dict) -> bool:
    """
    A channel is considered Hebrew if its name OR description
    contains Hebrew characters.
    """
    return (
        contains_hebrew(channel.get("name", "")) or
        contains_hebrew(channel.get("description", ""))
    )


def passes_size_filter(channel: dict, min_subs: int) -> bool:
    return (
        channel.get("subscribers", 0) >= min_subs and
        channel.get("videos", 0) >= MIN_VIDEOS
    )


# =============================================================================
# YOUTUBE API HELPERS
# =============================================================================

def build_youtube(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def search_channels(youtube, query: str, max_results: int = MAX_PER_QUERY) -> list:
    """Search YouTube for channels matching a query."""
    try:
        response = youtube.search().list(
            part="snippet",
            q=query,
            type="channel",
            maxResults=max_results,
            regionCode="IL"
        ).execute()
        return [item["snippet"]["channelId"] for item in response.get("items", [])]
    except HttpError as e:
        print(f"    ⚠️  API error for '{query}': {e.resp.status}")
        return []


def get_channel_details(youtube, channel_ids: list) -> list:
    """Fetch full details for up to 50 channel IDs at once."""
    if not channel_ids:
        return []
    try:
        response = youtube.channels().list(
            part="snippet,statistics",
            id=",".join(channel_ids[:50])
        ).execute()
    except HttpError as e:
        print(f"    ⚠️  Error fetching details: {e.resp.status}")
        return []

    results = []
    for item in response.get("items", []):
        stats   = item.get("statistics", {})
        snippet = item.get("snippet", {})
        results.append({
            "id":          item["id"],
            "name":        snippet.get("title", "Unknown"),
            "description": snippet.get("description", "")[:200],
            "subscribers": int(stats.get("subscriberCount", 0)),
            "videos":      int(stats.get("videoCount", 0)),
        })
    return results


def guess_category(channel: dict) -> str:
    text = (channel["name"] + " " + channel["description"]).lower()
    if any(k in text for k in ["אפיי", "עוגה", "עוגות", "קינוח", "סופגני"]):
        return "baking"
    if any(k in text for k in ["בשר", "גריל", "שיפוד", "שווארמה"]):
        return "meat"
    if any(k in text for k in ["לחם", "בצק", "שמרים"]):
        return "bread"
    if any(k in text for k in ["בריא", "דיאט", "טבעוני"]):
        return "healthy"
    if any(k in text for k in ["שף", "מסעדה"]):
        return "chef"
    return "cooking"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Discover Hebrew cooking channels")
    parser.add_argument("--api-key",  required=True, help="YouTube Data API v3 key")
    parser.add_argument("--output",   default="channels.yaml", help="Output file (default: channels.yaml)")
    parser.add_argument("--min-subs", type=int, default=MIN_SUBSCRIBERS,
                        help=f"Minimum subscribers (default: {MIN_SUBSCRIBERS:,})")
    args = parser.parse_args()

    youtube = build_youtube(args.api_key)

    print("=" * 60)
    print("  HEBREW COOKING CHANNEL DISCOVERY")
    print(f"  {len(DISCOVERY_QUERIES)} queries | min {args.min_subs:,} subscribers")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Phase 1: Collect all unique channel IDs across all queries
    # -------------------------------------------------------------------------
    all_channel_ids = set()

    for i, query in enumerate(DISCOVERY_QUERIES, 1):
        print(f"[{i:02d}/{len(DISCOVERY_QUERIES)}] '{query}'", end=" ... ")
        ids  = search_channels(youtube, query)
        new  = set(ids) - all_channel_ids
        all_channel_ids.update(ids)
        print(f"{len(new)} new  (total: {len(all_channel_ids)})")
        time.sleep(0.5)   # stay well within quota

    print(f"\n📋 Total unique channel IDs: {len(all_channel_ids)}")

    # -------------------------------------------------------------------------
    # Phase 2: Fetch full details in batches of 50
    # -------------------------------------------------------------------------
    print("\n📡 Fetching channel details ...")
    all_ids   = list(all_channel_ids)
    all_details = []

    for i in range(0, len(all_ids), 50):
        batch   = all_ids[i:i + 50]
        details = get_channel_details(youtube, batch)
        all_details.extend(details)
        print(f"   Fetched {min(i + 50, len(all_ids))}/{len(all_ids)}")
        time.sleep(0.5)

    # -------------------------------------------------------------------------
    # Phase 3: Filter — Hebrew only + size threshold
    # -------------------------------------------------------------------------
    hebrew_channels = [
        ch for ch in all_details
        if is_hebrew_channel(ch) and passes_size_filter(ch, args.min_subs)
    ]
    hebrew_channels.sort(key=lambda x: x["subscribers"], reverse=True)

    print(f"\n✅ Hebrew channels passing filter: {len(hebrew_channels)}")
    print(f"   (removed {len(all_details) - len(hebrew_channels)} non-Hebrew / too small)\n")

    # -------------------------------------------------------------------------
    # Phase 4: Print results table
    # -------------------------------------------------------------------------
    print("=" * 60)
    print(f"  {'CHANNEL NAME':<35} {'SUBS':>10}  CATEGORY")
    print("=" * 60)
    for ch in hebrew_channels:
        cat = guess_category(ch)
        print(f"  {ch['name']:<35} {ch['subscribers']:>10,}  {cat}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Phase 5: Write channels.yaml
    # -------------------------------------------------------------------------
    channel_entries = []
    for ch in hebrew_channels:
        channel_entries.append({
            "name":     ch["name"],
            "id":       ch["id"],
            "active":   True,
            "category": guess_category(ch),
            "notes":    f"{ch['subscribers']:,} subscribers | {ch['videos']} videos",
        })

    # Add placeholder for manual additions
    channel_entries.append({
        "name":     "Placeholder - add manually",
        "id":       "UC_REPLACE_WITH_REAL_ID",
        "active":   False,
        "category": "cooking",
        "notes":    "Add more channels here manually if needed",
    })

    output = {
        "channels": channel_entries,
        "discovery_queries": DISCOVERY_QUERIES,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        yaml.dump(output, f, allow_unicode=True,
                  default_flow_style=False, sort_keys=False)

    print(f"\n💾 Saved {len(hebrew_channels)} channels → {args.output}")
    print(f"\n📌 Next step:")
    print(f"   1. Review {args.output} — set active: false for any irrelevant channels")
    print(f"   2. Run: python collect.py --api-key YOUR_KEY --collect")
    print("=" * 60)


if __name__ == "__main__":
    main()