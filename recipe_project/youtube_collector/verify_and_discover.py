"""
Channel Verifier + Discovery
=============================
Group 11: Daniel Simanovsky & Roei Ben Artzi
NLP 2025a - Recipe Modification Extraction

Does two things:
1. VERIFIES existing channels by fetching their 5 most recent video titles
   and checking if they contain recipe/cooking keywords
2. DISCOVERS new Hebrew cooking channels using dish-specific queries

Outputs:
  - channels_verified.yaml  — verified existing channels
  - channels_new.yaml       — newly discovered channels
  - verification_report.txt — human-readable report for manual review

Usage:
    python verify_and_discover.py --api-key YOUR_KEY
    python verify_and_discover.py --api-key YOUR_KEY --verify-only
    python verify_and_discover.py --api-key YOUR_KEY --discover-only
"""

import os
import sys
import time
import argparse
import yaml
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# =============================================================================
# EXISTING CHANNELS TO VERIFY
# =============================================================================

EXISTING_CHANNELS = [
    {"name": "kobi edri's food",              "id": "UCaOPTGQ72rVCOrT7xHV9k5Q", "category": "cooking"},
    {"name": "BigDen תוכנית הבישול שלי",      "id": "UCLmJy4v0T9qNX2W7Gw5sTUg", "category": "cooking"},
    {"name": "רועי דהאן - Roy Dahan",          "id": "UC4cW1Q6mNp-Qzde5V5qHaDQ", "category": "cooking"},
    {"name": "שחר חן פודיק - Foodik",          "id": "UCboRgkGd3lVVfH4TEy0S9Ug", "category": "cooking"},
    {"name": "מפלצת העוגיות-טוני משיח",        "id": "UC-Ace2cJTSi5wiUOZcaWchw", "category": "cooking"},
    {"name": "Shiran matetyahu שירן מתתיהו",   "id": "UCRgcsgzlb2j8a1LefW7TRsQ", "category": "baking"},
    {"name": "פינת האוכל של סיון",             "id": "UCnR3T6_yRxYwtPYvrCtlXmQ", "category": "cooking"},
    {"name": "חן במטבח",                       "id": "UCc0zfbjbWVBsjeMCp4ZRh6Q", "category": "cooking"},
    {"name": "נעמה קדוש naama kadosh",          "id": "UCZ3i-27X5eF4IUdbJmBaGgA", "category": "cooking"},
    {"name": "אור שפיץ - העוגות של אור",       "id": "UCdyE8cULjLr4pxNcQfkTvrA", "category": "baking"},
    {"name": "The Cooking Foodie - Israel",    "id": "UCau6yGBJgwDHZhjyQ48Zgew", "category": "baking"},
    {"name": "Ben Shai",                       "id": "UChmK8Hq3O7Ugkt0ZVn-CVEA", "category": "cooking"},
    {"name": "בקלי קלות ליהי קרויץ",           "id": "UCYRjRbFiLOyKr89dryIDobw", "category": "baking"},
    {"name": "פודי - Foody",                   "id": "UCy_lqFqTpf7HTiv3nNT2SxQ", "category": "cooking"},
    {"name": "itay edri",                      "id": "UCwyhjk0dj2qTbtIkeqidS0Q", "category": "cooking"},
    {"name": "רובי מיכאל - Rubi Michael",      "id": "UCWkr-jAvT8CX29QYCIn4uDw", "category": "cooking"},
    {"name": "נועם זיגדון",                    "id": "UCigf3LCe_lKjZEcpEsfapdg", "category": "cooking"},
    {"name": "קורל חוטה",                      "id": "UC3e3v-BQixdL5j-kmLNR9Vg", "category": "cooking"},
    {"name": "הפרוייקט של בוזגלו",             "id": "UCjBZdwHHCJqcPNT4r4PkLsw", "category": "cooking"},
    {"name": "לבשל עם שרה",                    "id": "UCb1kyb-5hXmc2cBN51ShltA", "category": "cooking"},
    {"name": "נטלי לוין",                      "id": "UC24zeKglSXg7HY7BGQvbXGw", "category": "baking"},
    {"name": "foodsdictionary",                "id": "UC38zXk8d0yVsHWvmeoAVcgw", "category": "healthy"},
    {"name": "נלי ימפולסקי Nelly Yampolski",   "id": "UC_FT2QK2J7YRGzOZIvRKgnA", "category": "cooking"},
    {"name": "לינוי שטרית",                    "id": "UCMe6whTcQlhhC5uOGCXncdg", "category": "cooking"},
    {"name": "עידיתוש מתכונים מומלצים",        "id": "UCOq9GpfT6HDl1dxffiN8Liw", "category": "cooking"},
]

# =============================================================================
# DISCOVERY QUERIES — dish specific for maximum relevance
# =============================================================================

DISCOVERY_QUERIES = [
    # Meat
    "אסאדו מתכון",
    "שיפודי פרגיות מתכון",
    "שווארמה ביתית מתכון",
    "שניצל ביתי מתכון",
    "קציצות בשר מתכון",
    "נאגטס ביתי מתכון",
    "לזניה ביתית מתכון",
    # Bread
    "לחמניות באן מתכון",
    "לחם ביתי מתכון",
    "סינבון ביתי מתכון",
    "פיתה ביתית מתכון",
    "לחם מחמצת מתכון",
    # Desserts
    "סופגניות ביתיות מתכון",
    "עוגת שוקולד מתכון",
    "טירמיסו ביתי מתכון",
    "מלבי מתכון",
    "פחזניות מתכון",
    "עוגת גבינה מתכון",
    # Sides
    "חומוס ביתי מתכון",
    "צ'יפס ביתי מתכון",
    # Generic
    "מתכונים ביתיים ישראלי",
    "אפייה ביתית מתכונים",
    "בישול ישראלי מתכונים",
    "מתכונים לשבת",
    "שף ביתי ישראלי",
]

# =============================================================================
# COOKING KEYWORDS — used to score video titles
# =============================================================================

COOKING_KEYWORDS = [
    # Hebrew cooking words
    "מתכון", "מתכונים", "בישול", "אפייה", "מטבח",
    "עוגה", "עוגות", "לחם", "פיצה", "פסטה",
    "עוף", "בשר", "דג", "ירק", "סלט",
    "רוטב", "מרק", "תבשיל", "קינוח", "קינוחים",
    "שוקולד", "גבינה", "ביצה", "קמח", "שמרים",
    "תנור", "מחבת", "סיר", "אופים", "מבשלים",
    "שניצל", "חומוס", "פלאפל", "סביח", "שווארמה",
    "סופגניות", "פחזניות", "טירמיסו", "מלבי",
    "אסאדו", "פרגית", "קציצות", "לזניה",
    # English cooking words (for bilingual channels)
    "recipe", "cook", "bake", "food", "kitchen",
    "cake", "bread", "chicken", "beef", "pasta",
]

MIN_SUBSCRIBERS_NEW = 5_000   # for newly discovered channels
MIN_VIDEOS_NEW      = 20
VIDEOS_TO_CHECK     = 8       # video titles to fetch per channel for verification


# =============================================================================
# API HELPERS
# =============================================================================

def build_youtube(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def api_call(request, retries: int = 3):
    for attempt in range(retries):
        try:
            return request.execute()
        except HttpError as e:
            if e.resp.status == 403 and "quotaExceeded" in str(e):
                print("❌ Quota exceeded!")
                raise
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                raise
    return None


def get_recent_video_titles(youtube, channel_id: str, n: int = VIDEOS_TO_CHECK) -> list:
    """Fetch N most recent video titles from a channel."""
    try:
        response = api_call(youtube.search().list(
            part="snippet",
            channelId=channel_id,
            type="video",
            order="date",
            maxResults=n,
        ))
        return [item["snippet"]["title"] for item in response.get("items", [])]
    except Exception:
        return []


def cooking_score(titles: list) -> tuple:
    """
    Score how cooking-focused a channel is based on video titles.
    Returns (score 0-100, matched_keywords).
    """
    if not titles:
        return 0, []

    all_text   = " ".join(titles).lower()
    matched    = [kw for kw in COOKING_KEYWORDS if kw.lower() in all_text]
    # Score = % of titles containing at least one cooking keyword
    titles_with_kw = sum(
        1 for t in titles
        if any(kw.lower() in t.lower() for kw in COOKING_KEYWORDS)
    )
    score = int(100 * titles_with_kw / len(titles))
    return score, matched


def get_channel_details(youtube, channel_ids: list) -> list:
    """Fetch details for up to 50 channel IDs."""
    if not channel_ids:
        return []
    try:
        response = api_call(youtube.channels().list(
            part="snippet,statistics",
            id=",".join(channel_ids[:50])
        ))
        results = []
        for item in response.get("items", []):
            stats   = item.get("statistics", {})
            snippet = item.get("snippet", {})
            results.append({
                "id":          item["id"],
                "name":        snippet.get("title", "Unknown"),
                "description": snippet.get("description", "")[:150],
                "subscribers": int(stats.get("subscriberCount", 0)),
                "videos":      int(stats.get("videoCount", 0)),
            })
        return results
    except Exception:
        return []


def search_channels(youtube, query: str, max_results: int = 8) -> list:
    """Search for channels by query, return list of channel IDs."""
    try:
        response = api_call(youtube.search().list(
            part="snippet",
            q=query,
            type="channel",
            maxResults=max_results,
            regionCode="IL",
        ))
        return [item["snippet"]["channelId"] for item in response.get("items", [])]
    except Exception:
        return []


# =============================================================================
# STEP 1 — VERIFY EXISTING CHANNELS
# =============================================================================

def verify_existing_channels(youtube) -> tuple:
    """
    Fetch recent video titles for each existing channel and score them.
    Returns (verified_list, rejected_list, report_lines).
    """
    print("\n" + "=" * 65)
    print("  STEP 1: VERIFYING EXISTING CHANNELS")
    print("=" * 65)
    print(f"  Checking {len(EXISTING_CHANNELS)} channels ({VIDEOS_TO_CHECK} videos each)...\n")

    verified = []
    rejected = []
    report   = ["CHANNEL VERIFICATION REPORT", "=" * 65, ""]

    for i, ch in enumerate(EXISTING_CHANNELS, 1):
        print(f"  [{i:02d}/{len(EXISTING_CHANNELS)}] {ch['name'][:45]}", end=" ... ")

        titles = get_recent_video_titles(youtube, ch["id"])
        score, keywords = cooking_score(titles)

        status = "✅ COOKING" if score >= 50 else "⚠️  UNCLEAR" if score >= 25 else "❌ NOT COOKING"
        print(f"{status}  (score: {score}%)")

        report.append(f"{'✅' if score >= 50 else '⚠️ ' if score >= 25 else '❌'} {ch['name']}")
        report.append(f"   Score: {score}% | Keywords found: {', '.join(keywords[:5]) if keywords else 'none'}")
        if titles:
            for t in titles[:3]:
                report.append(f"   - {t[:70]}")
        report.append("")

        ch_entry = {
            "name":           ch["name"],
            "id":             ch["id"],
            "active":         score >= 25,   # keep if at least somewhat cooking
            "category":       ch["category"],
            "cooking_score":  score,
            "notes":          f"Verified: {score}% cooking score | Keywords: {', '.join(keywords[:3])}"
        }

        if score >= 25:
            verified.append(ch_entry)
        else:
            ch_entry["active"] = False
            rejected.append(ch_entry)
            print(f"           Recent titles: {' | '.join(t[:30] for t in titles[:2])}")

        time.sleep(0.4)

    print(f"\n  ✅ Verified (cooking): {len(verified)}")
    print(f"  ❌ Rejected (not cooking): {len(rejected)}")

    return verified, rejected, report


# =============================================================================
# STEP 2 — DISCOVER NEW CHANNELS
# =============================================================================

def discover_new_channels(youtube, existing_ids: set) -> list:
    """
    Run all discovery queries, fetch details, filter, verify.
    Returns list of new verified cooking channels.
    """
    print("\n" + "=" * 65)
    print("  STEP 2: DISCOVERING NEW CHANNELS")
    print(f"  Running {len(DISCOVERY_QUERIES)} queries...")
    print("=" * 65)

    all_new_ids = set()

    for i, query in enumerate(DISCOVERY_QUERIES, 1):
        ids  = search_channels(youtube, query)
        new  = set(ids) - existing_ids - all_new_ids
        all_new_ids.update(new)
        print(f"  [{i:02d}/{len(DISCOVERY_QUERIES)}] '{query}' → {len(new)} new (total: {len(all_new_ids)})")
        time.sleep(0.4)

    # Remove already-known channels
    truly_new = all_new_ids - existing_ids
    print(f"\n  📋 New unique channel IDs: {len(truly_new)}")

    # Fetch details in batches of 50
    print("\n  📡 Fetching channel details...")
    id_list    = list(truly_new)
    all_detail = []
    for i in range(0, len(id_list), 50):
        batch = id_list[i:i+50]
        all_detail.extend(get_channel_details(youtube, batch))
        print(f"     Fetched {min(i+50, len(id_list))}/{len(id_list)}")
        time.sleep(0.4)

    # Filter by size
    candidates = [
        ch for ch in all_detail
        if ch["subscribers"] >= MIN_SUBSCRIBERS_NEW
        and ch["videos"] >= MIN_VIDEOS_NEW
    ]
    candidates.sort(key=lambda x: x["subscribers"], reverse=True)
    print(f"\n  After size filter (≥{MIN_SUBSCRIBERS_NEW:,} subs): {len(candidates)} channels")

    # Verify each candidate
    print("\n  🔍 Verifying new channels by video titles...")
    new_verified = []

    for i, ch in enumerate(candidates, 1):
        print(f"  [{i:02d}/{len(candidates)}] {ch['name'][:45]}", end=" ... ")
        titles = get_recent_video_titles(youtube, ch["id"])
        score, keywords = cooking_score(titles)
        status = "✅" if score >= 50 else "⚠️ " if score >= 25 else "❌"
        print(f"{status} score: {score}%")

        if score >= 50:   # only add high-confidence cooking channels
            new_verified.append({
                "name":          ch["name"],
                "id":            ch["id"],
                "active":        True,
                "category":      "cooking",
                "cooking_score": score,
                "notes":         f"{ch['subscribers']:,} subscribers | {ch['videos']} videos | score: {score}%"
            })

        time.sleep(0.4)

    print(f"\n  ✅ New cooking channels found: {len(new_verified)}")
    return new_verified


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Verify existing + discover new cooking channels")
    parser.add_argument("--api-key",       required=True)
    parser.add_argument("--verify-only",   action="store_true")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--output-dir",    default=".")
    args = parser.parse_args()

    youtube     = build_youtube(args.api_key)
    output_dir  = Path(args.output_dir)
    existing_ids = {ch["id"] for ch in EXISTING_CHANNELS}

    verified_channels = []
    rejected_channels = []
    new_channels      = []
    report_lines      = []

    # Step 1: Verify
    if not args.discover_only:
        verified_channels, rejected_channels, report_lines = \
            verify_existing_channels(youtube)

    # Step 2: Discover
    if not args.verify_only:
        new_channels = discover_new_channels(youtube, existing_ids)

    # -------------------------------------------------------------------------
    # Build final channels.yaml
    # -------------------------------------------------------------------------
    all_active   = verified_channels + new_channels
    all_inactive = rejected_channels + [
        {"name": ch["name"], "id": ch["id"], "active": False,
         "category": "cooking", "notes": "REMOVED - not a cooking channel"}
        for ch in [
            {"name": "סולטיז",                "id": "UCu4uBQqHxWidxWITvvcPU3g"},
            {"name": "Noam Firuz",             "id": "UCdTRr26_KZcZ4sDwtja5hEw"},
            {"name": "עומר לוי",               "id": "UCBKl93KXEHgsOISVdSNVVng"},
            {"name": "Full House",             "id": "UCDnhyHDsgk7mXONiky5hS8w"},
            {"name": "Michal Matzov Vlogs",    "id": "UCfLf911WZr2xQNgPEmpgzIw"},
            {"name": "Aaron Layani",           "id": "UCrJjQBE-EUvqy-RcUZVTbrg"},
            {"name": "DR. POTATO",             "id": "UCmPd-UgdNWA78BHpLpgkjaw"},
            {"name": "Or Toledano",            "id": "UCnZNASIN61iAt81BHJ108NQ"},
        ]
    ]

    channels_yaml = {
        "channels": all_active + all_inactive,
        "discovery_queries": DISCOVERY_QUERIES,
    }

    out_path = output_dir / "channels_verified.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(channels_yaml, f, allow_unicode=True,
                  default_flow_style=False, sort_keys=False)

    # Save report
    report_path = output_dir / "verification_report.txt"
    report_lines += [
        "",
        "=" * 65,
        "NEW CHANNELS DISCOVERED",
        "=" * 65,
    ]
    for ch in new_channels:
        report_lines.append(f"✅ {ch['name']} — {ch['notes']}")
    if not new_channels:
        report_lines.append("  None found above threshold.")

    report_lines += [
        "",
        "=" * 65,
        "SUMMARY",
        "=" * 65,
        f"Verified existing:  {len(verified_channels)}",
        f"Rejected existing:  {len(rejected_channels)}",
        f"New channels found: {len(new_channels)}",
        f"Total active:       {len(all_active)}",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Final print
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print(f"  Verified existing:   {len(verified_channels)}")
    print(f"  Rejected existing:   {len(rejected_channels)}")
    print(f"  New channels found:  {len(new_channels)}")
    print(f"  Total active:        {len(all_active)}")
    print(f"\n  💾 Saved → {out_path}")
    print(f"  📄 Report → {report_path}")

    if all_active:
        print(f"\n  📋 Active channels by cooking score:")
        print("  " + "-" * 55)
        for ch in sorted(all_active, key=lambda x: x.get("cooking_score", 0), reverse=True):
            score = ch.get("cooking_score", "?")
            print(f"  {ch['name']:<45} {score}%")
        print("  " + "-" * 55)

    print(f"\n  ✅ Copy channels_verified.yaml → channels.yaml when satisfied")
    print("=" * 65)


if __name__ == "__main__":
    main()