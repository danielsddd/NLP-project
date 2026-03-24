#!/usr/bin/env python3
"""
Data integrity verification: video-level leakage check + dataset statistics.

Run this BEFORE any preprocessing or training.

Usage:
    cd recipe_project
    python scripts/verify_data_integrity.py \
        --original data/silver_labels/teacher_output.jsonl \
        --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \
        --output results/leakage_report.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict


def load_records(path):
    """Load JSONL records."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def check_video_overlap(orig, enr):
    """Check video_id overlap between two record sets."""
    orig_videos = set()
    orig_video_threads = defaultdict(list)
    for rec in orig:
        vid = rec.get("video_id", "")
        tid = rec.get("thread_id", "")
        if vid:
            orig_videos.add(vid)
            orig_video_threads[vid].append(tid)

    enr_videos = set()
    enr_video_threads = defaultdict(list)
    for rec in enr:
        vid = rec.get("video_id", "")
        tid = rec.get("thread_id", "")
        if vid:
            enr_videos.add(vid)
            enr_video_threads[vid].append(tid)

    overlap = orig_videos & enr_videos

    return {
        "original_unique_videos": len(orig_videos),
        "enriched_unique_videos": len(enr_videos),
        "overlap_count": len(overlap),
        "overlap_video_ids": sorted(list(overlap)),
        "overlap_details": {
            vid: {
                "original_threads": len(orig_video_threads[vid]),
                "enriched_threads": len(enr_video_threads[vid]),
            }
            for vid in sorted(overlap)[:20]  # Show details for first 20
        },
    }


def check_thread_overlap(orig, enr):
    """Check thread_id overlap (should be zero — different collection runs)."""
    orig_tids = {rec.get("thread_id", "") for rec in orig}
    enr_tids = {rec.get("thread_id", "") for rec in enr}
    overlap = orig_tids & enr_tids
    return {
        "original_unique_threads": len(orig_tids),
        "enriched_unique_threads": len(enr_tids),
        "overlap_count": len(overlap),
        "overlap_thread_ids": sorted(list(overlap))[:20],
    }


def compute_dataset_stats(records, name):
    """Compute statistics for a dataset."""
    n = len(records)
    if n == 0:
        return {"name": name, "count": 0}

    # has_modification rate
    n_pos = 0
    aspect_counts = Counter()
    total_mods = 0
    thread_lengths = []
    has_creator_reply = 0
    vote_methods = Counter()

    for rec in records:
        fl = rec.get("final_label") or rec.get("teacher_output") or {}
        has_mod = fl.get("has_modification", False)
        if has_mod:
            n_pos += 1
        for mod in fl.get("modifications", []):
            aspect = mod.get("aspect", "UNKNOWN")
            aspect_counts[aspect] += 1
            total_mods += 1

        # Thread length
        replies = rec.get("replies_texts", [])
        thread_lengths.append(1 + len(replies))

        if rec.get("has_creator_reply", False):
            has_creator_reply += 1

        vm = rec.get("vote_method", "unknown")
        vote_methods[vm] += 1

    avg_len = sum(thread_lengths) / n
    channels = len(set(rec.get("channel_id", "") for rec in records))
    videos = len(set(rec.get("video_id", "") for rec in records))

    return {
        "name": name,
        "threads": n,
        "positive_threads": n_pos,
        "negative_threads": n - n_pos,
        "positive_rate": round(n_pos / n * 100, 1),
        "total_modifications": total_mods,
        "aspect_distribution": dict(aspect_counts),
        "avg_comments_per_thread": round(avg_len, 1),
        "max_comments_per_thread": max(thread_lengths),
        "unique_channels": channels,
        "unique_videos": videos,
        "has_creator_reply_count": has_creator_reply,
        "vote_methods": dict(vote_methods),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify data integrity")
    parser.add_argument("--original", required=True,
                        help="Path to original teacher_output.jsonl")
    parser.add_argument("--enriched", required=True,
                        help="Path to enriched threads_positives_focus_labeled.jsonl")
    parser.add_argument("--output", default="results/leakage_report.json",
                        help="Output path for report JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("DATA INTEGRITY VERIFICATION")
    print("=" * 60)

    # Load
    print(f"\nLoading {args.original}...")
    orig = load_records(args.original)
    print(f"  {len(orig)} records")

    print(f"Loading {args.enriched}...")
    enr = load_records(args.enriched)
    print(f"  {len(enr)} records")

    # Video overlap
    print(f"\n{'='*60}")
    print("CHECK 1: Video-level overlap")
    print(f"{'='*60}")
    video_report = check_video_overlap(orig, enr)
    print(f"  Original unique videos:  {video_report['original_unique_videos']}")
    print(f"  Enriched unique videos:  {video_report['enriched_unique_videos']}")
    print(f"  Overlap:                 {video_report['overlap_count']}")

    if video_report["overlap_count"] > 0:
        print(f"\n  ⚠️  WARNING: {video_report['overlap_count']} videos appear in both sets!")
        print(f"  Risk: comments from the same video in train (enriched) and test (original)")
        print(f"  Showing first 10 overlapping video IDs:")
        for vid in video_report["overlap_video_ids"][:10]:
            d = video_report["overlap_details"].get(vid, {})
            print(f"    {vid}: {d.get('original_threads', '?')} orig + "
                  f"{d.get('enriched_threads', '?')} enriched threads")
        print(f"\n  RECOMMENDATION: Ensure overlapping videos' threads go to same split,")
        print(f"  OR remove overlapping videos from enriched set.")
    else:
        print(f"\n  ✅ No video-level leakage")

    # Thread overlap
    print(f"\n{'='*60}")
    print("CHECK 2: Thread-level overlap")
    print(f"{'='*60}")
    thread_report = check_thread_overlap(orig, enr)
    print(f"  Original unique threads: {thread_report['original_unique_threads']}")
    print(f"  Enriched unique threads: {thread_report['enriched_unique_threads']}")
    print(f"  Overlap:                 {thread_report['overlap_count']}")

    if thread_report["overlap_count"] > 0:
        print(f"  ❌ CRITICAL: {thread_report['overlap_count']} threads appear in BOTH files!")
        print(f"  These MUST be deduplicated before proceeding.")
    else:
        print(f"  ✅ No thread-level duplication")

    # Dataset statistics
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")

    orig_stats = compute_dataset_stats(orig, "original")
    enr_stats = compute_dataset_stats(enr, "enriched")

    for stats in [orig_stats, enr_stats]:
        print(f"\n  {stats['name'].upper()}:")
        print(f"    Threads:       {stats['threads']} "
              f"({stats['positive_threads']} pos = {stats['positive_rate']}%)")
        print(f"    Modifications: {stats['total_modifications']}")
        print(f"    Aspects:       {stats['aspect_distribution']}")
        print(f"    Avg comments:  {stats['avg_comments_per_thread']}")
        print(f"    Channels:      {stats['unique_channels']}")
        print(f"    Videos:        {stats['unique_videos']}")
        print(f"    Creator reply: {stats['has_creator_reply_count']}")
        print(f"    Vote methods:  {stats['vote_methods']}")

    # Save report
    report = {
        "video_overlap": video_report,
        "thread_overlap": thread_report,
        "original_stats": orig_stats,
        "enriched_stats": enr_stats,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to: {out_path}")

    # Final verdict
    print(f"\n{'='*60}")
    issues = []
    if video_report["overlap_count"] > 0:
        issues.append(f"Video overlap: {video_report['overlap_count']} videos")
    if thread_report["overlap_count"] > 0:
        issues.append(f"Thread overlap: {thread_report['overlap_count']} threads")

    if issues:
        print("VERDICT: ⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nAddress these before running preprocessing.")
    else:
        print("VERDICT: ✅ Data integrity verified. Safe to proceed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
