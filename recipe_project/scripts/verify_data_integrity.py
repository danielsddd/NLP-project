#!/usr/bin/env python3
"""
Data Integrity Verification
============================
Checks for video-level leakage between the original (10K) and enriched (5K)
datasets. If the same video_id appears in both, comments from that video
could end up in both train (enriched) and test (original), causing leakage.

This is Step 0.1 in MASTER_PLAN_v7 — run BEFORE anything else.

Usage:
    python scripts/verify_data_integrity.py

    # Or with custom paths:
    python scripts/verify_data_integrity.py \
        --original data/silver_labels/teacher_output.jsonl \
        --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \
        --output results/leakage_report.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter


def load_video_ids(path: str) -> dict:
    """Load video_ids and thread_ids from a JSONL file.

    Returns:
        dict mapping video_id → list of thread_ids
    """
    video_threads = defaultdict(list)
    thread_count = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            thread_count += 1
            vid = record.get("video_id", "unknown")
            tid = record.get("thread_id", f"thread_{thread_count}")
            video_threads[vid].append(tid)

    return video_threads


def check_leakage(original_path: str, enriched_path: str, output_path: str):
    """Check for video-level overlap and generate leakage report."""

    print(f"{'='*60}")
    print(f"DATA INTEGRITY CHECK — Video-Level Leakage")
    print(f"{'='*60}")
    print(f"  Original: {original_path}")
    print(f"  Enriched: {enriched_path}")
    print()

    # Load both
    orig_videos = load_video_ids(original_path)
    enr_videos = load_video_ids(enriched_path)

    orig_video_set = set(orig_videos.keys())
    enr_video_set = set(enr_videos.keys())
    overlap = orig_video_set & enr_video_set
    orig_only = orig_video_set - enr_video_set
    enr_only = enr_video_set - orig_video_set

    orig_thread_count = sum(len(v) for v in orig_videos.values())
    enr_thread_count = sum(len(v) for v in enr_videos.values())

    print(f"  Original: {len(orig_video_set)} unique videos, {orig_thread_count} threads")
    print(f"  Enriched: {len(enr_video_set)} unique videos, {enr_thread_count} threads")
    print(f"  Overlap:  {len(overlap)} videos")
    print()

    # Count threads in overlapping videos
    overlap_orig_threads = sum(len(orig_videos[v]) for v in overlap)
    overlap_enr_threads = sum(len(enr_videos[v]) for v in overlap)

    # Build report
    report = {
        "original": {
            "path": original_path,
            "unique_videos": len(orig_video_set),
            "total_threads": orig_thread_count,
        },
        "enriched": {
            "path": enriched_path,
            "unique_videos": len(enr_video_set),
            "total_threads": enr_thread_count,
        },
        "overlap": {
            "video_count": len(overlap),
            "original_threads_in_overlap": overlap_orig_threads,
            "enriched_threads_in_overlap": overlap_enr_threads,
            "overlapping_video_ids": sorted(list(overlap))[:50],  # cap for readability
        },
        "verdict": "",
        "recommended_action": "",
    }

    # Verdict
    if len(overlap) == 0:
        report["verdict"] = "CLEAN — no video overlap"
        report["recommended_action"] = "Proceed normally. No leakage risk."
        print(f"  ✅ VERDICT: {report['verdict']}")
        print(f"     {report['recommended_action']}")
    elif len(overlap) <= 10:
        report["verdict"] = f"MINOR — {len(overlap)} overlapping videos"
        report["recommended_action"] = (
            f"Option A: Remove {len(overlap)} overlapping videos from enriched set. "
            f"Option B: Ensure overlapping video threads go to SAME split. "
            f"Affected: {overlap_orig_threads} original + {overlap_enr_threads} enriched threads."
        )
        print(f"  ⚠️  VERDICT: {report['verdict']}")
        print(f"     {report['recommended_action']}")
        print(f"\n  Overlapping video IDs:")
        for vid in sorted(overlap):
            print(f"    {vid}: {len(orig_videos[vid])} orig + {len(enr_videos[vid])} enr threads")
    else:
        report["verdict"] = f"MAJOR — {len(overlap)} overlapping videos"
        report["recommended_action"] = (
            f"High leakage risk. {overlap_orig_threads} original threads share videos "
            f"with {overlap_enr_threads} enriched threads. "
            f"MUST either: (1) remove overlapping videos from enriched, or "
            f"(2) add all threads from overlapping videos to train only."
        )
        print(f"  ❌ VERDICT: {report['verdict']}")
        print(f"     {report['recommended_action']}")

    # Additional stats: channel distribution
    print(f"\n--- Channel Distribution ---")
    orig_channels = Counter()
    enr_channels = Counter()
    with open(original_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                orig_channels[rec.get("channel_id", "unknown")] += 1
            except (json.JSONDecodeError, AttributeError):
                continue
    with open(enriched_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                enr_channels[rec.get("channel_id", "unknown")] += 1
            except (json.JSONDecodeError, AttributeError):
                continue

    report["channel_overlap"] = len(set(orig_channels.keys()) & set(enr_channels.keys()))
    print(f"  Original channels:  {len(orig_channels)}")
    print(f"  Enriched channels:  {len(enr_channels)}")
    print(f"  Channel overlap:    {report['channel_overlap']}")

    # Positive rate check
    print(f"\n--- Positive Rate Check ---")
    orig_pos = 0
    with open(original_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                label = rec.get("final_label") or rec.get("teacher_output") or {}
                if label.get("has_modification", False):
                    orig_pos += 1
            except (json.JSONDecodeError, AttributeError):
                continue

    enr_pos = 0
    with open(enriched_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                label = rec.get("final_label") or rec.get("teacher_output") or {}
                if label.get("has_modification", False):
                    enr_pos += 1
            except (json.JSONDecodeError, AttributeError):
                continue

    print(f"  Original positive rate: {orig_pos}/{orig_thread_count} = "
          f"{100*orig_pos/max(1,orig_thread_count):.1f}%")
    print(f"  Enriched positive rate: {enr_pos}/{enr_thread_count} = "
          f"{100*enr_pos/max(1,enr_thread_count):.1f}%")

    report["positive_rates"] = {
        "original": round(orig_pos / max(1, orig_thread_count), 4),
        "enriched": round(enr_pos / max(1, enr_thread_count), 4),
    }

    # Save report
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Check for video-level data leakage")
    parser.add_argument("--original",
                        default="data/silver_labels/teacher_output.jsonl",
                        help="Path to original teacher output JSONL")
    parser.add_argument("--enriched",
                        default="data/silver_labels/threads_positives_focus_labeled.jsonl",
                        help="Path to enriched JSONL")
    parser.add_argument("--output",
                        default="results/leakage_report.json",
                        help="Output path for leakage report JSON")
    args = parser.parse_args()

    if not Path(args.original).exists():
        print(f"❌ Original file not found: {args.original}")
        return
    if not Path(args.enriched).exists():
        print(f"❌ Enriched file not found: {args.enriched}")
        return

    check_leakage(args.original, args.enriched, args.output)


if __name__ == "__main__":
    main()