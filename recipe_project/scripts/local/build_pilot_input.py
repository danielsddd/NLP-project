#!/usr/bin/env python3
"""
Build pilot_input.jsonl for Track A pilot evaluation.
=====================================================
Per master plan §4.4, the pilot needs ~100 threads where:
  1. Each thread_id has ground truth in gold_final.jsonl
  2. Each thread has full content (top_comment + replies) for re-labeling

We can't use gold_final.jsonl alone because it stores only the top comment text,
not the replies. So we:
  1. Read gold_final.jsonl → set of thread_ids with ground truth, mark positives
  2. Read silver files (teacher_output.jsonl + threads_positives_focus_labeled.jsonl)
     to recover the full thread content (top + replies + metadata)
  3. STRATIFIED sample: include all gold positives + random negatives to fill 100.
     Rationale: gold has 24 positives in 496 threads (4.8%). Random 100-thread
     slice → ~5 positives = wide CI on per-aspect F1. Stratified → tighter F1 CI
     for the metric we actually care about.
  4. Convert silver format → raw format (matches data/raw_youtube/threads.jsonl
     schema that run_pass1 expects)
  5. Write to data/silver_labels/pilot_input.jsonl

Usage:
    python scripts/local/build_pilot_input.py
    
    # Or with custom paths:
    python scripts/local/build_pilot_input.py \
        --gold data/gold_validation/gold_final.jsonl \
        --silver data/silver_labels/teacher_output.jsonl \
        --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \
        --output data/silver_labels/pilot_input.jsonl \
        --target-size 100 --seed 42

Limitations (acceptable for pilot):
  - Reply creator role (`is_creator`) is not preserved in silver — all replies
    are marked is_creator=False. The prompt's Rule 6 ("creator replies get
    0.90+ confidence") will be slightly less accurate, but this affects fewer
    than 10% of threads and does not change which spans are extracted.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from collections import Counter


# =============================================================================
# DEFAULT PATHS
# =============================================================================

DEFAULT_GOLD     = "data/gold_validation/gold_final.jsonl"
DEFAULT_SILVER   = "data/silver_labels/teacher_output.jsonl"
DEFAULT_ENRICHED = "data/silver_labels/threads_positives_focus_labeled.jsonl"
DEFAULT_OUTPUT   = "data/silver_labels/pilot_input.jsonl"
DEFAULT_TARGET   = 100
DEFAULT_SEED     = 42


# =============================================================================
# IO HELPERS
# =============================================================================

def load_jsonl(path: Path) -> list:
    """Load a JSONL file. Skips malformed lines with a warning."""
    if not path.exists():
        return []
    records = []
    malformed = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                malformed += 1
                if malformed <= 3:
                    print(f"  WARN: {path.name} line {i} malformed — {e}")
    if malformed > 3:
        print(f"  WARN: {malformed} total malformed lines in {path.name}")
    return records


def write_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =============================================================================
# FORMAT CONVERSION
# =============================================================================

def silver_to_raw(silver_rec: dict) -> dict:
    """Convert a silver-labeled record back to the raw threads.jsonl schema
    that run_pass1 expects.

    Silver schema (input):  flat fields (top_comment_text, replies_texts, ...)
    Raw schema   (output):  nested dicts (top_comment.text, replies[].text)
    """
    return {
        "thread_id":     silver_rec["thread_id"],
        "video_id":      silver_rec.get("video_id", ""),
        "channel_id":    silver_rec.get("channel_id", ""),
        "video_title":   silver_rec.get("video_title", ""),
        "channel_title": silver_rec.get("channel_title", ""),
        "top_comment":   {"text": silver_rec["top_comment_text"]},
        # is_creator is lost in silver — default to False. Acceptable for pilot.
        "replies": [
            {"text": r, "is_creator": False}
            for r in silver_rec.get("replies_texts", [])
        ],
        "has_creator_reply": silver_rec.get("has_creator_reply", False),
        "total_likes":       silver_rec.get("total_likes", 0),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Build stratified pilot_input.jsonl from gold + silver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--gold",     default=DEFAULT_GOLD,     help=f"Gold annotations (default: {DEFAULT_GOLD})")
    ap.add_argument("--silver",   default=DEFAULT_SILVER,   help=f"Main silver labels (default: {DEFAULT_SILVER})")
    ap.add_argument("--enriched", default=DEFAULT_ENRICHED, help=f"Enriched silver labels (default: {DEFAULT_ENRICHED})")
    ap.add_argument("--output",   default=DEFAULT_OUTPUT,   help=f"Output pilot file (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--target-size", type=int, default=DEFAULT_TARGET,
                    help=f"Target pilot size (default: {DEFAULT_TARGET})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"Random seed for negative sampling (default: {DEFAULT_SEED})")
    ap.add_argument("--no-stratify", action="store_true",
                    help="Disable stratification — take first N overlap threads instead")
    args = ap.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  BUILD PILOT INPUT")
    print(f"{'=' * 70}")
    print(f"  Gold:     {args.gold}")
    print(f"  Silver:   {args.silver}")
    print(f"  Enriched: {args.enriched}")
    print(f"  Output:   {args.output}")
    print(f"  Target:   {args.target_size} threads (seed={args.seed})")
    print(f"  Strategy: {'stratified (all gold pos + random neg)' if not args.no_stratify else 'first N'}")
    print()

    # ── 1. Load gold ────────────────────────────────────────────────────
    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"❌ Gold file not found: {gold_path}")
        sys.exit(1)
    gold = load_jsonl(gold_path)
    print(f"  Loaded {len(gold)} gold records from {gold_path.name}")

    gold_pos_ids = set()  # thread_ids where has_modification == True
    gold_neg_ids = set()  # thread_ids where has_modification == False
    for rec in gold:
        tid = rec.get("thread_id")
        if not tid:
            continue
        if rec.get("has_modification", False):
            gold_pos_ids.add(tid)
        else:
            gold_neg_ids.add(tid)
    print(f"    Gold positives: {len(gold_pos_ids)}")
    print(f"    Gold negatives: {len(gold_neg_ids)}")
    print(f"    Gold total:     {len(gold_pos_ids) + len(gold_neg_ids)}")

    # ── 2. Load silver (both files) ────────────────────────────────────
    silver_recs = load_jsonl(Path(args.silver))
    enriched_recs = load_jsonl(Path(args.enriched))
    print(f"\n  Loaded {len(silver_recs)} records from {Path(args.silver).name}")
    print(f"  Loaded {len(enriched_recs)} records from {Path(args.enriched).name}")

    # Combine into a single thread_id → record map.
    # Prefer main silver over enriched if duplicate thread_ids exist.
    combined = {}
    for rec in enriched_recs:
        tid = rec.get("thread_id")
        if tid:
            combined[tid] = rec
    for rec in silver_recs:
        tid = rec.get("thread_id")
        if tid:
            combined[tid] = rec  # main silver wins if duplicate
    print(f"  Combined silver universe: {len(combined)} unique threads")

    # ── 3. Find overlap (gold AND silver) ──────────────────────────────
    overlap_pos = sorted(gold_pos_ids & combined.keys())
    overlap_neg = sorted(gold_neg_ids & combined.keys())
    print(f"\n  Overlap (gold ∩ silver):")
    print(f"    Positives in both: {len(overlap_pos)} / {len(gold_pos_ids)} gold pos")
    print(f"    Negatives in both: {len(overlap_neg)} / {len(gold_neg_ids)} gold neg")

    if len(overlap_pos) + len(overlap_neg) < args.target_size:
        print(f"\n⚠️  WARNING: only {len(overlap_pos) + len(overlap_neg)} threads in overlap, "
              f"target is {args.target_size}. Will use all available.")

    # ── 4. Sample ──────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    selected_ids = []

    if args.no_stratify:
        all_overlap = overlap_pos + overlap_neg
        rng.shuffle(all_overlap)
        selected_ids = all_overlap[:args.target_size]
    else:
        # Stratified: include ALL gold positives, then random negatives to fill.
        selected_ids = list(overlap_pos)
        n_negatives_needed = args.target_size - len(selected_ids)
        if n_negatives_needed > 0:
            sampled_neg = rng.sample(
                overlap_neg,
                min(n_negatives_needed, len(overlap_neg))
            )
            selected_ids.extend(sampled_neg)
        rng.shuffle(selected_ids)  # avoid all positives clumping at the front

    print(f"\n  Selected {len(selected_ids)} threads:")
    n_pos_selected = sum(1 for t in selected_ids if t in gold_pos_ids)
    n_neg_selected = sum(1 for t in selected_ids if t in gold_neg_ids)
    print(f"    Gold positives: {n_pos_selected} ({n_pos_selected / max(1, len(selected_ids)) * 100:.1f}%)")
    print(f"    Gold negatives: {n_neg_selected} ({n_neg_selected / max(1, len(selected_ids)) * 100:.1f}%)")

    # ── 5. Convert silver → raw format and write ───────────────────────
    pilot_records = [silver_to_raw(combined[tid]) for tid in selected_ids]

    # Sanity check: each record has the required fields
    bad = []
    for rec in pilot_records:
        if not rec.get("top_comment", {}).get("text"):
            bad.append(rec.get("thread_id", "?"))
    if bad:
        print(f"\n⚠️  WARNING: {len(bad)} records have empty top_comment_text")
        print(f"  First few: {bad[:5]}")

    output_path = Path(args.output)
    write_jsonl(pilot_records, output_path)

    # ── 6. Report ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  ✅ DONE")
    print(f"{'=' * 70}")
    print(f"  Wrote {len(pilot_records)} threads → {output_path}")

    # Reply count distribution (sanity check on thread structure)
    reply_counts = Counter(len(r["replies"]) for r in pilot_records)
    print(f"\n  Reply count distribution:")
    for n_replies in sorted(reply_counts):
        print(f"    {n_replies} replies: {reply_counts[n_replies]} threads")

    print(f"\n  Next: run Pass 1 on this pilot file:")
    print(f"    python -m src.teacher_labeling.generate_labels \\")
    print(f"        -i {output_path} \\")
    print(f"        -o data/silver_labels/pilot_output.jsonl \\")
    print(f"        --batch-size 20")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()