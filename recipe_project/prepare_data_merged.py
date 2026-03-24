"""
Merge original (10K) + enriched (5K) teacher output with asymmetric split.

Val and test come ONLY from original data (natural ~12% positive rate).
Enriched data goes ONLY into training (no contamination).
Negative threads downsampled at thread level to configurable ratio.

Reuses extract_thread_examples() and align_example() from prepare_data.py.

Usage:
    python -m src.preprocessing.prepare_data_merged \
        --original data/silver_labels/teacher_output.jsonl \
        --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \
        --output-dir data/processed \
        --scheme BIO \
        --model dicta-il/dictabert \
        --downsample-ratio 3.0
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter

from transformers import AutoTokenizer

from .prepare_data import (
    extract_thread_examples,
    align_example,
    thread_level_split,
    visualize_alignment,
    BIO_LABEL2ID,
    BIO_ID2LABEL,
    IO_LABEL2ID,
    IO_ID2LABEL,
)


# =============================================================================
# LOAD RECORDS
# =============================================================================

def load_records(path):
    """Load JSONL records from teacher_output file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  WARNING: Skipping malformed JSON at {path}:{line_num}")
    return records


# =============================================================================
# VIDEO-LEVEL LEAKAGE CHECK
# =============================================================================

def check_video_overlap(orig_records, enr_records):
    """Check for video_id overlap between original and enriched datasets."""
    orig_videos = set()
    for rec in orig_records:
        vid = rec.get("video_id", "")
        if vid:
            orig_videos.add(vid)

    enr_videos = set()
    for rec in enr_records:
        vid = rec.get("video_id", "")
        if vid:
            enr_videos.add(vid)

    overlap = orig_videos & enr_videos

    print(f"\n  Video-level leakage check:")
    print(f"    Original videos: {len(orig_videos)}")
    print(f"    Enriched videos: {len(enr_videos)}")
    print(f"    Overlap:         {len(overlap)}")

    if overlap:
        print(f"    ⚠️  WARNING: {len(overlap)} videos appear in both datasets")
        print(f"    This means threads from the same video could end up in")
        print(f"    training (via enriched) and test (via original).")
        print(f"    Overlapping video_ids: {list(overlap)[:10]}{'...' if len(overlap) > 10 else ''}")
    else:
        print(f"    ✅ No video-level leakage")

    return overlap


# =============================================================================
# THREAD-LEVEL DOWNSAMPLING
# =============================================================================

def downsample_negative_threads(examples, ratio, seed=42):
    """
    Downsample at THREAD level (not token level).
    All examples from the same thread stay together.

    Args:
        examples: list of dicts with thread_id, has_modification
        ratio: negative-to-positive ratio (e.g., 3.0 means 3 neg per 1 pos)
        seed: random seed
    """
    random.seed(seed)

    # Group by thread_id
    by_thread = {}
    for ex in examples:
        tid = ex["thread_id"]
        if tid not in by_thread:
            by_thread[tid] = {"examples": [], "has_mod": False}
        by_thread[tid]["examples"].append(ex)
        if ex["has_modification"]:
            by_thread[tid]["has_mod"] = True

    pos_threads = {k: v for k, v in by_thread.items() if v["has_mod"]}
    neg_threads = {k: v for k, v in by_thread.items() if not v["has_mod"]}

    n_keep = int(len(pos_threads) * ratio)
    if n_keep >= len(neg_threads):
        print(f"  Downsample: keeping all {len(neg_threads)} neg threads "
              f"(ratio would be {len(neg_threads)/max(len(pos_threads),1):.1f}:1)")
        return examples

    sampled_neg_ids = random.sample(list(neg_threads.keys()), n_keep)

    result = []
    for info in pos_threads.values():
        result.extend(info["examples"])
    for tid in sampled_neg_ids:
        result.extend(neg_threads[tid]["examples"])

    random.shuffle(result)
    n_pos = sum(1 for e in result if e["has_modification"])
    print(f"  Downsample: {len(pos_threads)} pos threads + {len(sampled_neg_ids)} neg threads "
          f"→ {len(result)} examples ({n_pos} pos = {n_pos/len(result)*100:.1f}%)")
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge original + enriched data with asymmetric split"
    )
    parser.add_argument("--original", required=True,
                        help="Path to original teacher_output.jsonl (10K threads)")
    parser.add_argument("--enriched", required=True,
                        help="Path to enriched threads_positives_focus_labeled.jsonl (5K)")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--scheme", default="BIO", choices=["BIO", "IO"])
    parser.add_argument("--model", default="dicta-il/dictabert",
                        help="Tokenizer model for BIO alignment")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--downsample-ratio", type=float, default=3.0,
                        help="Negative-to-positive thread ratio in training")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Fraction of original data for training")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of original data for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    label2id = BIO_LABEL2ID if args.scheme == "BIO" else IO_LABEL2ID
    id2label = BIO_ID2LABEL if args.scheme == "BIO" else IO_ID2LABEL

    print(f"{'='*60}")
    print(f"PREPROCESSING: Merge + Asymmetric Split + BIO Alignment")
    print(f"{'='*60}")
    print(f"  Original:  {args.original}")
    print(f"  Enriched:  {args.enriched}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Scheme:    {args.scheme} ({len(label2id)} labels)")
    print(f"  Tokenizer: {args.model}")
    print(f"  Downsample ratio: {args.downsample_ratio}:1")

    # ─── Step 1: Load records ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 1: Loading teacher output records")
    print(f"{'='*60}")

    orig_records = load_records(args.original)
    print(f"  Original: {len(orig_records)} records")

    enr_records = load_records(args.enriched)
    print(f"  Enriched: {len(enr_records)} records")

    # ─── Step 2: Video-level leakage check ────────────────────────────
    print(f"\n{'='*60}")
    print("Step 2: Leakage check")
    print(f"{'='*60}")

    overlap = check_video_overlap(orig_records, enr_records)

    # Save leakage report
    leakage_report = {
        "original_videos": len(set(r.get("video_id", "") for r in orig_records)),
        "enriched_videos": len(set(r.get("video_id", "") for r in enr_records)),
        "overlap_count": len(overlap),
        "overlap_video_ids": sorted(list(overlap))[:50],  # cap at 50
    }
    with open(out / "leakage_report.json", "w") as f:
        json.dump(leakage_report, f, indent=2)
    print(f"  Saved leakage_report.json")

    # ─── Step 3: Extract examples ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 3: Extracting comment-level examples")
    print(f"{'='*60}")

    orig_examples = []
    for rec in orig_records:
        orig_examples.extend(extract_thread_examples(rec))

    enr_examples = []
    for rec in enr_records:
        enr_examples.extend(extract_thread_examples(rec))

    n_orig_pos = sum(1 for e in orig_examples if e["has_modification"])
    n_enr_pos = sum(1 for e in enr_examples if e["has_modification"])
    print(f"  Original: {len(orig_examples)} examples ({n_orig_pos} positive = "
          f"{n_orig_pos/max(len(orig_examples),1)*100:.1f}%)")
    print(f"  Enriched: {len(enr_examples)} examples ({n_enr_pos} positive = "
          f"{n_enr_pos/max(len(enr_examples),1)*100:.1f}%)")

    # ─── Step 4: Split original (thread-level, stratified) ────────────
    print(f"\n{'='*60}")
    print(f"Step 4: Splitting original data "
          f"({args.train_ratio:.0%}/{args.val_ratio:.0%}/"
          f"{1-args.train_ratio-args.val_ratio:.0%})")
    print(f"{'='*60}")

    orig_train, orig_val, orig_test = thread_level_split(
        orig_examples, args.train_ratio, args.val_ratio, args.seed
    )

    def count_pos(exs):
        return sum(1 for e in exs if e["has_modification"])

    print(f"  Original train: {len(orig_train)} ({count_pos(orig_train)} pos)")
    print(f"  Original val:   {len(orig_val)} ({count_pos(orig_val)} pos)")
    print(f"  Original test:  {len(orig_test)} ({count_pos(orig_test)} pos)")

    # ─── Step 5: Verify no enriched threads in val/test ───────────────
    enr_thread_ids = {e["thread_id"] for e in enr_examples}
    val_thread_ids = {e["thread_id"] for e in orig_val}
    test_thread_ids = {e["thread_id"] for e in orig_test}

    leak_val = enr_thread_ids & val_thread_ids
    leak_test = enr_thread_ids & test_thread_ids

    if leak_val or leak_test:
        print(f"\n  ❌ CONTAMINATION: enriched threads in val ({len(leak_val)}) "
              f"or test ({len(leak_test)})")
        print(f"  This should not happen if enriched has different thread_ids.")
        print(f"  Investigate before proceeding.")
    else:
        print(f"\n  ✅ No enriched thread IDs in val or test")

    # ─── Step 6: Merge original train + ALL enriched ──────────────────
    print(f"\n{'='*60}")
    print("Step 5: Merging training data")
    print(f"{'='*60}")

    merged_train = orig_train + enr_examples
    print(f"  Merged train (before downsample): {len(merged_train)} "
          f"({count_pos(merged_train)} pos = "
          f"{count_pos(merged_train)/max(len(merged_train),1)*100:.1f}%)")

    # ─── Step 7: Downsample negatives ─────────────────────────────────
    merged_train = downsample_negative_threads(
        merged_train, args.downsample_ratio, args.seed
    )

    # ─── Step 8: Align BIO labels ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 6: Aligning spans to BIO tokens")
    print(f"{'='*60}")

    def process_split(examples, name):
        processed = []
        stats = Counter()
        for i, ex in enumerate(examples):
            if (i + 1) % 1000 == 0:
                print(f"    Processing {name}: {i+1}/{len(examples)}...")

            result, s = align_example(
                ex["text"], ex["modifications"], tokenizer, label2id, args.max_length
            )

            if result is None:
                stats["skipped"] += 1
                continue

            for k, v in s.items():
                stats[k] += v

            # Attach metadata
            result["thread_id"] = ex["thread_id"]
            result["source_comment"] = ex["source_comment"]
            result["has_modification"] = ex["has_modification"]

            processed.append(result)

        n_pos = sum(1 for e in processed if e["has_modification"])
        total_spans = stats.get("aligned", 0) + stats.get("hallucinated", 0) + stats.get("truncated", 0)
        align_rate = stats.get("aligned", 0) / max(total_spans, 1)

        print(f"  {name}: {len(processed)} examples ({n_pos} pos)")
        print(f"    Aligned:      {stats.get('aligned', 0)}/{total_spans} "
              f"({align_rate*100:.1f}%)")
        print(f"    Hallucinated: {stats.get('hallucinated', 0)}")
        print(f"    Truncated:    {stats.get('truncated', 0)}")
        return processed

    train_proc = process_split(merged_train, "train_merged")
    val_proc = process_split(orig_val, "val")
    test_proc = process_split(orig_test, "test")

    # ─── Step 9: Save ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 7: Saving to JSONL")
    print(f"{'='*60}")

    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl(train_proc, out / "train_merged.jsonl")
    save_jsonl(val_proc, out / "val.jsonl")
    save_jsonl(test_proc, out / "test.jsonl")

    print(f"  train_merged.jsonl: {len(train_proc)} examples "
          f"({count_pos(train_proc)} pos = "
          f"{count_pos(train_proc)/max(len(train_proc),1)*100:.1f}%)")
    print(f"  val.jsonl:          {len(val_proc)} examples "
          f"({count_pos(val_proc)} pos = "
          f"{count_pos(val_proc)/max(len(val_proc),1)*100:.1f}%)")
    print(f"  test.jsonl:         {len(test_proc)} examples "
          f"({count_pos(test_proc)} pos = "
          f"{count_pos(test_proc)/max(len(test_proc),1)*100:.1f}%)")

    # Save stats
    stats_dict = {
        "train_merged": {"total": len(train_proc), "positive": count_pos(train_proc)},
        "val": {"total": len(val_proc), "positive": count_pos(val_proc)},
        "test": {"total": len(test_proc), "positive": count_pos(test_proc)},
        "config": vars(args),
        "video_overlap": len(overlap),
    }
    with open(out / "stats_merged.json", "w") as f:
        json.dump(stats_dict, f, indent=2)

    # ─── Eye test ─────────────────────────────────────────────────────
    pos = [e for e in train_proc if e["has_modification"]]
    if pos:
        print(f"\n--- EYE TEST (3 positive examples from training) ---")
        for ex in random.sample(pos, min(3, len(pos))):
            visualize_alignment(ex, tokenizer, id2label)

    # ─── O-token ratio ────────────────────────────────────────────────
    o_count = 0
    total_tokens = 0
    for ex in train_proc:
        for l in ex["labels"]:
            if l == -100:
                continue
            total_tokens += 1
            if l == 0:
                o_count += 1
    o_pct = o_count / max(total_tokens, 1) * 100
    print(f"\n  O-token ratio in training: {o_pct:.1f}%")
    print(f"  → Use entity F1 for early stopping, NOT token accuracy")

    print(f"\n{'='*60}")
    print(f"✅ Done. Next steps:")
    print(f"  python -m src.models.train_joint \\")
    print(f"      --data-dir {args.output_dir} \\")
    print(f"      --train-file train_merged.jsonl \\")
    print(f"      --model {args.model}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
