#!/usr/bin/env python3
"""
Merged Data Preprocessing — v1
================================
Merges original (10K) and enriched (5K) teacher outputs into a single
training set with configurable negative downsampling. Validation and test
splits come ONLY from the original data to preserve natural distribution.

Addresses MASTER_PLAN_v7 §6 and clarifications:
  - --downsample-ratio is a CLI arg so A2 ablation can sweep 2:1, 3:1, 4:1
  - Auto-computes inverse-frequency class weights and saves to stats file
  - Supports --scheme BIO|IO for ablation A7
  - Supports --unanimous-only for ablation A8 (3/3 agreement filter)

Usage:
    # Default (3:1 downsample):
    python -m src.preprocessing.prepare_data_merged \\
        --original data/silver_labels/teacher_output.jsonl \\
        --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \\
        --output-dir data/processed \\
        --model dicta-il/dictabert \\
        --downsample-ratio 3.0

    # Ablation A2 — sweep ratios:
    python -m src.preprocessing.prepare_data_merged \\
        --original data/silver_labels/teacher_output.jsonl \\
        --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \\
        --output-dir data/processed_ds2 \\
        --model dicta-il/dictabert \\
        --downsample-ratio 2.0

    # Ablation A6 — without enriched:
    python -m src.preprocessing.prepare_data_merged \\
        --original data/silver_labels/teacher_output.jsonl \\
        --output-dir data/processed_no_enriched \\
        --model dicta-il/dictabert

    # Ablation A7 — IO scheme:
    python -m src.preprocessing.prepare_data_merged \\
        --original data/silver_labels/teacher_output.jsonl \\
        --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \\
        --output-dir data/processed_io \\
        --model dicta-il/dictabert \\
        --scheme IO

    # Ablation A8 — unanimous labels only:
    python -m src.preprocessing.prepare_data_merged \\
        --original data/silver_labels/teacher_output.jsonl \\
        --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \\
        --output-dir data/processed_unanimous \\
        --model dicta-il/dictabert \\
        --unanimous-only
"""

import json
import random
import argparse
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

from transformers import AutoTokenizer

# Import from existing prepare_data.py — reuse all alignment logic
from src.preprocessing.prepare_data import (
    extract_thread_examples,
    align_example,
    thread_level_split,
    visualize_alignment,
    BIO_LABEL2ID,
    BIO_ID2LABEL,
    IO_LABEL2ID,
    IO_ID2LABEL,
    VALID_ASPECTS,
)


# =============================================================================
# CLASS WEIGHT COMPUTATION
# =============================================================================

def compute_class_weights(examples: List[Dict], label2id: Dict[str, int],
                          scheme: str = "BIO") -> Dict[str, float]:
    """Compute inverse-frequency class weights from actual training data.

    This replaces the hardcoded weights in MASTER_PLAN_v6 §7 with weights
    derived from the real label distribution. The formula is:

        weight_i = total_tokens / (num_classes * count_i)

    This is the standard sklearn 'balanced' formula. We also apply a cap
    so no single weight exceeds 20.0 (prevents instability from very rare
    labels).

    Returns:
        dict mapping label name → weight (e.g., {"O": 0.15, "B-SUBSTITUTION": 8.7, ...})
        Also returns a list ordered by label ID for direct use with CrossEntropyLoss.
    """
    label_counts = Counter()
    total_tokens = 0

    for ex in examples:
        for lid in ex["labels"]:
            if lid == -100:  # special tokens (CLS, SEP, PAD, subwords)
                continue
            total_tokens += 1
            label_counts[lid] += 1

    id2label = BIO_ID2LABEL if scheme == "BIO" else IO_ID2LABEL
    num_classes = len(label2id)

    weights = {}
    weights_list = [1.0] * num_classes  # fallback

    for lid in range(num_classes):
        label_name = id2label.get(lid, f"UNK_{lid}")
        count = label_counts.get(lid, 1)  # avoid div-by-zero
        raw_weight = total_tokens / (num_classes * count)
        capped_weight = min(raw_weight, 20.0)  # cap to prevent instability
        weights[label_name] = round(capped_weight, 4)
        weights_list[lid] = round(capped_weight, 4)

    return weights, weights_list


def compute_uniform_weights(label2id: Dict[str, int]) -> List[float]:
    """Return uniform weights (all 1.0) for ablation A1 comparison."""
    return [1.0] * len(label2id)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_threads(path: str, unanimous_only: bool = False) -> List[Dict]:
    """Load thread records from a JSONL file.

    Args:
        path: Path to JSONL file.
        unanimous_only: If True, keep only threads where vote_method
            indicates 3/3 unanimous agreement. For ablation A8.

    Returns:
        List of thread record dicts.
    """
    records = []
    skipped_non_unanimous = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if unanimous_only:
                vote = record.get("vote_method", "")
                # Keep only records with unanimous agreement (3/3)
                # Common formats: "unanimous", "3/3", "all_agree"
                # Also check if agreement_score >= 3
                agreement = record.get("agreement_score", 0)
                is_unanimous = (
                    "unanimous" in str(vote).lower()
                    or "3/3" in str(vote)
                    or "all_agree" in str(vote).lower()
                    or agreement >= 3
                )
                if not is_unanimous:
                    skipped_non_unanimous += 1
                    continue

            records.append(record)

    if unanimous_only and skipped_non_unanimous > 0:
        print(f"  Skipped {skipped_non_unanimous} non-unanimous threads (A8 filter)")

    return records


# =============================================================================
# DOWNSAMPLING
# =============================================================================

def downsample_negatives(examples: List[Dict], ratio: float,
                         seed: int = 42) -> List[Dict]:
    """Downsample negative (all-O) examples to achieve target neg:pos ratio.

    Args:
        examples: List of processed examples with 'has_modification' key.
        ratio: Target negatives-to-positives ratio.
            ratio=3.0 means keep 3 negatives per 1 positive.
            ratio=0.0 or ratio=None means no downsampling (keep all).
        seed: Random seed for reproducibility.

    Returns:
        Downsampled list of examples.
    """
    if ratio is None or ratio <= 0:
        return examples

    rng = random.Random(seed)

    positives = [ex for ex in examples if ex["has_modification"]]
    negatives = [ex for ex in examples if not ex["has_modification"]]

    n_pos = len(positives)
    target_neg = int(n_pos * ratio)

    if target_neg >= len(negatives):
        # Already at or below target ratio — keep all
        print(f"  Downsampling: {len(negatives)} negatives already ≤ target {target_neg}")
        return examples

    rng.shuffle(negatives)
    sampled_negatives = negatives[:target_neg]

    result = positives + sampled_negatives
    rng.shuffle(result)

    print(f"  Downsampling: {len(negatives)} → {len(sampled_negatives)} negatives "
          f"(ratio {ratio}:1, {n_pos} positives)")
    return result


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_merged(
    original_path: str,
    enriched_path: Optional[str],
    output_dir: str,
    model_name: str = "dicta-il/dictabert",
    max_length: int = 128,
    scheme: str = "BIO",
    downsample_ratio: float = 3.0,
    seed: int = 42,
    unanimous_only: bool = False,
    data_fraction: float = 1.0,
):
    """Full merged preprocessing pipeline.

    Steps:
        1. Load original + enriched threads
        2. Extract comment-level examples from both
        3. Split ORIGINAL into train/val/test at thread level (70/15/15)
        4. Add ALL enriched examples to train only
        5. Downsample negative examples in train
        6. (Optional) Subsample train for data size ablation A5
        7. Tokenize + align BIO labels
        8. Compute class weights from training data
        9. Save everything

    Args:
        original_path: Path to teacher_output.jsonl (10K threads)
        enriched_path: Path to enriched JSONL (5K threads), or None for A6 ablation
        output_dir: Where to save processed files
        model_name: HuggingFace model name for tokenizer
        max_length: Max token sequence length
        scheme: "BIO" or "IO"
        downsample_ratio: Neg:pos ratio for training. 0 = no downsampling.
        seed: Random seed
        unanimous_only: If True, filter to 3/3 agreement only (A8)
        data_fraction: Fraction of training data to use (A5 ablation, 0.0-1.0)
    """
    rng = random.Random(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label2id = BIO_LABEL2ID if scheme == "BIO" else IO_LABEL2ID
    id2label = BIO_ID2LABEL if scheme == "BIO" else IO_ID2LABEL
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"MERGED PREPROCESSING PIPELINE")
    print(f"{'='*70}")
    print(f"  Original:         {original_path}")
    print(f"  Enriched:         {enriched_path or 'NONE (A6 ablation)'}")
    print(f"  Output:           {output_dir}")
    print(f"  Model:            {model_name}")
    print(f"  Scheme:           {scheme}")
    print(f"  Downsample ratio: {downsample_ratio}:1")
    print(f"  Unanimous only:   {unanimous_only}")
    print(f"  Data fraction:    {data_fraction}")
    print(f"  Seed:             {seed}")
    print()

    # ─── Step 1: Load threads ────────────────────────────────────────────
    print("Step 1: Loading threads...")
    original_threads = load_threads(original_path, unanimous_only=unanimous_only)
    print(f"  Original: {len(original_threads)} threads")

    enriched_threads = []
    if enriched_path and Path(enriched_path).exists():
        enriched_threads = load_threads(enriched_path, unanimous_only=unanimous_only)
        print(f"  Enriched: {len(enriched_threads)} threads")
    else:
        print(f"  Enriched: SKIPPED (no file or A6 ablation)")

    # ─── Step 2: Extract comment-level examples ──────────────────────────
    print("\nStep 2: Extracting comment-level examples...")

    original_examples = []
    for rec in original_threads:
        original_examples.extend(extract_thread_examples(rec))

    enriched_examples = []
    for rec in enriched_threads:
        enriched_examples.extend(extract_thread_examples(rec))

    orig_pos = sum(1 for e in original_examples if e["has_modification"])
    orig_neg = len(original_examples) - orig_pos
    enr_pos = sum(1 for e in enriched_examples if e["has_modification"])
    enr_neg = len(enriched_examples) - enr_pos

    print(f"  Original examples: {len(original_examples)} ({orig_pos} pos, {orig_neg} neg)")
    print(f"  Enriched examples: {len(enriched_examples)} ({enr_pos} pos, {enr_neg} neg)")

    # ─── Step 3: Split ORIGINAL at thread level (70/15/15) ───────────────
    print("\nStep 3: Thread-level split of original data (70/15/15)...")
    train_orig, val_raw, test_raw = thread_level_split(
        original_examples, train_ratio=0.70, val_ratio=0.15, seed=seed
    )

    def count_pos(exs):
        return sum(1 for e in exs if e["has_modification"])

    print(f"  Train (original): {len(train_orig)} ({count_pos(train_orig)} pos)")
    print(f"  Val:              {len(val_raw)} ({count_pos(val_raw)} pos)")
    print(f"  Test:             {len(test_raw)} ({count_pos(test_raw)} pos)")

    # ─── Step 4: Merge enriched into train ───────────────────────────────
    print("\nStep 4: Merging enriched examples into train...")
    train_combined = train_orig + enriched_examples
    rng.shuffle(train_combined)
    print(f"  Train (merged): {len(train_combined)} "
          f"({count_pos(train_combined)} pos = "
          f"{100*count_pos(train_combined)/len(train_combined):.1f}%)")

    # ─── Step 5: Downsample negatives in train ───────────────────────────
    print("\nStep 5: Downsampling negatives...")
    train_downsampled = downsample_negatives(
        train_combined, ratio=downsample_ratio, seed=seed
    )
    print(f"  Train (downsampled): {len(train_downsampled)} "
          f"({count_pos(train_downsampled)} pos = "
          f"{100*count_pos(train_downsampled)/len(train_downsampled):.1f}%)")

    # ─── Step 5b: Data fraction subsample (A5 ablation) ─────────────────
    if data_fraction < 1.0:
        n_keep = max(1, int(len(train_downsampled) * data_fraction))
        rng.shuffle(train_downsampled)
        train_downsampled = train_downsampled[:n_keep]
        print(f"  Train (A5 {data_fraction:.0%} subsample): {len(train_downsampled)} "
              f"({count_pos(train_downsampled)} pos)")

    # ─── Step 6: Tokenize + align BIO labels ────────────────────────────
    print("\nStep 6: Tokenizing and aligning BIO labels...")

    def tokenize_and_align(examples: List[Dict], split_name: str) -> List[Dict]:
        processed = []
        stats = Counter()

        for i, ex in enumerate(examples):
            if (i + 1) % 2000 == 0:
                print(f"  [{split_name}] Processing {i + 1}/{len(examples)}...")

            result, align_stats = align_example(
                ex["text"], ex["modifications"], tokenizer, label2id, max_length
            )

            if result is None:
                stats["skipped"] += 1
                continue

            stats["aligned"] += align_stats.get("aligned", 0)
            stats["hallucinated"] += align_stats.get("hallucinated", 0)
            stats["truncated"] += align_stats.get("truncated", 0)

            # Preserve metadata
            result["thread_id"] = ex["thread_id"]
            result["source_comment"] = ex["source_comment"]
            result["has_modification"] = ex["has_modification"]
            result["modifications"] = ex["modifications"]
            result["vote_method"] = ex.get("vote_method", "unknown")
            processed.append(result)

        total_spans = stats["aligned"] + stats["hallucinated"] + stats["truncated"]
        align_rate = stats["aligned"] / total_spans if total_spans > 0 else 0
        print(f"  [{split_name}] {len(processed)} examples, "
              f"alignment {stats['aligned']}/{total_spans} ({100*align_rate:.1f}%), "
              f"skipped {stats.get('skipped', 0)}")

        return processed

    train_processed = tokenize_and_align(train_downsampled, "train")
    val_processed = tokenize_and_align(val_raw, "val")
    test_processed = tokenize_and_align(test_raw, "test")

    # ─── Step 7: Compute class weights from training data ────────────────
    print("\nStep 7: Computing class weights from training distribution...")
    class_weights_dict, class_weights_list = compute_class_weights(
        train_processed, label2id, scheme
    )

    uniform_weights_list = compute_uniform_weights(label2id)

    print(f"  Auto-computed inverse-frequency weights:")
    for label_name, weight in sorted(class_weights_dict.items(),
                                      key=lambda x: label2id.get(x[0], 99)):
        lid = label2id.get(label_name, "?")
        print(f"    {label_name:<20s} (id={lid}): {weight:.4f}")

    # ─── Step 8: Save everything ─────────────────────────────────────────
    print("\nStep 8: Saving processed data...")

    def save_jsonl(data: List[Dict], path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    save_jsonl(train_processed, out / "train_merged.jsonl")
    save_jsonl(val_processed, out / "val.jsonl")
    save_jsonl(test_processed, out / "test.jsonl")

    # Save statistics + class weights
    train_label_counts = Counter()
    for ex in train_processed:
        for lid in ex["labels"]:
            if lid != -100:
                train_label_counts[lid] += 1

    stats = {
        "config": {
            "original_path": original_path,
            "enriched_path": enriched_path,
            "model": model_name,
            "scheme": scheme,
            "downsample_ratio": downsample_ratio,
            "unanimous_only": unanimous_only,
            "data_fraction": data_fraction,
            "seed": seed,
        },
        "counts": {
            "original_threads": len(original_threads),
            "enriched_threads": len(enriched_threads),
            "train_examples": len(train_processed),
            "val_examples": len(val_processed),
            "test_examples": len(test_processed),
            "train_positive": count_pos(train_processed),
            "val_positive": count_pos(val_processed),
            "test_positive": count_pos(test_processed),
            "train_positive_rate": round(
                count_pos(train_processed) / max(1, len(train_processed)), 4
            ),
            "val_positive_rate": round(
                count_pos(val_processed) / max(1, len(val_processed)), 4
            ),
            "test_positive_rate": round(
                count_pos(test_processed) / max(1, len(test_processed)), 4
            ),
        },
        "label_distribution": {
            id2label.get(lid, f"UNK_{lid}"): count
            for lid, count in sorted(train_label_counts.items())
        },
        "class_weights": {
            "inverse_frequency": class_weights_dict,
            "inverse_frequency_list": class_weights_list,
            "uniform": uniform_weights_list,
            "description": (
                "inverse_frequency: auto-computed as total/(num_classes*count_i), "
                "capped at 20.0. Use with --class-weights flag in train_student.py. "
                "uniform: all 1.0, for A1 ablation comparison."
            ),
        },
    }

    with open(out / "stats_merged.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {out / 'train_merged.jsonl'} ({len(train_processed)} examples)")
    print(f"  Saved: {out / 'val.jsonl'} ({len(val_processed)} examples)")
    print(f"  Saved: {out / 'test.jsonl'} ({len(test_processed)} examples)")
    print(f"  Saved: {out / 'stats_merged.json'} (class weights + stats)")

    # ─── Eye test ────────────────────────────────────────────────────────
    print(f"\n--- EYE TEST (5 random positive training examples) ---")
    pos_train = [e for e in train_processed if e["has_modification"]]
    samples = random.sample(pos_train, min(5, len(pos_train)))
    for ex in samples:
        visualize_alignment(ex, tokenizer, id2label)

    # ─── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"  Train: {len(train_processed)} examples "
          f"({count_pos(train_processed)} pos = "
          f"{stats['counts']['train_positive_rate']*100:.1f}%)")
    print(f"  Val:   {len(val_processed)} examples "
          f"({count_pos(val_processed)} pos = "
          f"{stats['counts']['val_positive_rate']*100:.1f}%)")
    print(f"  Test:  {len(test_processed)} examples "
          f"({count_pos(test_processed)} pos = "
          f"{stats['counts']['test_positive_rate']*100:.1f}%)")
    print(f"\n  Class weights saved to stats_merged.json")
    print(f"  Use in training:")
    print(f"    python -m src.models.train_student \\")
    print(f"        --class-weights-file {out / 'stats_merged.json'} \\")
    print(f"        --focal-loss --focal-gamma 2.0")
    print(f"{'='*70}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merged preprocessing: original + enriched → train/val/test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run:
  python -m src.preprocessing.prepare_data_merged \\
      --original data/silver_labels/teacher_output.jsonl \\
      --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \\
      --output-dir data/processed --downsample-ratio 3.0

  # Ablation A2 (downsample sweep):
  for ratio in 2.0 3.0 4.0; do
      python -m src.preprocessing.prepare_data_merged \\
          --original data/silver_labels/teacher_output.jsonl \\
          --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \\
          --output-dir data/processed_ds${ratio} \\
          --downsample-ratio ${ratio}
  done

  # Ablation A5 (data size):
  for frac in 0.25 0.50 0.75; do
      python -m src.preprocessing.prepare_data_merged \\
          --original data/silver_labels/teacher_output.jsonl \\
          --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \\
          --output-dir data/processed_frac${frac} \\
          --data-fraction ${frac}
  done

  # Ablation A6 (no enriched):
  python -m src.preprocessing.prepare_data_merged \\
      --original data/silver_labels/teacher_output.jsonl \\
      --output-dir data/processed_no_enriched

  # Ablation A8 (unanimous only):
  python -m src.preprocessing.prepare_data_merged \\
      --original data/silver_labels/teacher_output.jsonl \\
      --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \\
      --output-dir data/processed_unanimous \\
      --unanimous-only
        """
    )

    parser.add_argument("--original", required=True,
                        help="Path to teacher_output.jsonl (10K original threads)")
    parser.add_argument("--enriched", default=None,
                        help="Path to enriched JSONL (5K threads). Omit for A6 ablation.")
    parser.add_argument("--output-dir", "-o", default="data/processed",
                        help="Output directory for processed files")
    parser.add_argument("--model", default="dicta-il/dictabert",
                        help="HuggingFace model for tokenizer (default: dicta-il/dictabert)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max token sequence length (default: 128)")
    parser.add_argument("--scheme", choices=["BIO", "IO"], default="BIO",
                        help="Tagging scheme: BIO or IO (default: BIO)")
    parser.add_argument("--downsample-ratio", type=float, default=3.0,
                        help="Neg:pos ratio for train downsampling. "
                             "0 = no downsampling. (default: 3.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--unanimous-only", action="store_true",
                        help="A8 ablation: keep only 3/3 unanimous agreement threads")
    parser.add_argument("--data-fraction", type=float, default=1.0,
                        help="A5 ablation: fraction of training data to use (0.0-1.0)")

    args = parser.parse_args()

    if args.data_fraction <= 0 or args.data_fraction > 1.0:
        parser.error(f"--data-fraction must be in (0.0, 1.0], got {args.data_fraction}")

    process_merged(
        original_path=args.original,
        enriched_path=args.enriched,
        output_dir=args.output_dir,
        model_name=args.model,
        max_length=args.max_length,
        scheme=args.scheme,
        downsample_ratio=args.downsample_ratio,
        seed=args.seed,
        unanimous_only=args.unanimous_only,
        data_fraction=args.data_fraction,
    )


if __name__ == "__main__":
    main()
