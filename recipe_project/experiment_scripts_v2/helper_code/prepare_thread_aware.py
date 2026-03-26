#!/usr/bin/env python3
"""
Thread-Aware Preprocessing
==========================
Like prepare_data_merged.py, but for reply-sourced examples, prepends
the parent comment (question) as context:

  Input text (reply only):    "כן בטח, אותה כמות"
  Output text (with context): "[שאלה] אפשר במקום חמאה שמן קוקוס? [תשובה] כן בטח, אותה כמות"

This gives the student model the question→answer context that the teacher
model had when generating silver labels.

For top-comment-sourced examples, the text is unchanged.

The span offsets are adjusted to account for the prepended question text.

Usage:
    python -m src.preprocessing.prepare_thread_aware \
        --original data/silver_labels/teacher_output.jsonl \
        --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \
        --output-dir data/processed_thread_aware \
        --model dicta-il/dictabert \
        --downsample-ratio 3.0
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

from src.preprocessing.prepare_data import (
    align_example,
    normalize_hebrew,
    visualize_alignment,
    thread_level_split,
    BIO_LABEL2ID,
    BIO_ID2LABEL,
    IO_LABEL2ID,
    IO_ID2LABEL,
    VALID_ASPECTS,
)
from src.preprocessing.prepare_data_merged import (
    load_threads,
    downsample_negatives,
    compute_class_weights,
)
from transformers import AutoTokenizer


# =============================================================================
# THREAD-AWARE EXAMPLE EXTRACTION
# =============================================================================

QUESTION_PREFIX = "[שאלה] "
ANSWER_PREFIX = " [תשובה] "


def extract_thread_aware_examples(record: Dict) -> List[Dict]:
    """Like extract_thread_examples, but prepend question context for replies.

    For reply-sourced modifications:
        text = "[שאלה] <top_comment> [תשובה] <reply_text>"
        span offsets are shifted by len(prefix + top_comment + separator)

    For top-comment-sourced modifications:
        text = <top_comment>  (unchanged)
    """
    thread_id = record["thread_id"]
    final_label = record.get("final_label") or record.get("teacher_output") or {}
    has_mod = final_label.get("has_modification", False)
    vote_method = record.get("vote_method", "unknown")
    top_text = record.get("top_comment_text", "")

    examples = []

    if not has_mod:
        # Negative: just top comment, no context needed
        if top_text and top_text.strip():
            examples.append({
                "text": top_text,
                "modifications": [],
                "thread_id": thread_id,
                "source_comment": "top",
                "has_modification": False,
                "vote_method": vote_method,
            })
        return examples

    # Group mods by source
    mods_by_source = defaultdict(list)
    for mod in final_label.get("modifications", []):
        src = mod.get("source_comment", "top")
        if mod.get("aspect") not in VALID_ASPECTS:
            continue
        mods_by_source[src].append({
            "span": mod.get("span", ""),
            "aspect": mod["aspect"],
            "confidence": mod.get("confidence", 0.0),
        })

    for src, mods in mods_by_source.items():
        if src == "top":
            # Top comment: no context needed
            text = top_text
        elif src.startswith("reply_"):
            try:
                idx = int(src.split("_")[1]) - 1
                replies = record.get("replies_texts", [])
                if 0 <= idx < len(replies):
                    reply_text = replies[idx]
                else:
                    continue
            except (ValueError, IndexError):
                continue

            if not reply_text or not reply_text.strip():
                continue

            # THREAD-AWARE: prepend question context
            if top_text and top_text.strip():
                text = f"{QUESTION_PREFIX}{top_text}{ANSWER_PREFIX}{reply_text}"
            else:
                text = reply_text
        else:
            continue

        if not text or not text.strip():
            continue

        examples.append({
            "text": text,
            "modifications": mods,
            "thread_id": thread_id,
            "source_comment": src,
            "has_modification": True,
            "vote_method": vote_method,
        })

    return examples


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_thread_aware(
    original_path: str,
    enriched_path: Optional[str],
    output_dir: str,
    model_name: str = "dicta-il/dictabert",
    max_length: int = 128,
    scheme: str = "BIO",
    downsample_ratio: float = 3.0,
    seed: int = 42,
):
    """Thread-aware preprocessing pipeline.

    Same as prepare_data_merged but uses extract_thread_aware_examples
    instead of extract_thread_examples.
    """
    rng = random.Random(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    label2id = BIO_LABEL2ID if scheme == "BIO" else IO_LABEL2ID
    id2label = BIO_ID2LABEL if scheme == "BIO" else IO_ID2LABEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Thread-Aware Preprocessing")
    print(f"  Model: {model_name}")
    print(f"  Scheme: {scheme}")
    print(f"  Max length: {max_length}")
    print(f"  Downsample ratio: {downsample_ratio}")

    # Load data
    print("\nStep 1: Loading threads...")
    original_threads = load_threads(original_path)
    print(f"  Original: {len(original_threads)} threads")

    enriched_threads = []
    if enriched_path and Path(enriched_path).exists():
        enriched_threads = load_threads(enriched_path)
        print(f"  Enriched: {len(enriched_threads)} threads")

    # Extract with thread-aware context
    print("\nStep 2: Extracting thread-aware examples...")
    original_examples = []
    for rec in original_threads:
        original_examples.extend(extract_thread_aware_examples(rec))

    enriched_examples = []
    for rec in enriched_threads:
        enriched_examples.extend(extract_thread_aware_examples(rec))

    orig_pos = sum(1 for e in original_examples if e["has_modification"])
    enr_pos = sum(1 for e in enriched_examples if e["has_modification"])
    print(f"  Original: {len(original_examples)} ({orig_pos} pos)")
    print(f"  Enriched: {len(enriched_examples)} ({enr_pos} pos)")

    # Thread-level split on original
    print("\nStep 3: Thread-level split (70/15/15)...")
    train_orig, val_raw, test_raw = thread_level_split(
        original_examples, train_ratio=0.70, val_ratio=0.15, seed=seed
    )

    # Merge enriched into train
    train_combined = train_orig + enriched_examples
    rng.shuffle(train_combined)

    # Downsample
    train_combined = downsample_negatives(train_combined, downsample_ratio, seed)

    # Tokenize + align
    print("\nStep 4: Tokenizing + aligning BIO labels...")

    def process_examples(examples, desc):
        processed = []
        skip = 0
        for ex in examples:
            result = align_example(
                ex["text"], ex["modifications"], tokenizer, label2id, max_length
            )
            if result is None:
                skip += 1
                continue
            result["thread_id"] = ex["thread_id"]
            result["source_comment"] = ex["source_comment"]
            result["has_modification"] = ex["has_modification"]
            result["vote_method"] = ex.get("vote_method", "unknown")
            processed.append(result)
        print(f"  {desc}: {len(processed)} OK, {skip} skipped")
        return processed

    train_processed = process_examples(train_combined, "Train")
    val_processed = process_examples(val_raw, "Val")
    test_processed = process_examples(test_raw, "Test")

    # Save
    def save_jsonl(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                # Convert tensors/arrays to lists
                serializable = {}
                for k, v in item.items():
                    if hasattr(v, 'tolist'):
                        serializable[k] = v.tolist()
                    else:
                        serializable[k] = v
                f.write(json.dumps(serializable, ensure_ascii=False) + "\n")

    save_jsonl(train_processed, out / "train_merged.jsonl")
    save_jsonl(val_processed, out / "val.jsonl")
    save_jsonl(test_processed, out / "test.jsonl")

    # Class weights
    weights = compute_class_weights(train_processed, label2id)
    count_pos = lambda exs: sum(1 for e in exs if e["has_modification"])
    stats = {
        "preprocessing": "thread_aware",
        "scheme": scheme,
        "model": model_name,
        "counts": {
            "train": len(train_processed),
            "val": len(val_processed),
            "test": len(test_processed),
            "train_positive_rate": count_pos(train_processed) / max(1, len(train_processed)),
        },
        "class_weights": {id2label[i]: w for i, w in enumerate(weights)},
    }
    with open(out / "stats_merged.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out}/")
    print(f"  Train: {len(train_processed)} ({count_pos(train_processed)} pos)")
    print(f"  Val:   {len(val_processed)} ({count_pos(val_processed)} pos)")
    print(f"  Test:  {len(test_processed)} ({count_pos(test_processed)} pos)")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Thread-aware preprocessing: prepend question context to replies"
    )
    parser.add_argument("--original", required=True)
    parser.add_argument("--enriched", default=None)
    parser.add_argument("--output-dir", "-o", default="data/processed_thread_aware")
    parser.add_argument("--model", default="dicta-il/dictabert")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--scheme", choices=["BIO", "IO"], default="BIO")
    parser.add_argument("--downsample-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    process_thread_aware(
        original_path=args.original,
        enriched_path=args.enriched,
        output_dir=args.output_dir,
        model_name=args.model,
        max_length=args.max_length,
        scheme=args.scheme,
        downsample_ratio=args.downsample_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
