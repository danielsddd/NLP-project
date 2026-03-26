#!/usr/bin/env python3
"""
Thread-Aware Preprocessing
==========================
For reply-sourced examples, prepends the parent question as context:
  "[שאלה] <question> [תשובה] <reply>"

CRITICAL DESIGN: We tokenize the FULL text (with prefix) so the model
sees the question context. But we align spans against the REPLY TEXT
ONLY, then shift token indices by the prefix length. This avoids the
bug where find_span_in_text matches a duplicate span in the question.

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
    find_span_in_text,
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
import re


# =============================================================================
# CONSTANTS
# =============================================================================

QUESTION_PREFIX = "[שאלה] "
ANSWER_PREFIX = " [תשובה] "


# =============================================================================
# THREAD-AWARE ALIGNMENT
# =============================================================================

def align_thread_aware_example(
    full_text: str,
    reply_text: str,
    prefix_len: int,
    modifications: List[Dict],
    tokenizer,
    label2id: Dict[str, int],
    max_length: int = 128,
):
    """Align spans to tokens in thread-aware text.

    The key insight: we search for spans in reply_text (avoiding false
    matches in the question), then shift the character offsets by
    prefix_len to get positions in full_text. Token alignment then
    proceeds normally on full_text.

    Args:
        full_text: "[שאלה] <question> [תשובה] <reply>"
        reply_text: The reply portion only
        prefix_len: len("[שאלה] <question> [תשובה] ") — chars before reply
        modifications: List of {span, aspect, confidence}
        tokenizer: HuggingFace tokenizer
        label2id: BIO or IO label map
        max_length: Max token sequence length

    Returns:
        dict with input_ids, attention_mask, labels, text — or None
    """
    if not full_text or not full_text.strip():
        return None

    # Tokenize the FULL text (model sees question + reply)
    encoding = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors=None,
    )

    offsets = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    use_bio = any(k.startswith("B-") for k in label2id)

    # Initialize labels
    labels_str = []
    for offset in offsets:
        if offset[0] == 0 and offset[1] == 0:
            labels_str.append(None)  # special token → -100
        else:
            labels_str.append("O")

    max_char = max((o[1] for o in offsets), default=0)

    aligned = 0
    hallucinated = 0
    truncated = 0

    for mod in modifications:
        span = mod.get("span", "")
        aspect = mod.get("aspect", "")

        if not span or aspect not in VALID_ASPECTS:
            hallucinated += 1
            continue

        # CRITICAL: Search for span in REPLY TEXT ONLY
        result = find_span_in_text(span, reply_text)
        if result is None:
            hallucinated += 1
            continue

        reply_start, reply_end = result

        # Shift to full_text coordinates
        start_char = reply_start + prefix_len
        end_char = reply_end + prefix_len

        # Verify the span actually matches at this position in full_text
        actual = full_text[start_char:end_char]
        expected = reply_text[reply_start:reply_end]
        if actual != expected:
            # Offset mismatch — prefix_len calculation was wrong, skip
            hallucinated += 1
            continue

        if start_char >= max_char:
            truncated += 1
            continue

        # Map to tokens
        first_token = True
        for tok_idx, offset in enumerate(offsets):
            tok_start, tok_end = offset[0], offset[1]
            if tok_start == 0 and tok_end == 0:
                continue
            if tok_start < end_char and tok_end > start_char:
                if labels_str[tok_idx] is None:
                    continue
                if labels_str[tok_idx] != "O":
                    continue  # already tagged by earlier span
                if use_bio:
                    labels_str[tok_idx] = f"B-{aspect}" if first_token else f"I-{aspect}"
                else:
                    labels_str[tok_idx] = f"I-{aspect}"
                first_token = False

        if not first_token:
            aligned += 1
        else:
            hallucinated += 1

    # Convert labels to IDs
    label_ids = []
    for label in labels_str:
        if label is None:
            label_ids.append(-100)
        else:
            label_ids.append(label2id.get(label, 0))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": label_ids,
        "text": full_text,
        "_align_stats": {"aligned": aligned, "hallucinated": hallucinated, "truncated": truncated},
    }


def align_plain_example(text, modifications, tokenizer, label2id, max_length=128):
    """Standard alignment (no thread context). Delegates to prepare_data."""
    from src.preprocessing.prepare_data import align_example as _align
    result = _align(text, modifications, tokenizer, label2id, max_length)
    if result is None:
        return None
    # align_example returns (result_dict, stats_dict) tuple
    if isinstance(result, tuple):
        return result[0]
    return result


# =============================================================================
# THREAD-AWARE EXAMPLE EXTRACTION
# =============================================================================

def extract_thread_aware_examples(record: Dict) -> List[Dict]:
    """Extract examples, prepending question context for reply-sourced mods.

    Returns list of dicts. Reply-sourced examples have extra fields:
        reply_text: the raw reply text (for safe span alignment)
        prefix_len: character count before the reply in full_text
        is_thread_aware: True
    """
    thread_id = record["thread_id"]
    final_label = record.get("final_label") or record.get("teacher_output") or {}
    has_mod = final_label.get("has_modification", False)
    vote_method = record.get("vote_method", "unknown")
    top_text = record.get("top_comment_text", "")

    examples = []

    if not has_mod:
        if top_text and top_text.strip():
            examples.append({
                "text": top_text,
                "modifications": [],
                "thread_id": thread_id,
                "source_comment": "top",
                "has_modification": False,
                "vote_method": vote_method,
                "is_thread_aware": False,
            })
        return examples

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
            text = top_text
            if not text or not text.strip():
                continue
            examples.append({
                "text": text,
                "modifications": mods,
                "thread_id": thread_id,
                "source_comment": src,
                "has_modification": True,
                "vote_method": vote_method,
                "is_thread_aware": False,
            })

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

            if top_text and top_text.strip():
                prefix = f"{QUESTION_PREFIX}{top_text}{ANSWER_PREFIX}"
                full_text = f"{prefix}{reply_text}"
                prefix_len = len(prefix)
            else:
                full_text = reply_text
                prefix_len = 0

            examples.append({
                "text": full_text,
                "reply_text": reply_text,
                "prefix_len": prefix_len,
                "modifications": mods,
                "thread_id": thread_id,
                "source_comment": src,
                "has_modification": True,
                "vote_method": vote_method,
                "is_thread_aware": prefix_len > 0,
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

    # Load
    print("\nStep 1: Loading threads...")
    original_threads = load_threads(original_path)
    print(f"  Original: {len(original_threads)} threads")

    enriched_threads = []
    if enriched_path and Path(enriched_path).exists():
        enriched_threads = load_threads(enriched_path)
        print(f"  Enriched: {len(enriched_threads)} threads")

    # Extract
    print("\nStep 2: Extracting thread-aware examples...")
    original_examples = []
    for rec in original_threads:
        original_examples.extend(extract_thread_aware_examples(rec))
    enriched_examples = []
    for rec in enriched_threads:
        enriched_examples.extend(extract_thread_aware_examples(rec))

    orig_pos = sum(1 for e in original_examples if e["has_modification"])
    enr_pos = sum(1 for e in enriched_examples if e["has_modification"])
    thread_aware_count = sum(1 for e in original_examples + enriched_examples if e.get("is_thread_aware"))
    print(f"  Original: {len(original_examples)} ({orig_pos} pos)")
    print(f"  Enriched: {len(enriched_examples)} ({enr_pos} pos)")
    print(f"  Thread-aware examples: {thread_aware_count}")

    # Split
    print("\nStep 3: Thread-level split (70/15/15)...")
    train_orig, val_raw, test_raw = thread_level_split(
        original_examples, train_ratio=0.70, val_ratio=0.15, seed=seed
    )

    train_combined = train_orig + enriched_examples
    rng.shuffle(train_combined)
    train_combined = downsample_negatives(train_combined, downsample_ratio, seed)

    # Align
    print("\nStep 4: Tokenizing + aligning...")
    total_stats = {"aligned": 0, "hallucinated": 0, "truncated": 0}

    def process_examples(examples, desc):
        processed = []
        skip = 0
        for ex in examples:
            if ex.get("is_thread_aware") and "reply_text" in ex:
                # Thread-aware: align spans against reply portion only
                result = align_thread_aware_example(
                    full_text=ex["text"],
                    reply_text=ex["reply_text"],
                    prefix_len=ex["prefix_len"],
                    modifications=ex["modifications"],
                    tokenizer=tokenizer,
                    label2id=label2id,
                    max_length=max_length,
                )
            else:
                # Standard alignment
                result = align_plain_example(
                    ex["text"], ex["modifications"], tokenizer, label2id, max_length
                )

            if result is None:
                skip += 1
                continue

            # Collect stats
            stats = result.pop("_align_stats", {})
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)

            result["thread_id"] = ex["thread_id"]
            result["source_comment"] = ex["source_comment"]
            result["has_modification"] = ex["has_modification"]
            result["vote_method"] = ex.get("vote_method", "unknown")
            result["is_thread_aware"] = ex.get("is_thread_aware", False)

            # Remove helper fields from output
            result.pop("reply_text", None)
            result.pop("prefix_len", None)

            processed.append(result)
        print(f"  {desc}: {len(processed)} OK, {skip} skipped")
        return processed

    train_processed = process_examples(train_combined, "Train")
    val_processed = process_examples(val_raw, "Val")
    test_processed = process_examples(test_raw, "Test")

    print(f"\n  Alignment stats: {total_stats}")

    # Save
    def save_jsonl(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
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
        "downsample_ratio": downsample_ratio,
        "counts": {
            "train": len(train_processed),
            "val": len(val_processed),
            "test": len(test_processed),
            "train_positive_rate": count_pos(train_processed) / max(1, len(train_processed)),
            "val_positive_rate": count_pos(val_processed) / max(1, len(val_processed)),
            "test_positive_rate": count_pos(test_processed) / max(1, len(test_processed)),
            "thread_aware_examples": sum(1 for e in train_processed if e.get("is_thread_aware")),
        },
        "class_weights": {id2label[i]: w for i, w in enumerate(weights)},
        "alignment_stats": total_stats,
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
