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
    extract_thread_examples,
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
    """
    use_bio = "B-SUBSTITUTION" in label2id

    encoding = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"]

    # None = special token (→ -100), "O" = non-entity
    labels_str = []
    for start, end in offset_mapping:
        if start == 0 and end == 0:
            labels_str.append(None)
        else:
            labels_str.append("O")

    aligned = 0
    hallucinated = 0
    truncated = 0

    for mod in modifications:
        aspect = mod.get("aspect", "")
        if aspect not in VALID_ASPECTS:
            continue

        span_text = mod.get("span", "")
        if not span_text or not span_text.strip():
            continue

        # Search in reply_text ONLY (avoids false matches in the prefix question)
        pos = find_span_in_text(span_text, reply_text)
        if pos is None:
            hallucinated += 1
            continue

        span_start_in_reply, span_end_in_reply = pos
        start_char = span_start_in_reply + prefix_len
        end_char = span_end_in_reply + prefix_len

        max_char_in_encoding = max(
            (end for start, end in offset_mapping if end > 0), default=0
        )
        if start_char >= max_char_in_encoding:
            truncated += 1
            continue

        first_token = True
        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == 0 and tok_end == 0:
                continue
            if tok_start < end_char and tok_end > start_char:
                if labels_str[tok_idx] is None:
                    continue
                if labels_str[tok_idx] != "O":
                    continue
                if use_bio:
                    labels_str[tok_idx] = f"B-{aspect}" if first_token else f"I-{aspect}"
                else:
                    labels_str[tok_idx] = f"I-{aspect}"
                first_token = False

        if not first_token:
            aligned += 1
        else:
            hallucinated += 1

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
    if isinstance(result, tuple):
        return result[0]
    return result


# =============================================================================
# THREAD-AWARE EXAMPLE EXTRACTION
# =============================================================================

def _get_top_text(record: Dict) -> str:
    """
    Try every plausible field name to find the parent/question text for
    thread-aware prefixing. Returns empty string if nothing is found —
    callers must handle that gracefully (no prefix applied).
    """
    # Strategy 0: check top_comment_text (the actual key used in your JSONL)
    if record.get("top_comment_text") and isinstance(record.get("top_comment_text"), str):
        t = record["top_comment_text"].strip()
        if t: return t

    # Strategy 1: explicit top_comment field (dict or string)
    tc = record.get("top_comment")
    if isinstance(tc, dict):
        t = tc.get("text", "").strip()
        if t:
            return t
    elif isinstance(tc, str) and tc.strip():
        return tc.strip()

    # Strategy 2: comments list — find the one flagged as top
    for c in record.get("comments", []):
        if isinstance(c, dict) and c.get("is_top_comment", False):
            t = c.get("text", "").strip()
            if t:
                return t

    # Strategy 3: first entry in comments list as last resort
    comments = record.get("comments", [])
    if comments and isinstance(comments[0], dict):
        t = comments[0].get("text", "").strip()
        if t:
            return t

    return ""


def extract_thread_aware_examples(record: Dict) -> List[Dict]:
    """
    Delegate data extraction to extract_thread_examples (already correct
    for the actual JSONL format), then enrich each example with the
    thread-aware prefix if a parent/question text is found in the record.

    This avoids re-implementing the data-format-specific extraction logic
    and guarantees we get the same example count as the non-thread-aware
    preprocessing.
    """
    # Use the existing, tested extraction — handles all JSONL quirks
    base_examples = extract_thread_examples(record)
    if not base_examples:
        return []

    top_text = _get_top_text(record)

    result = []
    for ex in base_examples:
        reply_text = ex.get("text", "")

        # Add prefix only when: we have a question AND the reply isn't the
        # top comment itself (guard against prefixing a comment with itself)
        if top_text and reply_text.strip() != top_text.strip():
            prefix = f"{QUESTION_PREFIX}{top_text}{ANSWER_PREFIX}"
            full_text = f"{prefix}{reply_text}"
            prefix_len = len(prefix)
            is_thread_aware = True
        else:
            full_text = reply_text
            prefix_len = 0
            is_thread_aware = False

        result.append({
            **ex,                          # thread_id, source_comment, modifications,
                                           # has_modification, vote_method — all from base
            "text": full_text,             # overwrite with prefixed version
            "reply_text": reply_text,      # original reply text for span search
            "prefix_len": prefix_len,      # char offset to shift spans
            "is_thread_aware": is_thread_aware,
        })

    return result


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
    thread_aware_count = sum(
        1 for e in original_examples + enriched_examples if e.get("is_thread_aware")
    )
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
                result = align_plain_example(
                    ex["text"], ex["modifications"], tokenizer, label2id, max_length
                )

            if result is None:
                skip += 1
                continue

            stats = result.pop("_align_stats", {})
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)

            result["thread_id"] = ex["thread_id"]
            result["source_comment"] = ex["source_comment"]
            result["has_modification"] = ex["has_modification"]
            result["vote_method"] = ex.get("vote_method", "unknown")
            result["is_thread_aware"] = ex.get("is_thread_aware", False)

            result.pop("reply_text", None)
            result.pop("prefix_len", None)

            processed.append(result)
        print(f"  {desc}: {len(processed)} OK, {skip} skipped")
        return processed

    train_processed = process_examples(train_combined, "Train")
    val_processed = process_examples(val_raw, "Val")
    test_processed = process_examples(test_raw, "Test")

    print(f"\n  Alignment stats: {total_stats}")

    # ── Save JSONL files ──────────────────────────────────────────────────────
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

    # ── Class weights + stats ─────────────────────────────────────────────────
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

    # ── FIX: label mapping files required by train_student.py ────────────────
    label2id_out = {k: int(v) for k, v in label2id.items()}
    id2label_out = {str(v): k for k, v in label2id.items()}
    with open(out / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id_out, f, indent=2, ensure_ascii=False)
    with open(out / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label_out, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out}/")
    print(f"  train_merged.jsonl : {len(train_processed)} ({count_pos(train_processed)} pos)")
    print(f"  val.jsonl          : {len(val_processed)} ({count_pos(val_processed)} pos)")
    print(f"  test.jsonl         : {len(test_processed)} ({count_pos(test_processed)} pos)")
    print(f"  stats_merged.json  : class weights + counts")
    print(f"  label2id.json      : {scheme} mapping ({len(label2id)} labels)")
    print(f"  id2label.json      : {scheme} mapping ({len(id2label)} labels)")


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