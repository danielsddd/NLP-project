#!/usr/bin/env python3
"""
Tokenize gold_enriched_threads.jsonl for thread-aware IO evaluation.
====================================================================
For reply examples: uses thread_aware_text ("[שאלה] Q [תשובה] R") and
aligns spans against reply_text only, shifted by prefix_len.
For top comments: uses comment_text directly (same as standard gold).

Output matches the format expected by evaluate_crf.py.

Usage:
    python scripts/local/prepare_gold_thread_aware.py

    # Or with explicit paths:
    python scripts/local/prepare_gold_thread_aware.py \
        --gold data/gold_validation/gold_enriched_threads.jsonl \
        --output data/gold_validation/gold_tokenized_dictabert_io_thread.jsonl \
        --model dicta-il/dictabert
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter

# Find project root
def _find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "src" / "preprocessing" / "prepare_data.py").exists():
            return parent
    raise RuntimeError(f"Could not locate project root from {start}")

PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer
from src.preprocessing.prepare_data import (
    find_span_in_text,
    normalize_hebrew,
    IO_LABEL2ID,
    VALID_ASPECTS,
)


def align_thread_aware_gold(
    full_text: str,
    reply_text: str,
    prefix_len: int,
    modifications: list,
    tokenizer,
    label2id: dict,
    max_length: int = 128,
):
    """Align spans for thread-aware examples.
    
    Spans are searched in reply_text, then character offsets are shifted
    by prefix_len to map into full_text. Tokenization runs on full_text.
    """
    encoding = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors=None,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offsets = encoding["offset_mapping"]

    # Initialize all real tokens as O, special tokens as -100
    labels = []
    for i, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:
            labels.append(-100)  # special token
        else:
            labels.append(label2id["O"])

    aligned = 0
    hallucinated = 0

    for mod in modifications:
        span_text = mod.get("span", "").strip()
        aspect = mod.get("aspect", "").upper().strip()
        if not span_text or aspect not in VALID_ASPECTS:
            continue

        # Find span in reply_text (not full_text — avoids false matches in question)
        # NOTE: find_span_in_text signature is (span, text) — span first!
        found = find_span_in_text(span_text, reply_text)
        if found is None:
            # Fallback: try in full text (some spans may cross boundaries)
            found_full = find_span_in_text(span_text, full_text)
            if found_full is not None:
                # Use directly — no prefix shift needed
                char_start, char_end = found_full
                tagged = False
                for tok_idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_start == 0 and tok_end == 0:
                        continue
                    if labels[tok_idx] == -100:
                        continue
                    if tok_start < char_end and tok_end > char_start:
                        if labels[tok_idx] == label2id["O"]:
                            labels[tok_idx] = label2id.get(f"I-{aspect}", 0)
                            tagged = True
                if tagged:
                    aligned += 1
                else:
                    hallucinated += 1
                continue
            else:
                hallucinated += 1
                continue

        # Shift offsets to full_text coordinates
        char_start = found[0] + prefix_len
        char_end = found[1] + prefix_len

        # Map character positions to token indices
        tagged = False
        for tok_idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:
                continue
            if labels[tok_idx] == -100:
                continue
            if tok_start < char_end and tok_end > char_start:
                if labels[tok_idx] == label2id["O"]:
                    labels[tok_idx] = label2id.get(f"I-{aspect}", 0)
                    tagged = True

        if tagged:
            aligned += 1
        else:
            hallucinated += 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "text": full_text,
    }, {"aligned": aligned, "hallucinated": hallucinated}


def align_plain_gold(text, modifications, tokenizer, label2id, max_length=128):
    """Standard alignment for top-comment examples (no thread context)."""
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors=None,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offsets = encoding["offset_mapping"]

    labels = []
    for i, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:
            labels.append(-100)
        else:
            labels.append(label2id["O"])

    aligned = 0
    hallucinated = 0

    for mod in modifications:
        span_text = mod.get("span", "").strip()
        aspect = mod.get("aspect", "").upper().strip()
        if not span_text or aspect not in VALID_ASPECTS:
            continue

        found = find_span_in_text(span_text, text)
        if found is None:
            hallucinated += 1
            continue

        char_start, char_end = found
        tagged = False
        for tok_idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:
                continue
            if labels[tok_idx] == -100:
                continue
            if tok_start < char_end and tok_end > char_start:
                if labels[tok_idx] == label2id["O"]:
                    labels[tok_idx] = label2id.get(f"I-{aspect}", 0)
                    tagged = True

        if tagged:
            aligned += 1
        else:
            hallucinated += 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "text": text,
    }, {"aligned": aligned, "hallucinated": hallucinated}


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize enriched gold for thread-aware IO evaluation"
    )
    parser.add_argument(
        "--gold",
        default="data/gold_validation/gold_enriched_threads.jsonl",
        help="Path to enriched gold file",
    )
    parser.add_argument(
        "--output",
        default="data/gold_validation/gold_tokenized_dictabert_io_thread.jsonl",
        help="Output path for tokenized gold",
    )
    parser.add_argument(
        "--model",
        default="dicta-il/dictabert",
        help="HuggingFace model name for tokenizer",
    )
    parser.add_argument(
        "--max-length", type=int, default=128,
        help="Max token sequence length",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    output_path = Path(args.output)

    if not gold_path.exists():
        print(f"ERROR: gold file not found: {gold_path}")
        sys.exit(1)

    print(f"{'=' * 70}")
    print(f"  THREAD-AWARE GOLD TOKENIZER (IO scheme)")
    print(f"{'=' * 70}")
    print(f"  Gold file:   {gold_path}")
    print(f"  Tokenizer:   {args.model}")
    print(f"  Output:      {output_path}")
    print(f"  Max length:  {args.max_length}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    label2id = IO_LABEL2ID

    # Load records
    records = []
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"\n  Loaded {len(records)} gold records")

    # Process
    output_records = []
    stats = Counter()
    total_aligned = 0
    total_hallucinated = 0

    for rec in records:
        is_reply = rec.get("is_reply", False)
        gold_mods = rec.get("gold_modifications", [])

        # Validate modifications
        clean_mods = []
        for m in gold_mods:
            span = (m.get("span") or "").strip()
            aspect = (m.get("aspect") or "").upper().strip()
            if span and aspect in VALID_ASPECTS:
                clean_mods.append({"span": span, "aspect": aspect})

        if is_reply and rec.get("thread_aware_text"):
            # Thread-aware: use [שאלה]...[תשובה]... format
            full_text = rec["thread_aware_text"]
            reply_text = rec.get("reply_text", rec.get("comment_text", ""))
            prefix_len = rec.get("prefix_len", 0)

            result, align_stats = align_thread_aware_gold(
                full_text, reply_text, prefix_len,
                clean_mods, tokenizer, label2id, args.max_length,
            )
            stats["thread_aware"] += 1
        else:
            # Standard: use comment_text directly
            text = (rec.get("comment_text") or "").strip()
            if not text:
                stats["skipped_empty"] += 1
                continue

            result, align_stats = align_plain_gold(
                text, clean_mods, tokenizer, label2id, args.max_length,
            )
            stats["plain"] += 1

        total_aligned += align_stats["aligned"]
        total_hallucinated += align_stats["hallucinated"]

        # Add metadata
        result["thread_id"] = rec.get("thread_id", "")
        result["has_modification"] = rec.get("has_modification", False)
        result["comment_position"] = rec.get("comment_position", "top")
        result["is_reply"] = is_reply
        result["gold_modifications"] = clean_mods

        output_records.append(result)
        stats["success"] += 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in output_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pos = sum(1 for r in output_records if r.get("has_modification"))
    n_reply = sum(1 for r in output_records if r.get("is_reply"))

    print(f"\n  Written: {len(output_records)} records")
    print(f"  Positive: {n_pos} ({n_pos/max(len(output_records),1)*100:.1f}%)")
    print(f"  Thread-aware (reply): {n_reply}")
    print(f"  Plain (top comment): {len(output_records) - n_reply}")
    print(f"  Spans aligned: {total_aligned}")
    print(f"  Spans hallucinated: {total_hallucinated}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()