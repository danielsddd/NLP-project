#!/usr/bin/env python3
"""
Data Preprocessing — v4
Converts teacher silver labels to token-level BIO tags for AlephBERT.

Usage:
    python -m src.preprocessing.prepare_data \
        --input data/silver_labels/teacher_output.jsonl \
        --output-dir data/processed

What this does:
    1. Reads teacher_output.jsonl (threads with Gemini labels)
    2. For each labeled modification:
       a. Picks the source comment text (top or reply)
       b. Finds the span in the text (exact -> stripped -> normalized fallback)
       c. Maps character positions to AlephBERT token positions
       d. Assigns BIO tags
    3. Splits into train/val/test (80/10/10)
    4. Runs an "eye test" showing 5 random aligned examples
"""

import json
import re
import random
import argparse
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoTokenizer

# =============================================================================
# LABEL SCHEMAS
# =============================================================================

BIO_LABEL2ID = {
    "O": 0,
    "B-SUBSTITUTION": 1, "I-SUBSTITUTION": 2,
    "B-QUANTITY": 3,      "I-QUANTITY": 4,
    "B-TECHNIQUE": 5,     "I-TECHNIQUE": 6,
    "B-ADDITION": 7,      "I-ADDITION": 8,
}
BIO_ID2LABEL = {v: k for k, v in BIO_LABEL2ID.items()}

IO_LABEL2ID = {
    "O": 0,
    "I-SUBSTITUTION": 1, "I-QUANTITY": 2,
    "I-TECHNIQUE": 3,     "I-ADDITION": 4,
}
IO_ID2LABEL = {v: k for k, v in IO_LABEL2ID.items()}

VALID_ASPECTS = {"SUBSTITUTION", "QUANTITY", "TECHNIQUE", "ADDITION"}

# =============================================================================
# HEBREW NORMALIZATION & SPAN FINDING
# =============================================================================

_FINALS = str.maketrans('\u05da\u05dd\u05df\u05e3\u05e5', '\u05db\u05de\u05e0\u05e4\u05e6')
_PUNCT_EDGE = '.,!?;:\'"()[]{}--...'


def _normalize_char(ch):
    """Normalize a single character: apply final-letter map, classify as content or punctuation."""
    mapped = ch.translate(_FINALS)
    if re.match(r'[\w\u0590-\u05FF]', mapped):
        return mapped, True   # word/Hebrew char -> keep
    elif mapped in ' \t\n\r':
        return ' ', True      # whitespace -> keep as space
    else:
        return '', False      # punctuation -> drop


def _build_normalized(text):
    """Normalize text and build a position map back to the original.

    Returns:
        normalized: The normalized string (finals mapped, punct stripped, ws collapsed)
        orig_positions: List where orig_positions[i] = original text index for normalized[i]

    This is used ONLY for span matching. AlephBERT always sees the original text.
    """
    chars = []
    positions = []
    for orig_i, ch in enumerate(text):
        mapped, is_content = _normalize_char(ch)
        if not is_content:
            continue
        if mapped == ' ':
            if chars and chars[-1] == ' ':
                continue  # collapse consecutive whitespace
            chars.append(' ')
            positions.append(orig_i)
        else:
            chars.append(mapped)
            positions.append(orig_i)

    # Strip leading/trailing whitespace
    joined = ''.join(chars)
    stripped = joined.strip()
    leading = len(joined) - len(joined.lstrip())
    positions = positions[leading:leading + len(stripped)]
    return stripped, positions


def find_span_in_text(text, span):
    """Find where a span starts and ends in the original text.

    Tries three strategies in order:
      1. Exact substring match (~85% of cases)
      2. Strip edge punctuation from span, retry (~10%)
      3. Full normalization with position mapping (~5%)

    Returns:
        (start_char, end_char) as a half-open interval, or (-1, -1) if not found.
    """
    if not text or not span:
        return -1, -1

    # Strategy 1: exact match
    idx = text.find(span)
    if idx != -1:
        return idx, idx + len(span)

    # Strategy 2: strip punctuation from span edges and retry
    stripped = span.strip(_PUNCT_EDGE)
    if stripped and stripped != span:
        idx = text.find(stripped)
        if idx != -1:
            return idx, idx + len(stripped)

    # Strategy 3: normalize both, find in normalized, map positions back
    norm_text, text_positions = _build_normalized(text)
    norm_span, _ = _build_normalized(span)

    if not norm_span:
        return -1, -1

    norm_idx = norm_text.find(norm_span)
    if norm_idx == -1:
        return -1, -1

    norm_end = norm_idx + len(norm_span) - 1
    if norm_idx >= len(text_positions) or norm_end >= len(text_positions):
        return -1, -1

    orig_start = text_positions[norm_idx]
    orig_end = text_positions[norm_end] + 1  # exclusive end
    return orig_start, orig_end

# =============================================================================
# SOURCE TEXT EXTRACTION
# =============================================================================

def get_source_text(record):
    """Extract the comment text the student model will train on.

    The teacher labels include source_comment ("top" or "reply_N") which tells
    us which comment in the thread actually contains the modification.
    We train the student on THAT comment only, not the full thread.

    Returns:
        (text, modifications_list)
    """
    teacher = record.get("teacher_output", {})
    mods = teacher.get("modifications", [])
    top_text = record.get("top_comment_text", "")
    replies = record.get("replies_texts", [])

    if not mods:
        return top_text, []

    source = mods[0].get("source_comment", "top")

    if source == "top":
        return top_text, mods

    # Parse "reply_N" -> index
    try:
        idx = int(source.split("_")[1]) - 1  # reply_1 -> index 0
        if 0 <= idx < len(replies):
            source_mods = [m for m in mods if m.get("source_comment") == source]
            return replies[idx], source_mods if source_mods else mods
    except (ValueError, IndexError):
        pass

    # Fallback: try top comment
    return top_text, mods

# =============================================================================
# BIO ALIGNMENT
# =============================================================================

def align_example(text, modifications, tokenizer, label2id, max_length=128):
    """Align character-level spans to token-level BIO tags.

    Args:
        text: Source comment text (Hebrew string)
        modifications: List of {"span": ..., "aspect": ...} from teacher
        tokenizer: AlephBERT tokenizer
        label2id: Tag string -> ID mapping
        max_length: Max token sequence length

    Returns:
        (example_dict, stats_dict) or (None, stats_dict) on failure.
    """
    if not text or not text.strip():
        return None, {"skip_reason": "empty_text"}

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoding["offset_mapping"][0].tolist()
    n = len(offsets)

    labels = ["O"] * n
    use_bio = any(k.startswith("B-") for k in label2id)
    stats = {"aligned": 0, "hallucinated": 0}

    for mod in modifications:
        span_text = mod.get("span", "")
        aspect = mod.get("aspect", "")

        if not span_text or aspect not in VALID_ASPECTS:
            stats["hallucinated"] += 1
            continue

        start_char, end_char = find_span_in_text(text, span_text)
        if start_char == -1:
            stats["hallucinated"] += 1
            continue

        first_token = True
        tagged_any = False
        for tok_idx in range(n):
            tok_start, tok_end = offsets[tok_idx]
            if tok_start == tok_end:
                continue  # special token
            if tok_end <= start_char or tok_start >= end_char:
                continue  # no overlap
            if labels[tok_idx] != "O":
                continue  # already tagged

            if use_bio:
                labels[tok_idx] = f"B-{aspect}" if first_token else f"I-{aspect}"
            else:
                labels[tok_idx] = f"I-{aspect}"
            first_token = False
            tagged_any = True

        if tagged_any:
            stats["aligned"] += 1
        else:
            stats["hallucinated"] += 1

    # Convert to IDs. Special tokens (offset 0,0) get -100.
    label_ids = []
    for idx in range(n):
        tok_start, tok_end = offsets[idx]
        if tok_start == 0 and tok_end == 0:
            label_ids.append(-100)
        else:
            label_ids.append(label2id.get(labels[idx], 0))

    return {
        "input_ids": encoding["input_ids"][0].tolist(),
        "attention_mask": encoding["attention_mask"][0].tolist(),
        "labels": label_ids,
        "text": text,
    }, stats

# =============================================================================
# FILE PROCESSING & SPLITTING
# =============================================================================

def process_file(input_path, output_dir, model_name="onlplab/alephbert-base",
                 max_length=128, scheme="BIO", seed=42,
                 train_ratio=0.8, val_ratio=0.1, include_empty=True):
    """Process silver labels JSONL into train/val/test splits with BIO tags."""
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label2id = BIO_LABEL2ID if scheme == "BIO" else IO_LABEL2ID
    id2label = BIO_ID2LABEL if scheme == "BIO" else IO_ID2LABEL
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Processing {input_path}")
    print(f"  Model: {model_name}, Scheme: {scheme} ({len(label2id)} labels), MaxLen: {max_length}")

    examples_with_mods = []
    examples_no_mods = []
    total_stats = Counter()

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                total_stats["json_errors"] += 1
                continue

            teacher = record.get("teacher_output", {})
            has_mod = teacher.get("has_modification", False)
            text, mods = get_source_text(record)

            result, stats = align_example(
                text, mods if has_mod else [],
                tokenizer, label2id, max_length
            )

            if result is None:
                total_stats["skipped"] += 1
                continue

            total_stats["total"] += 1
            total_stats["aligned"] += stats.get("aligned", 0)
            total_stats["hallucinated"] += stats.get("hallucinated", 0)

            has_tags = any(l > 0 for l in result["labels"] if l != -100)
            if has_tags:
                examples_with_mods.append(result)
                total_stats["with_mods"] += 1
            else:
                examples_no_mods.append(result)
                total_stats["no_mods"] += 1

            if line_num % 1000 == 0:
                print(f"  Processed {line_num} records...")

    if not include_empty:
        examples_no_mods = []

    all_examples = examples_with_mods + examples_no_mods
    random.shuffle(all_examples)

    n = len(all_examples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": all_examples[:n_train],
        "val": all_examples[n_train:n_train + n_val],
        "test": all_examples[n_train + n_val:],
    }

    for split_name, examples in splits.items():
        path = out / f"{split_name}.jsonl"
        with open(path, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        print(f"  {split_name}: {len(examples)} examples -> {path}")

    with open(out / "label2id.json", 'w') as f:
        json.dump(label2id, f, indent=2)
    with open(out / "id2label.json", 'w') as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)

    # Aspect token distribution
    aspect_counts = Counter()
    for ex in examples_with_mods:
        for l in ex["labels"]:
            if l > 0 and l != -100:
                tag = id2label.get(l, "O")
                if "-" in tag:
                    aspect_counts[tag.split("-", 1)[-1]] += 1

    # Hallucination rate
    total_spans = total_stats["aligned"] + total_stats["hallucinated"]
    halluc_pct = (total_stats["hallucinated"] / total_spans * 100) if total_spans > 0 else 0.0

    stats_dict = {
        "total_records": total_stats["total"],
        "with_modifications": total_stats["with_mods"],
        "no_modifications": total_stats["no_mods"],
        "aligned_spans": total_stats["aligned"],
        "hallucinated_spans": total_stats["hallucinated"],
        "hallucination_rate_pct": round(halluc_pct, 1),
        "skipped": total_stats["skipped"],
        "scheme": scheme, "num_labels": len(label2id), "max_length": max_length,
        "splits": {k: len(v) for k, v in splits.items()},
        "aspect_token_counts": dict(aspect_counts),
    }
    with open(out / "stats.json", 'w') as f:
        json.dump(stats_dict, f, indent=2)

    print(f"\n{'='*60}")
    print(f"PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total records:      {stats_dict['total_records']}")
    print(f"  With modifications: {stats_dict['with_modifications']}")
    print(f"  Aligned spans:      {stats_dict['aligned_spans']}")
    print(f"  Hallucinated spans: {stats_dict['hallucinated_spans']} ({halluc_pct:.1f}%)")
    if halluc_pct > 5:
        print(f"  WARNING: Hallucination rate > 5%! Review teacher prompt.")
    print(f"  Scheme:             {scheme} ({len(label2id)} labels)")
    print(f"  Splits:             {stats_dict['splits']}")
    print(f"  Aspect tokens:      {dict(aspect_counts)}")
    print(f"{'='*60}")

    # Eye test
    print(f"\n--- EYE TEST (verify alignment is correct) ---")
    n_samples = min(5, len(examples_with_mods))
    if n_samples > 0:
        for i, ex in enumerate(random.sample(examples_with_mods, n_samples), 1):
            print(f"\n  Example {i}:")
            visualize_alignment(ex, tokenizer, id2label)
    else:
        print("  No examples with modifications to show!")

    return stats_dict


def visualize_alignment(example, tokenizer, id2label):
    """Print token-label alignment for manual verification."""
    tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
    labels = example["labels"]
    text = example.get("text", "")
    if text:
        print(f"    Text: {text[:100]}{'...' if len(text) > 100 else ''}")

    tagged = []
    for tok, lid in zip(tokens, labels):
        if tok == "[PAD]":
            break
        if lid == -100:
            continue
        tag = id2label.get(lid, "O")
        if tag != "O":
            tagged.append(f"{tok} -> {tag}")

    print(f"    Tags: {', '.join(tagged) if tagged else '(none)'}")

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocess silver labels for training")
    parser.add_argument("--input", "-i", required=True, help="Input teacher_output.jsonl")
    parser.add_argument("--output-dir", "-o", default="data/processed")
    parser.add_argument("--model", default="onlplab/alephbert-base")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--scheme", choices=["BIO", "IO"], default="BIO")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--no-empty", action="store_true", help="Exclude no-modification examples")
    args = parser.parse_args()

    process_file(
        args.input, args.output_dir, args.model, args.max_length,
        args.scheme, args.seed, args.train_ratio, args.val_ratio,
        include_empty=not args.no_empty,
    )


if __name__ == "__main__":
    main()