#!/usr/bin/env python3
"""
Data Preprocessing — v4
Converts teacher silver labels to token-level BIO tags for AlephBERT.

Usage:
    python -m src.preprocessing.prepare_data \
        --input data/silver_labels/teacher_output.jsonl \
        --output-dir data/processed
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
    "I-TECHNIQUE": 3, "I-ADDITION": 4,
}
IO_ID2LABEL = {v: k for k, v in IO_LABEL2ID.items()}

VALID_ASPECTS = {"SUBSTITUTION", "QUANTITY", "TECHNIQUE", "ADDITION"}

# =============================================================================
# HEBREW NORMALIZATION
# =============================================================================

_FINALS = str.maketrans('ךםןףץ', 'כמנפצ')

def normalize_hebrew(text):
    """Normalize Hebrew text for span matching only.
    Strips punctuation, normalizes final letters, collapses whitespace.
    Used ONLY for matching — AlephBERT sees the original text."""
    text = re.sub(r'[^\w\s\u0590-\u05FF]', '', text)
    text = text.translate(_FINALS)
    return re.sub(r'\s+', ' ', text).strip()

# =============================================================================
# CORE ALIGNMENT
# =============================================================================

def get_source_text(record):
    """Extract the text that the student model will see.
    Uses source_comment field to pick top_comment or the right reply."""
    teacher = record.get("teacher_output", {})
    mods = teacher.get("modifications", [])

    if not mods:
        # No modifications — use top comment text
        return record.get("top_comment_text", ""), mods

    # Find which comment to use based on first modification's source
    source = mods[0].get("source_comment", "top")
    if source == "top":
        return record.get("top_comment_text", ""), mods

    # source_comment is "reply_N" — extract reply index
    replies = record.get("replies_texts", [])
    try:
        idx = int(source.split("_")[1]) - 1  # reply_1 → index 0
        if 0 <= idx < len(replies):
            return replies[idx], mods
    except (ValueError, IndexError):
        pass

    # Fallback: try top comment
    return record.get("top_comment_text", ""), mods

def align_example(text, modifications, tokenizer, label2id, max_length=128):
    """Align character-level spans to token-level BIO tags.

    Returns dict with input_ids, attention_mask, labels, or None on failure.
    Also returns stats dict for tracking.
    """
    if not text or not text.strip():
        return None, {"skipped": "empty_text"}

    # Tokenize with offset mapping
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offsets = encoding["offset_mapping"][0].tolist()  # [(start, end), ...]
    labels = ["O"] * max_length
    stats = {"aligned": 0, "hallucinated": 0}

    # Determine if we use BIO or IO
    use_bio = any(k.startswith("B-") for k in label2id)

    # Normalized text for fuzzy matching
    norm_text = normalize_hebrew(text)

    for mod in modifications:
        span = mod.get("span", "")
        aspect = mod.get("aspect", "")

        if not span or aspect not in VALID_ASPECTS:
            stats["hallucinated"] += 1
            continue

        # Try exact match first, then normalized match
        start_char = text.find(span)
        if start_char == -1:
            norm_span = normalize_hebrew(span)
            norm_idx = norm_text.find(norm_span)
            if norm_idx == -1:
                stats["hallucinated"] += 1
                continue
            # Map normalized position back to original text
            # Walk through original text to find corresponding position
            start_char = _map_norm_to_orig(text, norm_text, norm_idx, len(norm_span))
            if start_char == -1:
                stats["hallucinated"] += 1
                continue

        end_char = start_char + len(span)

        # Map character range to token positions
        first_token = True
        for tok_idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end:
                continue  # special token
            if tok_start < end_char and tok_end > start_char:
                if labels[tok_idx] != "O":
                    continue  # already tagged by earlier span
                if use_bio:
                    labels[tok_idx] = f"B-{aspect}" if first_token else f"I-{aspect}"
                else:
                    labels[tok_idx] = f"I-{aspect}"
                first_token = False

        if not first_token:  # at least one token was tagged
            stats["aligned"] += 1
        else:
            stats["hallucinated"] += 1

    # Convert labels to IDs, -100 for special tokens
    label_ids = []
    for idx, label in enumerate(labels):
        if offsets[idx] == [0, 0]:  # special token ([CLS], [SEP], [PAD])
            label_ids.append(-100)
        else:
            label_ids.append(label2id.get(label, 0))

    return {
        "input_ids": encoding["input_ids"][0].tolist(),
        "attention_mask": encoding["attention_mask"][0].tolist(),
        "labels": label_ids,
        "text": text,
    }, stats

def _map_norm_to_orig(original, normalized, norm_idx, norm_len):
    """Map a position in normalized text back to the original text.
    Simple approach: walk both strings keeping a position mapping."""
    norm_pos, orig_pos = 0, 0
    start_orig = -1

    norm_orig = normalize_hebrew(original[:1])  # just to use same logic

    # Build position mapping
    o2n = []  # original index → normalized index
    ni = 0
    for oi, ch in enumerate(original):
        norm_ch = normalize_hebrew(ch)
        if norm_ch and ni < len(normalized):
            o2n.append(ni)
            if norm_ch == normalized[ni]:
                ni += 1
            # else: character was removed in normalization
        else:
            o2n.append(ni)

    # Find original position where normalized index == norm_idx
    for oi, ni in enumerate(o2n):
        if ni == norm_idx and start_orig == -1:
            return oi

    return -1

# =============================================================================
# PROCESS FILE
# =============================================================================

def process_file(input_path, output_dir, model_name="onlplab/alephbert-base",
                 max_length=128, scheme="BIO", seed=42,
                 train_ratio=0.8, val_ratio=0.1, include_empty=True):
    """Process silver labels file into train/val/test splits."""
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label2id = BIO_LABEL2ID if scheme == "BIO" else IO_LABEL2ID
    id2label = BIO_ID2LABEL if scheme == "BIO" else IO_ID2LABEL
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Process all records
    examples_with_mods, examples_no_mods = [], []
    total_stats = Counter()

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            teacher = record.get("teacher_output", {})
            has_mod = teacher.get("has_modification", False)
            text, mods = get_source_text(record)

            result, stats = align_example(text, mods if has_mod else [], tokenizer, label2id, max_length)
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

    if not include_empty:
        examples_no_mods = []

    # Split
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

    # Save
    for split_name, examples in splits.items():
        path = out / f"{split_name}.jsonl"
        with open(path, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        print(f"  {split_name}: {len(examples)} examples → {path}")

    # Save metadata
    with open(out / "label2id.json", 'w') as f:
        json.dump(label2id, f, indent=2)
    with open(out / "id2label.json", 'w') as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)

    # Aspect distribution
    aspect_counts = Counter()
    for ex in examples_with_mods:
        for l in ex["labels"]:
            if l > 0 and l != -100:
                tag = id2label.get(l, "O")
                aspect = tag.split("-", 1)[-1] if "-" in tag else tag
                aspect_counts[aspect] += 1

    stats_dict = {
        "total": total_stats["total"],
        "with_modifications": total_stats["with_mods"],
        "no_modifications": total_stats["no_mods"],
        "aligned_spans": total_stats["aligned"],
        "hallucinated_spans": total_stats["hallucinated"],
        "skipped": total_stats["skipped"],
        "scheme": scheme,
        "num_labels": len(label2id),
        "splits": {k: len(v) for k, v in splits.items()},
        "aspect_counts": dict(aspect_counts),
    }
    with open(out / "stats.json", 'w') as f:
        json.dump(stats_dict, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"PREPROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"  Total:           {stats_dict['total']}")
    print(f"  With mods:       {stats_dict['with_modifications']}")
    print(f"  Aligned spans:   {stats_dict['aligned_spans']}")
    print(f"  Hallucinated:    {stats_dict['hallucinated_spans']}")
    print(f"  Scheme:          {scheme} ({len(label2id)} labels)")
    print(f"  Aspects:         {dict(aspect_counts)}")
    print(f"{'='*50}")

    # Eye test: show 5 random examples
    print(f"\n--- EYE TEST (5 random examples) ---")
    samples = random.sample(examples_with_mods, min(5, len(examples_with_mods)))
    for ex in samples:
        visualize_alignment(ex, tokenizer, id2label)

def visualize_alignment(example, tokenizer, id2label):
    """Print token-label alignment for manual verification."""
    tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
    labels = example["labels"]
    print(f"\nText: {example.get('text', '')[:80]}...")
    for i, (tok, lid) in enumerate(zip(tokens, labels)):
        if tok in ("[PAD]",):
            break
        tag = id2label.get(lid, "SKIP") if lid != -100 else "---"
        if tag not in ("O", "---", "SKIP"):
            print(f"  [{i:3d}] {tok:20s} → {tag}")

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
    parser.add_argument("--no-empty", action="store_true", help="Exclude no-mod examples")
    args = parser.parse_args()

    process_file(args.input, args.output_dir, args.model, args.max_length,
                 args.scheme, args.seed, include_empty=not args.no_empty)

if __name__ == "__main__":
    main()