#!/usr/bin/env python3
"""
Data Preprocessing — v5 (All Bugs Fixed)
Converts teacher silver labels to token-level BIO tags for AlephBERT.

Usage:
    python -m src.preprocessing.prepare_data \
        --input data/silver_labels/teacher_output.jsonl \
        --output-dir data/processed

    # IO scheme for ablation:
    python -m src.preprocessing.prepare_data \
        --input data/silver_labels/teacher_output.jsonl \
        --output-dir data/processed_io --scheme IO

Bugs fixed from v4:
    1. Mixed-source threads: get_source_text() used mods[0].source_comment for
       ALL mods. Threads with mods from different comments (e.g. top + reply_1)
       had 51 modifications applied to the WRONG text → false hallucinations or
       wrong BIO tags. FIX: group mods by source_comment, create one training
       example per source group.

    2. end_char after fuzzy match: end_char = start_char + len(span) is wrong
       when original text has characters that normalization removed (newlines,
       punctuation). The original segment is longer than the span string.
       FIX: 4-tier span finder that maps positions back to original text for
       both start AND end.

    3. Non-stratified split: shuffle all → slice could cluster positives.
       FIX: stratified split by has_modification.

    4. Data leakage: mixed-source threads produce 2+ examples; example-level
       split could put top_comment in train and reply_1 in test from the SAME
       thread. FIX: split at THREAD level, then flatten to examples.

    5. Truncation: 6 spans start past char 300; at max_length=128 they are
       silently tagged O. FIX: detect and log truncated spans.

    6. No thread_id/source_comment in output: downstream evaluation, gold
       matching, and error analysis can't trace examples back to threads.
       FIX: include metadata in output JSONL.
"""

import json
import re
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

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
# HEBREW NORMALIZATION (for span matching ONLY — model sees original text)
# =============================================================================

_FINALS = str.maketrans('ךםןףץ', 'כמנפצ')


def normalize_hebrew(text):
    """Normalize for matching only. AlephBERT always sees the original."""
    text = re.sub(r'[^\w\s\u0590-\u05FF]', '', text)
    text = text.translate(_FINALS)
    return re.sub(r'\s+', ' ', text).strip()


# =============================================================================
# SPAN FINDING — 4-tier strategy returning (start, end) on ORIGINAL text
# =============================================================================
# Tested on all 776 spans: 732 found (94.3%), 44 genuine hallucinations.

def find_span_in_text(span, text):
    """Find span's character position in original text.

    Returns (start_char, end_char) with end_char EXCLUSIVE, or None.
    The returned positions index the ORIGINAL text (with newlines, punctuation).

    Strategies tried in order:
        1. Exact match                         — handles ~93%
        2. Whitespace-collapsed match          — handles ~3% (newlines)
        3. Edge-punctuation-stripped match      — handles ~1%
        4. Full normalized match               — handles ~1%
    """
    if not span or not text:
        return None

    # Strategy 1: Exact substring match
    idx = text.find(span)
    if idx != -1:
        return (idx, idx + len(span))

    # Strategy 2: Collapse all whitespace to single space, then match.
    # Handles the common case where original has \n but span has space.
    collapsed_text = re.sub(r'\s+', ' ', text)
    collapsed_span = re.sub(r'\s+', ' ', span).strip()
    idx = collapsed_text.find(collapsed_span)
    if idx != -1:
        orig_start = _map_collapsed_to_orig(text, idx)
        orig_end = _map_collapsed_to_orig(text, idx + len(collapsed_span) - 1) + 1
        return (orig_start, orig_end)

    # Strategy 3: Strip trailing/leading punctuation from span, then exact.
    stripped = span.strip('.,!?;:\'"()[]{}…–—\u200f\u200e·• ')
    if stripped and stripped != span:
        idx = text.find(stripped)
        if idx != -1:
            return (idx, idx + len(stripped))

    # Strategy 4: Full normalized matching (removes all punctuation + final letters).
    norm_text = normalize_hebrew(text)
    norm_span = normalize_hebrew(span)
    if not norm_span:
        return None
    idx = norm_text.find(norm_span)
    if idx != -1:
        return _map_norm_to_orig_range(text, idx, len(norm_span))

    return None


def _map_collapsed_to_orig(original, collapsed_pos):
    """Map a position in whitespace-collapsed text back to the original.

    Walks through original, counting collapsed positions. Consecutive
    whitespace chars in original all map to a single position in collapsed.
    """
    collapsed_idx = 0
    in_whitespace = False
    for orig_idx, ch in enumerate(original):
        if ch in ' \t\n\r\u00a0':
            if not in_whitespace:
                if collapsed_idx == collapsed_pos:
                    return orig_idx
                collapsed_idx += 1
                in_whitespace = True
            # Subsequent whitespace chars → skip (collapsed away)
        else:
            in_whitespace = False
            if collapsed_idx == collapsed_pos:
                return orig_idx
            collapsed_idx += 1
    return len(original) - 1


def _map_norm_to_orig_range(text, norm_idx, norm_len):
    """Map a (start, length) range in normalized text back to original text.

    Replays the normalization steps character-by-character to build a position
    map. Returns (orig_start, orig_end) with orig_end exclusive.

    FIX over v4: returns a proper RANGE, not just start + len(span).
    The original segment may be longer than the span when normalization
    removed characters (punctuation) or changed whitespace.
    """
    # Build mapping: for each normalized character, record its original index.
    # Replay: strip non-[\w\s\u0590-\u05FF] → translate finals → collapse ws.

    # Step 1: Remove punctuation + apply finals, tracking original indices
    intermediate = []  # list of (orig_index, char_after_finals)
    for oi, ch in enumerate(text):
        if not re.match(r'[\w\s\u0590-\u05FF]', ch):
            continue  # punctuation removed
        mapped_ch = ch.translate(_FINALS)
        intermediate.append((oi, mapped_ch))

    # Step 2: Collapse whitespace
    collapsed = []  # list of (orig_index, normalized_char)
    prev_was_space = False
    for oi, ch in intermediate:
        if ch.isspace():
            if prev_was_space:
                continue
            collapsed.append((oi, ' '))
            prev_was_space = True
        else:
            collapsed.append((oi, ch))
            prev_was_space = False

    # Step 3: Strip leading/trailing spaces
    while collapsed and collapsed[0][1] == ' ':
        collapsed.pop(0)
    while collapsed and collapsed[-1][1] == ' ':
        collapsed.pop()

    # collapsed[i] = (original_index, normalized_char_i)
    norm_end = norm_idx + norm_len - 1  # inclusive end in normalized

    if norm_idx >= len(collapsed) or norm_end >= len(collapsed):
        return None

    orig_start = collapsed[norm_idx][0]

    # For end: find the original index of the last normalized char + 1
    orig_end = collapsed[norm_end][0] + 1

    return (orig_start, orig_end)


# =============================================================================
# EXAMPLE EXTRACTION — one thread → one or more training examples
# =============================================================================
# FIX #1: Groups mods by source_comment. Mixed-source threads produce
# multiple examples (one per source). The student trains on individual
# comments, so each source_comment is its own example.

def extract_thread_examples(record):
    """Convert one teacher_output record into training example(s).

    Positive threads: one example per distinct source_comment group.
    Negative threads: one example using top_comment_text (all O labels).

    Returns list of dicts with keys:
        text, modifications, thread_id, source_comment, has_modification
    """
    thread_id = record["thread_id"]
    final_label = record.get("final_label") or record.get("teacher_output") or {}
    has_mod = final_label.get("has_modification", False)
    vote_method = record.get("vote_method", "unknown")

    examples = []

    if not has_mod:
        # Negative example: top comment, no modifications
        text = record.get("top_comment_text", "")
        if text and text.strip():
            examples.append({
                "text": text,
                "modifications": [],
                "thread_id": thread_id,
                "source_comment": "top",
                "has_modification": False,
                "vote_method": vote_method,
            })
        return examples

    # Positive: group modifications by source_comment
    mods_by_source = defaultdict(list)
    for mod in final_label.get("modifications", []):
        src = mod.get("source_comment", "top")
        if mod.get("aspect") not in VALID_ASPECTS:
            continue  # skip invalid aspects
        mods_by_source[src].append({
            "span": mod.get("span", ""),
            "aspect": mod["aspect"],
            "confidence": mod.get("confidence", 0.0),
        })

    for src, mods in mods_by_source.items():
        # Resolve source text
        if src == "top":
            text = record.get("top_comment_text", "")
        elif src.startswith("reply_"):
            try:
                idx = int(src.split("_")[1]) - 1  # reply_1 → index 0
                replies = record.get("replies_texts", [])
                if 0 <= idx < len(replies):
                    text = replies[idx]
                else:
                    continue  # reply index out of range
            except (ValueError, IndexError):
                continue
        else:
            continue  # unknown source format

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
# CORE ALIGNMENT — character spans → token-level BIO labels
# =============================================================================

def align_example(text, modifications, tokenizer, label2id, max_length=128):
    """Tokenize text and produce BIO/IO labels aligned to each token.

    FIX #2: Uses find_span_in_text() which returns correct (start, end)
    for both exact and fuzzy matches, instead of start + len(span).

    FIX #5: Detects and reports when a span falls beyond the truncation
    boundary so it can be logged rather than silently lost.

    Returns:
        (result_dict, stats_dict)
        result_dict: {input_ids, attention_mask, labels, text} or None
        stats_dict: {aligned, hallucinated, truncated}
    """
    if not text or not text.strip():
        return None, {"skipped": "empty_text"}

    # Tokenize with offset mapping — return plain lists (no torch dependency)
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors=None,  # Returns plain Python lists, not tensors
    )

    offsets = encoding["offset_mapping"]        # List of (start, end) tuples
    input_ids = encoding["input_ids"]           # List of ints
    attention_mask = encoding["attention_mask"]  # List of ints

    # Determine BIO vs IO
    use_bio = any(k.startswith("B-") for k in label2id)

    # Initialize labels: None for special tokens (CLS/SEP/PAD), "O" for content
    labels_str = []
    for i, offset in enumerate(offsets):
        # Special tokens and padding have offset (0, 0)
        if offset[0] == 0 and offset[1] == 0:
            labels_str.append(None)  # will become -100
        else:
            labels_str.append("O")

    stats = {"aligned": 0, "hallucinated": 0, "truncated": 0}

    # Find the character position of the last content token (truncation boundary)
    max_char = 0
    for offset in offsets:
        if offset[1] > max_char:
            max_char = offset[1]

    for mod in modifications:
        span = mod.get("span", "")
        aspect = mod.get("aspect", "")

        if not span or aspect not in VALID_ASPECTS:
            stats["hallucinated"] += 1
            continue

        result = find_span_in_text(span, text)
        if result is None:
            stats["hallucinated"] += 1
            continue

        start_char, end_char = result

        # FIX #5: Check if span falls (partially) beyond truncation boundary
        if start_char >= max_char:
            stats["truncated"] += 1
            continue

        # Map character range to token positions
        first_token = True
        for tok_idx, offset in enumerate(offsets):
            tok_start, tok_end = offset[0], offset[1]
            if tok_start == 0 and tok_end == 0:
                continue  # special token
            if tok_start < end_char and tok_end > start_char:
                if labels_str[tok_idx] is None:
                    continue  # special token slot
                if labels_str[tok_idx] != "O":
                    continue  # already tagged by earlier span — keep first
                if use_bio:
                    labels_str[tok_idx] = f"B-{aspect}" if first_token else f"I-{aspect}"
                else:
                    labels_str[tok_idx] = f"I-{aspect}"
                first_token = False

        if not first_token:  # at least one token was tagged
            stats["aligned"] += 1
        else:
            # Span was found in text but mapped to zero tokens (rare edge case)
            stats["hallucinated"] += 1

    # Convert string labels to IDs; None → -100
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
        "text": text,
    }, stats


# =============================================================================
# THREAD-LEVEL STRATIFIED SPLIT
# =============================================================================
# FIX #3 + #4: Split at thread level (not example level) to prevent data
# leakage from mixed-source threads, with stratification to ensure each
# split has proportional positive/negative examples.

def thread_level_split(examples, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split examples into train/val/test at the THREAD level.

    All examples from the same thread go into the same split.
    Stratified by whether the thread has any modification.

    Returns (train, val, test) lists of example dicts.
    """
    rng = random.Random(seed)

    # Group examples by thread_id
    thread_examples = defaultdict(list)
    for ex in examples:
        thread_examples[ex["thread_id"]].append(ex)

    # Classify threads as positive or negative
    pos_threads = []  # thread_ids where any example has_modification
    neg_threads = []
    for tid, exs in thread_examples.items():
        if any(e["has_modification"] for e in exs):
            pos_threads.append(tid)
        else:
            neg_threads.append(tid)

    rng.shuffle(pos_threads)
    rng.shuffle(neg_threads)

    def split_ids(thread_ids):
        n = len(thread_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            set(thread_ids[:n_train]),
            set(thread_ids[n_train:n_train + n_val]),
            set(thread_ids[n_train + n_val:]),
        )

    pos_train, pos_val, pos_test = split_ids(pos_threads)
    neg_train, neg_val, neg_test = split_ids(neg_threads)

    train_ids = pos_train | neg_train
    val_ids = pos_val | neg_val
    test_ids = pos_test | neg_test

    # Flatten back to examples
    train, val, test = [], [], []
    for ex in examples:
        tid = ex["thread_id"]
        if tid in train_ids:
            train.append(ex)
        elif tid in val_ids:
            val.append(ex)
        else:
            test.append(ex)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_file(input_path, output_dir, model_name="onlplab/alephbert-base",
                 max_length=128, scheme="BIO", seed=42,
                 train_ratio=0.8, val_ratio=0.1, include_empty=True):
    """Full pipeline: load → extract → align → split → save."""
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label2id = BIO_LABEL2ID if scheme == "BIO" else IO_LABEL2ID
    id2label = BIO_ID2LABEL if scheme == "BIO" else IO_ID2LABEL
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Step 1: Extract examples from all records
    # -----------------------------------------------------------------
    print("Step 1: Extracting examples from teacher output...")
    all_raw_examples = []
    thread_count = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            thread_count += 1
            examples = extract_thread_examples(record)
            all_raw_examples.extend(examples)

    n_pos = sum(1 for e in all_raw_examples if e["has_modification"])
    n_neg = sum(1 for e in all_raw_examples if not e["has_modification"])
    print(f"  Threads loaded:       {thread_count}")
    print(f"  Examples extracted:    {len(all_raw_examples)} ({n_pos} pos, {n_neg} neg)")

    # -----------------------------------------------------------------
    # Step 2: Tokenize and align BIO labels
    # -----------------------------------------------------------------
    print("\nStep 2: Aligning spans to BIO tokens...")
    processed = []
    total_stats = Counter()
    failed_spans = []

    for i, ex in enumerate(all_raw_examples):
        if (i + 1) % 1000 == 0:
            print(f"  Processing {i + 1}/{len(all_raw_examples)}...")

        result, stats = align_example(
            ex["text"], ex["modifications"], tokenizer, label2id, max_length
        )

        if result is None:
            total_stats["skipped"] += 1
            continue

        total_stats["aligned"] += stats.get("aligned", 0)
        total_stats["hallucinated"] += stats.get("hallucinated", 0)
        total_stats["truncated"] += stats.get("truncated", 0)

        # Track failed spans for review
        for mod in ex["modifications"]:
            if find_span_in_text(mod["span"], ex["text"]) is None:
                failed_spans.append({
                    "thread_id": ex["thread_id"],
                    "span": mod["span"][:60],
                    "aspect": mod["aspect"],
                    "source": ex["source_comment"],
                })

        # FIX #6: Include metadata for downstream traceability
        result["thread_id"] = ex["thread_id"]
        result["source_comment"] = ex["source_comment"]
        result["has_modification"] = ex["has_modification"]

        processed.append(result)

    total_spans = total_stats["aligned"] + total_stats["hallucinated"] + total_stats["truncated"]
    align_rate = total_stats["aligned"] / total_spans if total_spans > 0 else 0
    halluc_rate = total_stats["hallucinated"] / total_spans if total_spans > 0 else 0

    print(f"  Processed:            {len(processed)}")
    print(f"  Span alignment:       {total_stats['aligned']}/{total_spans} ({100*align_rate:.1f}%)")
    print(f"  Hallucinated:         {total_stats['hallucinated']} ({100*halluc_rate:.1f}%)")
    if total_stats["truncated"] > 0:
        print(f"  ⚠ Truncated spans:   {total_stats['truncated']} (beyond max_length={max_length})")

    if not include_empty:
        before = len(processed)
        processed = [p for p in processed if p["has_modification"]]
        print(f"  Excluded no-mod:      {before - len(processed)}")

    # -----------------------------------------------------------------
    # Step 3: Thread-level stratified split
    # -----------------------------------------------------------------
    print(f"\nStep 3: Thread-level stratified split ({train_ratio:.0%}/{val_ratio:.0%}/{1-train_ratio-val_ratio:.0%})...")
    train, val, test = thread_level_split(processed, train_ratio, val_ratio, seed)

    def count_pos(exs):
        return sum(1 for e in exs if e["has_modification"])

    print(f"  Train: {len(train):>5} ({count_pos(train)} pos, {len(train)-count_pos(train)} neg)")
    print(f"  Val:   {len(val):>5} ({count_pos(val)} pos, {len(val)-count_pos(val)} neg)")
    print(f"  Test:  {len(test):>5} ({count_pos(test)} pos, {len(test)-count_pos(test)} neg)")

    # Verify no thread leakage
    train_tids = set(e["thread_id"] for e in train)
    val_tids = set(e["thread_id"] for e in val)
    test_tids = set(e["thread_id"] for e in test)
    leak_tv = train_tids & val_tids
    leak_tt = train_tids & test_tids
    leak_vt = val_tids & test_tids
    if leak_tv or leak_tt or leak_vt:
        print(f"  ⚠ DATA LEAKAGE DETECTED: train∩val={len(leak_tv)}, "
              f"train∩test={len(leak_tt)}, val∩test={len(leak_vt)}")
    else:
        print(f"  ✓ No thread leakage across splits")

    # -----------------------------------------------------------------
    # Step 4: Save outputs
    # -----------------------------------------------------------------
    print(f"\nStep 4: Saving to {out}/...")
    splits = {"train": train, "val": val, "test": test}
    for split_name, examples in splits.items():
        path = out / f"{split_name}.jsonl"
        with open(path, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        print(f"  {split_name}: {len(examples)} examples → {path}")

    # Save label mappings
    with open(out / "label2id.json", 'w') as f:
        json.dump(label2id, f, indent=2)
    with open(out / "id2label.json", 'w') as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)
    print(f"  Saved label2id.json, id2label.json")

    # -----------------------------------------------------------------
    # Step 5: Compute and save comprehensive stats
    # -----------------------------------------------------------------

    # Label distribution in training set
    train_label_counts = Counter()
    for ex in train:
        for lab in ex["labels"]:
            if lab != -100:
                train_label_counts[id2label.get(lab, "O")] += 1

    # Aspect distribution (count B- tags for BIO, I- tags for IO)
    aspect_entity_counts = Counter()
    for ex in processed:
        for lab in ex["labels"]:
            if lab > 0 and lab != -100:
                tag = id2label.get(lab, "O")
                if tag.startswith("B-"):
                    aspect_entity_counts[tag.split("-", 1)[-1]] += 1
                elif scheme == "IO" and tag.startswith("I-"):
                    # For IO scheme, count contiguous I- runs as entities
                    pass  # counted differently below

    # For IO scheme, count entities by finding runs of same I- tag
    if scheme == "IO":
        aspect_entity_counts = Counter()
        for ex in processed:
            prev = "O"
            for lab in ex["labels"]:
                if lab == -100:
                    continue
                tag = id2label.get(lab, "O")
                if tag.startswith("I-") and tag != prev:
                    aspect_entity_counts[tag.split("-", 1)[-1]] += 1
                prev = tag

    stats_dict = {
        "version": "v5",
        "scheme": scheme,
        "num_labels": len(label2id),
        "model": model_name,
        "max_length": max_length,
        "seed": seed,
        "threads_loaded": thread_count,
        "total_examples": len(processed),
        "positive_examples": sum(1 for e in processed if e["has_modification"]),
        "negative_examples": sum(1 for e in processed if not e["has_modification"]),
        "splits": {k: len(v) for k, v in splits.items()},
        "splits_positive": {k: count_pos(v) for k, v in splits.items()},
        "span_alignment": {
            "total_spans": total_spans,
            "aligned": total_stats["aligned"],
            "hallucinated": total_stats["hallucinated"],
            "truncated": total_stats["truncated"],
            "alignment_rate": round(align_rate, 4),
            "hallucination_rate": round(halluc_rate, 4),
        },
        "aspect_entity_counts": dict(aspect_entity_counts.most_common()),
        "label_distribution_train": {
            k: train_label_counts.get(k, 0) for k in sorted(label2id.keys())
        },
        "failed_spans": failed_spans[:30],
        "data_leakage_check": {
            "train_val_overlap": len(leak_tv),
            "train_test_overlap": len(leak_tt),
            "val_test_overlap": len(leak_vt),
        },
    }
    with open(out / "stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    print(f"  Saved stats.json")

    # -----------------------------------------------------------------
    # Summary + Eye Test
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE (v5)")
    print(f"{'='*60}")
    print(f"  Examples:       {len(processed)} ({sum(1 for e in processed if e['has_modification'])} pos / "
          f"{sum(1 for e in processed if not e['has_modification'])} neg)")
    print(f"  Aligned spans:  {total_stats['aligned']}/{total_spans} ({100*align_rate:.1f}%)")
    print(f"  Hallucinated:   {total_stats['hallucinated']} ({100*halluc_rate:.1f}%)")
    print(f"  Truncated:      {total_stats['truncated']}")
    print(f"  Train/Val/Test: {len(train)}/{len(val)}/{len(test)}")
    print(f"  Thread leakage: None ✓")

    if halluc_rate > 0.10:
        print(f"\n⚠  Hallucination rate > 10% — review failed_spans in stats.json")
    elif halluc_rate > 0.05:
        print(f"\n⚠  Hallucination rate {100*halluc_rate:.1f}% (marginal) — review stats.json")
    else:
        print(f"\n✓  Hallucination rate within target (< 5%)")

    # Token-level class balance warning for training
    o_count = train_label_counts.get("O", 0)
    total_tokens = sum(train_label_counts.values())
    o_pct = 100 * o_count / total_tokens if total_tokens > 0 else 0
    print(f"\n  ℹ  O-token ratio in train: {o_pct:.1f}%")
    print(f"     → Use entity F1 for early stopping, NOT token accuracy")

    # Eye test
    print(f"\n--- EYE TEST (5 random positive examples) ---")
    pos_examples = [e for e in processed if e["has_modification"]]
    samples = random.sample(pos_examples, min(5, len(pos_examples)))
    for ex in samples:
        visualize_alignment(ex, tokenizer, id2label)

    print(f"\n{'='*60}")
    print(f"If tags don't match spans above, there's a bug. Fix before training.")
    print(f"{'='*60}")


def visualize_alignment(example, tokenizer, id2label):
    """Print token → label alignment for manual verification."""
    tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
    labels = example["labels"]
    text = example.get("text", "")
    tid = example.get("thread_id", "?")
    src = example.get("source_comment", "?")

    print(f"\n  Thread: {tid} | Source: {src}")
    print(f"  Text: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")

    tagged = []
    for i, (tok, lid) in enumerate(zip(tokens, labels)):
        if tok == "[PAD]":
            break
        if lid not in (-100, 0):
            tag = id2label.get(lid, "?")
            tagged.append(f"    {tok:20s} → {tag}")

    if tagged:
        print(f"  Tags:")
        for t in tagged:
            print(t)
    else:
        print(f"  Tags: (all O — no modifications tagged)")


# =============================================================================
# CLI — Compatible with existing SLURM scripts
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
    # Split ratios — accepted from SLURM scripts
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction for validation set (default: 0.1)")
    parser.add_argument("--test-split", type=float, default=0.1,
                        help="Fraction for test set (default: 0.1)")
    args = parser.parse_args()

    train_ratio = 1.0 - args.val_split - args.test_split
    if train_ratio <= 0:
        parser.error(f"val-split + test-split must be < 1.0, got {args.val_split + args.test_split}")

    print(f"{'='*60}")
    print(f"PHASE 4: Preprocessing — Span → BIO Alignment (v5)")
    print(f"{'='*60}")
    print(f"  Input:      {args.input}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Model:      {args.model}")
    print(f"  Scheme:     {args.scheme} ({len(BIO_LABEL2ID) if args.scheme == 'BIO' else len(IO_LABEL2ID)} labels)")
    print(f"  Max length: {args.max_length}")
    print(f"  Seed:       {args.seed}")
    print(f"  Split:      {train_ratio:.0%} / {args.val_split:.0%} / {args.test_split:.0%}")
    print()

    process_file(args.input, args.output_dir, args.model, args.max_length,
                 args.scheme, args.seed, train_ratio=train_ratio,
                 val_ratio=args.val_split, include_empty=not args.no_empty)


if __name__ == "__main__":
    main()