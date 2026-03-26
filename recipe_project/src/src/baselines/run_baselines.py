#!/usr/bin/env python3
"""
Baselines — v4
Implements Majority, Random, and Keyword baselines.
For mBERT baseline, use train_student.py with --model bert-base-multilingual-cased.

Usage:
    python -m src.baselines.run_baselines --data-dir data/processed
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter

from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

# =============================================================================
# LABEL SCHEMA
# =============================================================================

BIO_LABEL2ID = {
    "O": 0, "B-SUBSTITUTION": 1, "I-SUBSTITUTION": 2,
    "B-QUANTITY": 3, "I-QUANTITY": 4,
    "B-TECHNIQUE": 5, "I-TECHNIQUE": 6,
    "B-ADDITION": 7, "I-ADDITION": 8,
}
BIO_ID2LABEL = {v: k for k, v in BIO_LABEL2ID.items()}

# =============================================================================
# HEBREW KEYWORDS
# =============================================================================

ASPECT_KEYWORDS = {
    "SUBSTITUTION": ["במקום", "החלפתי", "השתמשתי ב", "ולא", "במקום ה", "החלפתי את"],
    "QUANTITY": ["יותר", "פחות", "כפול", "חצי", "הוספתי", "הפחתתי", "הכפלתי", "פי שניים"],
    "TECHNIQUE": ["דקות", "שעות", "מעלות", "תנור", "כיריים", "לערבב", "לקפל", "על אש"],
    "ADDITION": ["הוספתי גם", "שמתי גם", "אפשר להוסיף", "הוספתי עוד", "שמתי עוד"],
}

# =============================================================================
# BASELINES
# =============================================================================

def majority_predict(examples, id2label):
    """Predict O for every token."""
    all_preds = []
    for ex in examples:
        preds = []
        for l in ex["labels"]:
            if l == -100:
                continue
            preds.append("O")
        all_preds.append(preds)
    return all_preds

def random_predict(examples, id2label, label_dist, seed=42):
    """Predict random labels based on training distribution."""
    rng = random.Random(seed)
    labels_list = list(label_dist.keys())
    weights = list(label_dist.values())

    all_preds = []
    for ex in examples:
        preds = []
        for l in ex["labels"]:
            if l == -100:
                continue
            chosen = rng.choices(labels_list, weights=weights, k=1)[0]
            preds.append(chosen)
        all_preds.append(preds)
    return all_preds

def keyword_predict(examples, id2label, tokenizer):
    """Predict using Hebrew keyword matching."""
    all_preds = []
    for ex in examples:
        text = ex.get("text", "")
        tokens = tokenizer.convert_ids_to_tokens(ex["input_ids"])
        offsets = _get_offsets(text, tokenizer, len(ex["input_ids"]))
        labels_out = ["O"] * len(ex["input_ids"])

        for aspect, keywords in ASPECT_KEYWORDS.items():
            for kw in keywords:
                start = text.find(kw)
                if start == -1:
                    continue
                end = start + len(kw)
                first = True
                for idx, (ts, te) in enumerate(offsets):
                    if ts == te:
                        continue
                    if ts < end and te > start and labels_out[idx] == "O":
                        labels_out[idx] = f"B-{aspect}" if first else f"I-{aspect}"
                        first = False

        # Filter to non-special tokens
        preds = []
        for lid, label in zip(ex["labels"], labels_out):
            if lid == -100:
                continue
            preds.append(label)
        all_preds.append(preds)
    return all_preds

def _get_offsets(text, tokenizer, length):
    """Get offset mapping for a text."""
    enc = tokenizer(text, truncation=True, max_length=length,
                    padding="max_length", return_offsets_mapping=True)
    return enc["offset_mapping"]

# =============================================================================
# EVALUATION
# =============================================================================

def get_gold_tags(examples, id2label):
    """Extract gold tag sequences from examples."""
    all_gold = []
    for ex in examples:
        gold = []
        for l in ex["labels"]:
            if l == -100:
                continue
            gold.append(id2label.get(l, "O"))
        all_gold.append(gold)
    return all_gold

def evaluate_baseline(name, preds, gold):
    """Compute and print metrics."""
    f1 = f1_score(gold, preds)
    p = precision_score(gold, preds)
    r = recall_score(gold, preds)

    # Token accuracy
    correct, total = 0, 0
    for gs, ps in zip(gold, preds):
        for g, pr in zip(gs, ps):
            total += 1
            if g == pr:
                correct += 1
    acc = correct / total if total else 0

    print(f"  {name:<20s}  Acc={acc:.3f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    return {"name": name, "token_accuracy": acc, "precision": p, "recall": r, "f1": f1}

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run baselines")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/baselines")
    parser.add_argument("--split", default="test", help="Which split to evaluate on")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load label mapping
    with open(data_dir / "id2label.json") as f:
        id2label_str = json.load(f)
    id2label = {int(k): v for k, v in id2label_str.items()}

    # Load test data
    test_path = data_dir / f"{args.split}.jsonl"
    examples = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples from {test_path}")

    # Compute label distribution from training data
    train_path = data_dir / "train.jsonl"
    label_counts = Counter()
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            for l in ex["labels"]:
                if l != -100:
                    label_counts[id2label.get(l, "O")] += 1

    total_labels = sum(label_counts.values())
    label_dist = {k: v / total_labels for k, v in label_counts.items()}

    # Gold tags
    gold = get_gold_tags(examples, id2label)

    # Load tokenizer for keyword baseline
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("onlplab/alephbert-base")

    # Run baselines
    print(f"\n{'='*70}")
    print(f"BASELINE RESULTS ({args.split} set, {len(examples)} examples)")
    print(f"{'='*70}")

    results = []

    # 1. Majority
    preds = majority_predict(examples, id2label)
    results.append(evaluate_baseline("Majority (all O)", preds, gold))

    # 2. Random
    preds = random_predict(examples, id2label, label_dist)
    results.append(evaluate_baseline("Random", preds, gold))

    # 3. Keyword
    preds = keyword_predict(examples, id2label, tokenizer)
    results.append(evaluate_baseline("Keyword Rules", preds, gold))

    print(f"{'='*70}")

    # Save results
    with open(out_dir / "baseline_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'baseline_results.json'}")

if __name__ == "__main__":
    main()