#!/usr/bin/env python3
"""
Teacher Upper Bound Evaluation: Compare silver labels (teacher) against gold annotations.

This establishes the CEILING for student model performance.
If teacher F1 < 0.50, student cannot exceed 0.50 regardless of architecture.

Also computes inter-annotator agreement if two annotator files are provided.

Usage:
    # Teacher vs gold:
    python scripts/evaluate_teacher.py \
        --silver data/silver_labels/teacher_output.jsonl \
        --gold data/gold_validation/gold_final.jsonl \
        --output results/teacher_upper_bound.json

    # Inter-annotator agreement:
    python scripts/evaluate_teacher.py \
        --silver data/silver_labels/teacher_output.jsonl \
        --gold data/gold_validation/gold_final.jsonl \
        --annotator-a data/gold_validation/gold_daniel.jsonl \
        --annotator-b data/gold_validation/gold_roei.jsonl \
        --output results/teacher_upper_bound.json
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

from seqeval.metrics import (
    f1_score as seq_f1,
    precision_score as seq_precision,
    recall_score as seq_recall,
    classification_report as seq_report,
)

VALID_ASPECTS = {"SUBSTITUTION", "QUANTITY", "TECHNIQUE", "ADDITION"}


# =============================================================================
# LOAD DATA
# =============================================================================

def load_gold(path):
    """
    Load gold annotations.

    Expected format per line:
    {
        "thread_id": "Ugw123",
        "comment_text": "...",
        "comment_position": "top",
        "has_modification": true/false,
        "gold_modifications": [{"span": "...", "aspect": "SUBSTITUTION"}],
        "annotator": "daniel"
    }
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_silver(path):
    """Load silver labels (teacher_output.jsonl)."""
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records[rec["thread_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return records


# =============================================================================
# CONVERT MODIFICATIONS TO WORD-LEVEL BIO
# =============================================================================

def mods_to_word_bio(text, modifications):
    """Convert modifications to word-level BIO sequence."""
    words = text.split()
    bio = ["O"] * len(words)

    for mod in modifications:
        span = mod.get("span", "").strip()
        aspect = mod.get("aspect", "")
        if not span or aspect not in VALID_ASPECTS:
            continue

        span_words = span.split()
        if not span_words:
            continue

        # Find span in word list
        for i in range(len(words) - len(span_words) + 1):
            # Exact match
            if words[i:i + len(span_words)] == span_words:
                for j in range(len(span_words)):
                    pos = i + j
                    if bio[pos] == "O":
                        prefix = "B" if j == 0 else "I"
                        bio[pos] = f"{prefix}-{aspect}"
                break
            # Fuzzy match (strip punctuation from words)
            clean_words = [re.sub(r'[^\w\u0590-\u05FF]', '', w) for w in words[i:i + len(span_words)]]
            clean_span = [re.sub(r'[^\w\u0590-\u05FF]', '', w) for w in span_words]
            if clean_words == clean_span:
                for j in range(len(span_words)):
                    pos = i + j
                    if bio[pos] == "O":
                        prefix = "B" if j == 0 else "I"
                        bio[pos] = f"{prefix}-{aspect}"
                break

    return bio


# =============================================================================
# TEACHER VS GOLD EVALUATION
# =============================================================================

def evaluate_teacher_vs_gold(silver_records, gold_records):
    """
    Compare teacher (silver) labels against gold annotations.
    Returns entity-level metrics using seqeval.
    """
    all_teacher_preds = []
    all_gold_labels = []
    matched = 0
    unmatched = 0

    # Binary agreement
    binary_agree = 0
    binary_total = 0

    for gold_item in gold_records:
        tid = gold_item["thread_id"]
        silver_rec = silver_records.get(tid)

        if silver_rec is None:
            unmatched += 1
            continue

        matched += 1
        text = gold_item["comment_text"]
        words = text.split()
        n_words = len(words)

        if n_words == 0:
            continue

        # Gold BIO
        gold_mods = gold_item.get("gold_modifications", [])
        gold_bio = mods_to_word_bio(text, gold_mods)

        # Silver BIO — need to find the right comment in the silver record
        position = gold_item.get("comment_position", "top")
        silver_label = silver_rec.get("final_label") or silver_rec.get("teacher_output") or {}
        silver_mods = []

        for mod in silver_label.get("modifications", []):
            src = mod.get("source_comment", "top")
            if src == position:
                silver_mods.append(mod)

        teacher_bio = mods_to_word_bio(text, silver_mods)

        # Ensure same length
        min_len = min(len(gold_bio), len(teacher_bio), n_words)
        gold_bio = gold_bio[:min_len]
        teacher_bio = teacher_bio[:min_len]

        if gold_bio:
            all_gold_labels.append(gold_bio)
            all_teacher_preds.append(teacher_bio)

        # Binary agreement
        gold_has = gold_item.get("has_modification", False)
        silver_has = silver_label.get("has_modification", False)
        if gold_has == silver_has:
            binary_agree += 1
        binary_total += 1

    if not all_gold_labels:
        return {
            "error": "No matching gold/silver examples found",
            "matched": matched,
            "unmatched": unmatched,
        }

    results = {
        "matched_examples": matched,
        "unmatched_examples": unmatched,
        "entity_f1": seq_f1(all_gold_labels, all_teacher_preds),
        "entity_precision": seq_precision(all_gold_labels, all_teacher_preds),
        "entity_recall": seq_recall(all_gold_labels, all_teacher_preds),
        "binary_agreement": binary_agree / max(binary_total, 1),
        "classification_report": seq_report(all_gold_labels, all_teacher_preds),
    }

    return results


# =============================================================================
# INTER-ANNOTATOR AGREEMENT
# =============================================================================

def compute_cohens_kappa(labels_a, labels_b):
    """Compute Cohen's kappa for binary has_modification agreement."""
    n = len(labels_a)
    if n == 0:
        return 0.0

    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agree / n

    # Expected agreement
    pos_a = sum(labels_a) / n
    pos_b = sum(labels_b) / n
    pe = pos_a * pos_b + (1 - pos_a) * (1 - pos_b)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def evaluate_inter_annotator(records_a, records_b):
    """Compare two annotators' gold labels."""
    # Index by thread_id + position
    lookup_a = {}
    for rec in records_a:
        key = (rec["thread_id"], rec.get("comment_position", "top"))
        lookup_a[key] = rec

    binary_a = []
    binary_b = []
    span_golds_a = []
    span_golds_b = []
    matched = 0

    for rec_b in records_b:
        key = (rec_b["thread_id"], rec_b.get("comment_position", "top"))
        rec_a = lookup_a.get(key)
        if rec_a is None:
            continue

        matched += 1

        # Binary
        binary_a.append(1 if rec_a.get("has_modification", False) else 0)
        binary_b.append(1 if rec_b.get("has_modification", False) else 0)

        # Span-level
        text = rec_a.get("comment_text", rec_b.get("comment_text", ""))
        if text:
            bio_a = mods_to_word_bio(text, rec_a.get("gold_modifications", []))
            bio_b = mods_to_word_bio(text, rec_b.get("gold_modifications", []))
            min_len = min(len(bio_a), len(bio_b))
            if min_len > 0:
                span_golds_a.append(bio_a[:min_len])
                span_golds_b.append(bio_b[:min_len])

    kappa = compute_cohens_kappa(binary_a, binary_b)

    span_f1 = 0.0
    if span_golds_a and span_golds_b:
        span_f1 = seq_f1(span_golds_a, span_golds_b)

    return {
        "matched_examples": matched,
        "binary_kappa": round(kappa, 4),
        "binary_agreement_rate": round(sum(a == b for a, b in zip(binary_a, binary_b)) / max(len(binary_a), 1), 4),
        "span_f1": round(span_f1, 4),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Teacher upper bound evaluation")
    parser.add_argument("--silver", required=True,
                        help="Path to teacher_output.jsonl (silver labels)")
    parser.add_argument("--gold", required=True,
                        help="Path to gold_final.jsonl (adjudicated gold labels)")
    parser.add_argument("--annotator-a", default=None,
                        help="Path to annotator A's raw annotations (for kappa)")
    parser.add_argument("--annotator-b", default=None,
                        help="Path to annotator B's raw annotations (for kappa)")
    parser.add_argument("--output", default="results/teacher_upper_bound.json")
    args = parser.parse_args()

    print("=" * 60)
    print("TEACHER UPPER BOUND EVALUATION")
    print("=" * 60)

    # Load data
    silver = load_silver(args.silver)
    print(f"Silver records: {len(silver)}")

    gold = load_gold(args.gold)
    print(f"Gold records:   {len(gold)}")

    # Teacher vs Gold
    print(f"\n{'='*60}")
    print("TEACHER (SILVER) vs. GOLD")
    print(f"{'='*60}")

    results = evaluate_teacher_vs_gold(silver, gold)

    if "error" in results:
        print(f"  ERROR: {results['error']}")
    else:
        print(f"  Matched:            {results['matched_examples']}")
        print(f"  Entity F1:          {results['entity_f1']:.4f}")
        print(f"  Entity Precision:   {results['entity_precision']:.4f}")
        print(f"  Entity Recall:      {results['entity_recall']:.4f}")
        print(f"  Binary Agreement:   {results['binary_agreement']:.4f}")
        print(f"\n  Classification Report:")
        print(results.get("classification_report", "N/A"))

        # GO/NO-GO interpretation
        f1 = results["entity_f1"]
        print(f"\n  {'='*50}")
        if f1 >= 0.65:
            print(f"  ✅ Teacher F1 = {f1:.2f} → Labels are GOOD. Target student: 0.45-0.55")
        elif f1 >= 0.50:
            print(f"  ✅ Teacher F1 = {f1:.2f} → Labels are OK. Target student: 0.35-0.45")
        elif f1 >= 0.40:
            print(f"  ⚠️  Teacher F1 = {f1:.2f} → Labels are NOISY. Target student: 0.30-0.40")
        else:
            print(f"  ❌ Teacher F1 = {f1:.2f} → Labels are BROKEN. STOP and fix labeling.")
            print(f"     Student CANNOT exceed teacher quality.")
        print(f"  {'='*50}")

    # Inter-annotator agreement (if both files provided)
    iaa_results = None
    if args.annotator_a and args.annotator_b:
        print(f"\n{'='*60}")
        print("INTER-ANNOTATOR AGREEMENT")
        print(f"{'='*60}")

        ann_a = load_gold(args.annotator_a)
        ann_b = load_gold(args.annotator_b)
        print(f"  Annotator A: {len(ann_a)} records")
        print(f"  Annotator B: {len(ann_b)} records")

        iaa_results = evaluate_inter_annotator(ann_a, ann_b)
        print(f"  Matched:             {iaa_results['matched_examples']}")
        print(f"  Binary Cohen's κ:    {iaa_results['binary_kappa']}")
        print(f"  Binary Agreement:    {iaa_results['binary_agreement_rate']}")
        print(f"  Span-level F1:       {iaa_results['span_f1']}")

        if iaa_results["binary_kappa"] >= 0.70:
            print(f"  ✅ κ ≥ 0.70 → Task is well-defined")
        else:
            print(f"  ⚠️  κ < 0.70 → Task definition may need clarification")

    # Save report
    report = {
        "teacher_vs_gold": {k: v for k, v in results.items()
                            if k != "classification_report"},
        "classification_report": results.get("classification_report", ""),
    }
    if iaa_results:
        report["inter_annotator_agreement"] = iaa_results

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {out_path}")


if __name__ == "__main__":
    main()
