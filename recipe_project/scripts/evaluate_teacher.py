#!/usr/bin/env python3
"""
Teacher Evaluation — Upper Bound Check
========================================
Compares silver (teacher) labels against gold (human) annotations to
establish the ceiling for student model performance. This is the GO/NO-GO
gate from MASTER_PLAN_v7 §5.3.

If teacher span F1 < 0.40, the silver labels are too noisy and you should
re-examine the labeling pipeline before training student models.

Usage:
    python scripts/evaluate_teacher.py

    python scripts/evaluate_teacher.py \
        --gold data/gold_validation/gold_final.jsonl \
        --silver data/silver_labels/teacher_output.jsonl \
        --output results/teacher_upper_bound.json
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


def load_gold(path: str) -> Dict[str, Dict]:
    """Load gold annotations indexed by thread_id.

    Expected format per line:
    {
        "thread_id": "Ugw123abc",
        "comment_text": "...",
        "comment_position": "top",  # or "reply_1", etc.
        "has_modification": true/false,
        "gold_modifications": [{"span": "...", "aspect": "SUBSTITUTION"}, ...],
        "annotator": "adjudicated"
    }
    """
    gold = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Key: thread_id + comment_position for uniqueness
            tid = rec.get("thread_id", "")
            pos = rec.get("comment_position", "top")
            key = f"{tid}_{pos}"
            gold[key] = rec
    return gold


def load_silver(path: str, gold_thread_ids: set) -> Dict[str, Dict]:
    """Load silver labels, filtered to only threads in the gold set.

    We need to match gold thread_ids against silver records.
    """
    silver = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = rec.get("thread_id", "")
            if tid not in gold_thread_ids:
                continue
            silver[tid] = rec
    return silver


def extract_spans(modifications: List[Dict]) -> List[Tuple[str, str]]:
    """Extract (span_text, aspect) tuples from a modifications list."""
    spans = []
    for mod in modifications:
        span = mod.get("span", "").strip()
        aspect = mod.get("aspect", "UNKNOWN")
        if span:
            spans.append((span, aspect))
    return spans


def compute_binary_metrics(gold: Dict, silver: Dict) -> Dict:
    """Compute binary classification metrics (has_modification yes/no)."""
    tp, fp, fn, tn = 0, 0, 0, 0

    gold_thread_ids = set()
    for key, rec in gold.items():
        tid = rec.get("thread_id", "")
        gold_thread_ids.add(tid)

    for key, g_rec in gold.items():
        tid = g_rec.get("thread_id", "")
        g_has_mod = g_rec.get("has_modification", False)

        # Find matching silver record
        s_rec = silver.get(tid, {})
        s_label = s_rec.get("final_label") or s_rec.get("teacher_output") or {}
        s_has_mod = s_label.get("has_modification", False)

        if g_has_mod and s_has_mod:
            tp += 1
        elif g_has_mod and not s_has_mod:
            fn += 1
        elif not g_has_mod and s_has_mod:
            fp += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total,
    }


def compute_span_metrics(gold: Dict, silver: Dict) -> Dict:
    """Compute span-level entity metrics using seqeval-style matching.

    For each gold thread, we compare gold spans against silver spans.
    A span is a match if both the text and aspect match (exact match).
    We also compute a relaxed match (text overlap + correct aspect).
    """
    # Exact span matching
    exact_tp = 0
    exact_fp = 0
    exact_fn = 0

    # Relaxed matching (text substring overlap + same aspect)
    relaxed_tp = 0
    relaxed_fp = 0
    relaxed_fn = 0

    per_aspect = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for key, g_rec in gold.items():
        tid = g_rec.get("thread_id", "")
        g_mods = g_rec.get("gold_modifications", [])
        g_spans = extract_spans(g_mods)

        # Get silver spans for this thread
        s_rec = silver.get(tid, {})
        s_label = s_rec.get("final_label") or s_rec.get("teacher_output") or {}
        s_mods = s_label.get("modifications", [])
        s_spans = extract_spans(s_mods)

        # Exact matching
        g_set = set(g_spans)
        s_set = set(s_spans)

        matched_gold = set()
        matched_silver = set()

        # Exact matches
        for gs in g_spans:
            if gs in s_set and gs not in matched_gold:
                exact_tp += 1
                per_aspect[gs[1]]["tp"] += 1
                matched_gold.add(gs)
                matched_silver.add(gs)

        # Relaxed matching for unmatched
        unmatched_gold = [gs for gs in g_spans if gs not in matched_gold]
        unmatched_silver = [ss for ss in s_spans if ss not in matched_silver]

        relaxed_matched_g = set()
        relaxed_matched_s = set()

        for i, gs in enumerate(unmatched_gold):
            g_text, g_aspect = gs
            for j, ss in enumerate(unmatched_silver):
                if j in relaxed_matched_s:
                    continue
                s_text, s_aspect = ss
                # Relaxed: same aspect + text overlap (one contains the other)
                if g_aspect == s_aspect and (g_text in s_text or s_text in g_text):
                    relaxed_tp += 1
                    relaxed_matched_g.add(i)
                    relaxed_matched_s.add(j)
                    break

        # Count FP/FN
        for gs in g_spans:
            if gs not in matched_gold:
                exact_fn += 1
                per_aspect[gs[1]]["fn"] += 1
        for ss in s_spans:
            if ss not in matched_silver:
                exact_fp += 1
                per_aspect[ss[1]]["fp"] += 1

        relaxed_fn_count = len(unmatched_gold) - len(relaxed_matched_g)
        relaxed_fp_count = len(unmatched_silver) - len(relaxed_matched_s)
        relaxed_fn += relaxed_fn_count
        relaxed_fp += relaxed_fp_count

    # Compute F1s
    def calc_f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return round(p, 4), round(r, 4), round(f1, 4)

    exact_p, exact_r, exact_f1 = calc_f1(exact_tp, exact_fp, exact_fn)

    total_relaxed_tp = exact_tp + relaxed_tp
    total_relaxed_fp = exact_fp + relaxed_fp
    total_relaxed_fn = exact_fn + relaxed_fn
    relax_p, relax_r, relax_f1 = calc_f1(total_relaxed_tp, total_relaxed_fp, total_relaxed_fn)

    # Per-aspect breakdown
    per_aspect_results = {}
    for aspect, counts in per_aspect.items():
        ap, ar, af1 = calc_f1(counts["tp"], counts["fp"], counts["fn"])
        per_aspect_results[aspect] = {"precision": ap, "recall": ar, "f1": af1}

    return {
        "exact": {
            "precision": exact_p,
            "recall": exact_r,
            "f1": exact_f1,
            "tp": exact_tp, "fp": exact_fp, "fn": exact_fn,
        },
        "relaxed": {
            "precision": relax_p,
            "recall": relax_r,
            "f1": relax_f1,
            "tp": total_relaxed_tp, "fp": total_relaxed_fp, "fn": total_relaxed_fn,
        },
        "per_aspect": per_aspect_results,
    }


def evaluate_teacher(gold_path: str, silver_path: str, output_path: str):
    """Run full teacher evaluation and print go/no-go decision."""

    print(f"{'='*60}")
    print(f"TEACHER EVALUATION — Upper Bound (GO/NO-GO GATE)")
    print(f"{'='*60}")
    print(f"  Gold:   {gold_path}")
    print(f"  Silver: {silver_path}")
    print()

    # Load gold
    gold = load_gold(gold_path)
    print(f"  Gold examples loaded: {len(gold)}")

    # Get thread IDs from gold
    gold_thread_ids = set()
    for key, rec in gold.items():
        gold_thread_ids.add(rec.get("thread_id", ""))

    # Load matching silver
    silver = load_silver(silver_path, gold_thread_ids)
    print(f"  Silver records matched: {len(silver)}")

    unmatched = gold_thread_ids - set(silver.keys())
    if unmatched:
        print(f"  ⚠️  {len(unmatched)} gold threads NOT found in silver labels")

    # Binary metrics
    print(f"\n--- Binary Classification (has_modification) ---")
    binary = compute_binary_metrics(gold, silver)
    print(f"  Accuracy:  {binary['accuracy']:.4f}")
    print(f"  Precision: {binary['precision']:.4f}")
    print(f"  Recall:    {binary['recall']:.4f}")
    print(f"  F1:        {binary['f1']:.4f}")
    print(f"  (TP={binary['tp']}, FP={binary['fp']}, FN={binary['fn']}, TN={binary['tn']})")

    # Span metrics
    print(f"\n--- Span-Level Entity Metrics ---")
    span = compute_span_metrics(gold, silver)

    print(f"\n  Exact match:")
    print(f"    Precision: {span['exact']['precision']:.4f}")
    print(f"    Recall:    {span['exact']['recall']:.4f}")
    print(f"    F1:        {span['exact']['f1']:.4f}")

    print(f"\n  Relaxed match (text overlap + correct aspect):")
    print(f"    Precision: {span['relaxed']['precision']:.4f}")
    print(f"    Recall:    {span['relaxed']['recall']:.4f}")
    print(f"    F1:        {span['relaxed']['f1']:.4f}")

    if span["per_aspect"]:
        print(f"\n  Per-aspect breakdown (exact):")
        for aspect, metrics in sorted(span["per_aspect"].items()):
            print(f"    {aspect:<15s}  P={metrics['precision']:.3f}  "
                  f"R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")

    # GO/NO-GO decision
    teacher_f1 = span["relaxed"]["f1"]  # Use relaxed as primary metric
    binary_acc = binary["accuracy"]

    print(f"\n{'='*60}")
    print(f"DECISION")
    print(f"{'='*60}")
    print(f"  Teacher span F1 (relaxed): {teacher_f1:.4f}")
    print(f"  Teacher binary accuracy:   {binary_acc:.4f}")

    if teacher_f1 >= 0.65:
        decision = "GO — Labels are good"
        target = "Student target: 0.45-0.55 F1"
    elif teacher_f1 >= 0.50:
        decision = "GO — Labels are acceptable"
        target = "Student target: 0.35-0.45 F1"
    elif teacher_f1 >= 0.40:
        decision = "CAUTION — Labels are noisy"
        target = "Student target: 0.30-0.40 F1"
    else:
        if binary_acc >= 0.85:
            decision = "CONDITIONAL GO — Span labels are poor but binary is decent"
            target = ("Binary detection is usable but span extraction will be weak. "
                      "Consider re-labeling or using simpler task formulation.")
        else:
            decision = "NO-GO — Labels are broken"
            target = "STOP. Re-examine labeling pipeline before training."

    print(f"\n  >>> {decision}")
    print(f"  >>> {target}")
    print(f"{'='*60}")

    # Save report
    report = {
        "gold_path": gold_path,
        "silver_path": silver_path,
        "gold_count": len(gold),
        "silver_matched": len(silver),
        "binary_metrics": binary,
        "span_metrics": span,
        "decision": decision,
        "target": target,
        "teacher_span_f1_relaxed": teacher_f1,
        "teacher_binary_accuracy": binary_acc,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate teacher (silver labels) against gold annotations"
    )
    parser.add_argument("--gold",
                        default="data/gold_validation/gold_final.jsonl",
                        help="Path to adjudicated gold annotations JSONL")
    parser.add_argument("--silver",
                        default="data/silver_labels/teacher_output.jsonl",
                        help="Path to silver labels JSONL")
    parser.add_argument("--output",
                        default="results/teacher_upper_bound.json",
                        help="Output path for evaluation report")
    args = parser.parse_args()

    if not Path(args.gold).exists():
        print(f"❌ Gold file not found: {args.gold}")
        print(f"   Complete gold annotation (Step 0.2) before running this.")
        return
    if not Path(args.silver).exists():
        print(f"❌ Silver file not found: {args.silver}")
        return

    evaluate_teacher(args.gold, args.silver, args.output)


if __name__ == "__main__":
    main()
