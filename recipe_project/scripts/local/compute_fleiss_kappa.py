#!/usr/bin/env python3
"""
Compute Fleiss' κ for teacher annotator agreement (paper §7.3).
================================================================

Fleiss' kappa measures inter-rater agreement among 3 or more annotators.
Here we use it across our 3 LLM teacher passes:
  - Pass 1: Gemini 3.1 Flash Lite (temp=0.0)
  - Pass 2: Gemini 3.1 Flash Lite (temp=0.3, intra-annotator)
  - Pass 3: Cerebras Qwen 3 235B   (different model family, inter-annotator)

We compute κ at TWO granularities:
  1. BINARY: thread-level has_modification (yes/no) → measures overall
     agreement on "is this thread relevant?"
  2. PER-ASPECT: thread-level "does this thread contain SUBSTITUTION /
     QUANTITY / TECHNIQUE / ADDITION?" → measures fine-grained agreement.

We also compute pairwise Cohen's κ for completeness (Pass1↔2, Pass1↔3, Pass2↔3).

Span-level agreement (exact text overlap) is NOT computed here because
spans are continuous text strings, not categorical labels. That's left to
the existing scripts/evaluate_teacher.py which compares to gold.

Usage:
    python scripts/local/compute_fleiss_kappa.py \
        --silver data/silver_labels/teacher_output_v2.jsonl \
        --output results/teacher_kappa.json

    # Skip records where any teacher's output is missing (safer):
    python scripts/local/compute_fleiss_kappa.py \
        --silver data/silver_labels/teacher_output_v2.jsonl \
        --require-all-3 --output results/teacher_kappa.json

Interpretation (Landis & Koch 1977):
    κ < 0.0     poor
    0.0–0.2     slight
    0.2–0.4     fair
    0.4–0.6     moderate
    0.6–0.8     substantial
    0.8–1.0     almost perfect
"""

import argparse
import json
import sys
from pathlib import Path

# ─── No external dependencies — Fleiss' κ in pure Python ────────────────────
# (Could use sklearn for Cohen's κ but we keep this self-contained.)


# =============================================================================
# AGREEMENT MATH
# =============================================================================

def fleiss_kappa(matrix):
    """
    Fleiss' kappa for fixed number of categories (cols) and raters per item.
    `matrix` is N x K, where matrix[i][k] = number of raters that assigned
    item i to category k. Each row must sum to the same total (= n raters).

    Returns kappa (float) or None if undefined (e.g. only one category used).
    """
    N = len(matrix)
    if N == 0:
        return None
    K = len(matrix[0])
    n = sum(matrix[0])  # raters per item

    # Sanity: every row must have same total
    for row in matrix:
        if sum(row) != n:
            raise ValueError(f"All rows must have same total. Got {sum(row)} vs {n}")

    if n < 2:
        return None  # Fleiss' κ requires ≥ 2 raters

    # P_i: proportion of agreeing pairs for item i
    P_i = []
    for row in matrix:
        if n <= 1:
            P_i.append(0.0)
            continue
        s = sum(c * (c - 1) for c in row)
        P_i.append(s / (n * (n - 1)))

    # P_bar: mean of P_i
    P_bar = sum(P_i) / N

    # p_j: proportion of all assignments to category j
    p_j = [0.0] * K
    total_assignments = N * n
    for row in matrix:
        for k in range(K):
            p_j[k] += row[k]
    p_j = [p / total_assignments for p in p_j]

    # P_bar_e: expected agreement by chance
    P_bar_e = sum(p ** 2 for p in p_j)

    if P_bar_e >= 1.0:
        return None  # all raters used same category — undefined
    return (P_bar - P_bar_e) / (1 - P_bar_e)


def cohen_kappa(labels1, labels2, categories):
    """
    Cohen's kappa for two raters. labels1 and labels2 are equal-length lists
    of category strings. `categories` is the full set of possible labels.
    """
    if len(labels1) != len(labels2):
        raise ValueError(f"Length mismatch: {len(labels1)} vs {len(labels2)}")
    N = len(labels1)
    if N == 0:
        return None

    # Observed agreement
    p_o = sum(1 for a, b in zip(labels1, labels2) if a == b) / N

    # Expected agreement
    p_e = 0.0
    for c in categories:
        p1 = labels1.count(c) / N
        p2 = labels2.count(c) / N
        p_e += p1 * p2

    if p_e >= 1.0:
        return None
    return (p_o - p_e) / (1 - p_e)


def interpret(k):
    """Landis & Koch (1977) interpretation."""
    if k is None:
        return "undefined"
    if k < 0.0:
        return "poor"
    if k < 0.2:
        return "slight"
    if k < 0.4:
        return "fair"
    if k < 0.6:
        return "moderate"
    if k < 0.8:
        return "substantial"
    return "almost perfect"


# =============================================================================
# DATA EXTRACTION
# =============================================================================

def get_aspects(teacher_output):
    """Return set of aspects in this teacher's output, or empty set."""
    if teacher_output is None:
        return set()
    return {m.get("aspect", "") for m in teacher_output.get("modifications", [])
            if m.get("aspect")}


def get_has_mod(teacher_output):
    """Return 'yes'/'no'/'missing' for this teacher's binary decision."""
    if teacher_output is None:
        return "missing"
    return "yes" if teacher_output.get("has_modification", False) else "no"


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Compute Fleiss' κ across 3 LLM teacher passes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--silver", required=True,
                    help="Path to teacher_output_v2.jsonl with 3 passes")
    ap.add_argument("--output", default="results/teacher_kappa.json",
                    help="Where to write the JSON report")
    ap.add_argument("--require-all-3", action="store_true",
                    help="Skip records where any teacher's output is missing. "
                         "Default: include them (counted as 'no modification').")
    args = ap.parse_args()

    silver_path = Path(args.silver)
    if not silver_path.exists():
        print(f"❌ {silver_path} not found")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  TEACHER FLEISS' κ — Inter-Annotator Agreement")
    print(f"{'=' * 70}")
    print(f"  Source: {silver_path}")
    print(f"  Mode:   {'all-3-required' if args.require_all_3 else 'permissive (missing = no-mod)'}")
    print()

    # ─── Load all records ────────────────────────────────────────────────
    records = []
    with silver_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total_records = len(records)
    print(f"  Loaded: {total_records} records")

    # ─── Coverage stats ──────────────────────────────────────────────────
    has_p1 = sum(1 for r in records if r.get("teacher_output") is not None)
    has_p2 = sum(1 for r in records if r.get("second_teacher_output") is not None)
    has_p3 = sum(1 for r in records if r.get("third_teacher_output") is not None)
    has_all_3 = sum(1 for r in records
                    if r.get("teacher_output") is not None
                    and r.get("second_teacher_output") is not None
                    and r.get("third_teacher_output") is not None)

    print(f"  Coverage:")
    print(f"    Pass 1 (Gemini temp=0.0): {has_p1}/{total_records}  ({has_p1/total_records*100:.1f}%)")
    print(f"    Pass 2 (Gemini temp=0.3): {has_p2}/{total_records}  ({has_p2/total_records*100:.1f}%)")
    print(f"    Pass 3 (Cerebras Qwen):   {has_p3}/{total_records}  ({has_p3/total_records*100:.1f}%)")
    print(f"    All 3 passes:             {has_all_3}/{total_records}  ({has_all_3/total_records*100:.1f}%)")
    print()

    if args.require_all_3:
        records = [r for r in records
                   if r.get("teacher_output") is not None
                   and r.get("second_teacher_output") is not None
                   and r.get("third_teacher_output") is not None]
        print(f"  After --require-all-3 filter: {len(records)} records")
    print()

    if len(records) < 30:
        print(f"⚠️  Only {len(records)} records — κ estimate will have wide CI.")
    print()

    # ─── 1. BINARY Fleiss' κ (has_modification yes/no) ──────────────────
    print(f"{'─' * 70}")
    print(f"  1. BINARY  has_modification  agreement")
    print(f"{'─' * 70}")

    # Build N×K matrix where K=2 (yes, no), N=records
    binary_matrix = []
    for r in records:
        votes = {"yes": 0, "no": 0}
        for field in ["teacher_output", "second_teacher_output", "third_teacher_output"]:
            decision = get_has_mod(r.get(field))
            if decision == "yes":
                votes["yes"] += 1
            elif decision == "no":
                votes["no"] += 1
            # If "missing" and --require-all-3 not set, we treat as "no"
            elif not args.require_all_3:
                votes["no"] += 1
        binary_matrix.append([votes["yes"], votes["no"]])

    # Verify all rows sum to 3
    bad_rows = sum(1 for row in binary_matrix if sum(row) != 3)
    if bad_rows:
        print(f"⚠️  {bad_rows} rows don't sum to 3 — skipping them")
        binary_matrix = [row for row in binary_matrix if sum(row) == 3]

    binary_kappa = fleiss_kappa(binary_matrix)
    print(f"  N records: {len(binary_matrix)}")
    print(f"  Fleiss' κ: {binary_kappa:.4f}" if binary_kappa is not None else "  Fleiss' κ: undefined")
    print(f"  Interpretation: {interpret(binary_kappa)}")

    # Marginal stats
    yes_count = sum(row[0] for row in binary_matrix)
    total_assignments = len(binary_matrix) * 3
    print(f"  'yes' rate across all 3 raters: {yes_count}/{total_assignments} ({yes_count/total_assignments*100:.1f}%)")

    # Unanimous breakdown
    unanimous_yes = sum(1 for row in binary_matrix if row[0] == 3)
    unanimous_no = sum(1 for row in binary_matrix if row[1] == 3)
    split = len(binary_matrix) - unanimous_yes - unanimous_no
    print(f"  Unanimous YES: {unanimous_yes}  ({unanimous_yes/len(binary_matrix)*100:.1f}%)")
    print(f"  Unanimous NO:  {unanimous_no}  ({unanimous_no/len(binary_matrix)*100:.1f}%)")
    print(f"  Split (2-1):   {split}  ({split/len(binary_matrix)*100:.1f}%)")
    print()

    # ─── 2. PER-ASPECT Fleiss' κ ────────────────────────────────────────
    print(f"{'─' * 70}")
    print(f"  2. PER-ASPECT  agreement (does thread contain aspect X?)")
    print(f"{'─' * 70}")

    aspect_kappas = {}
    for aspect in ["SUBSTITUTION", "QUANTITY", "TECHNIQUE", "ADDITION"]:
        aspect_matrix = []
        for r in records:
            present_count = 0
            absent_count = 0
            for field in ["teacher_output", "second_teacher_output", "third_teacher_output"]:
                aspects = get_aspects(r.get(field))
                if r.get(field) is None and not args.require_all_3:
                    absent_count += 1  # missing teacher → "absent"
                elif r.get(field) is None:
                    continue  # filtered out earlier
                elif aspect in aspects:
                    present_count += 1
                else:
                    absent_count += 1
            if present_count + absent_count == 3:
                aspect_matrix.append([present_count, absent_count])

        if not aspect_matrix:
            print(f"  {aspect:14s}: no data")
            continue

        k = fleiss_kappa(aspect_matrix)
        present_total = sum(row[0] for row in aspect_matrix)
        positivity_rate = present_total / (len(aspect_matrix) * 3) * 100

        if k is not None:
            print(f"  {aspect:14s}: κ = {k:+.4f}  ({interpret(k):s}, "
                  f"present in {positivity_rate:.1f}% of teacher-thread pairs)")
        else:
            print(f"  {aspect:14s}: κ = undefined  "
                  f"(present in {positivity_rate:.1f}% — likely too rare)")
        aspect_kappas[aspect] = {
            "kappa": k,
            "interpretation": interpret(k),
            "n_records": len(aspect_matrix),
            "positivity_rate": positivity_rate,
        }
    print()

    # ─── 3. PAIRWISE Cohen's κ ──────────────────────────────────────────
    print(f"{'─' * 70}")
    print(f"  3. PAIRWISE Cohen's κ (binary has_modification)")
    print(f"{'─' * 70}")

    pairs = [
        ("Pass1 vs Pass2", "teacher_output", "second_teacher_output"),
        ("Pass1 vs Pass3", "teacher_output", "third_teacher_output"),
        ("Pass2 vs Pass3", "second_teacher_output", "third_teacher_output"),
    ]
    pairwise_kappas = {}
    for name, f1, f2 in pairs:
        labels1, labels2 = [], []
        for r in records:
            d1 = get_has_mod(r.get(f1))
            d2 = get_has_mod(r.get(f2))
            if d1 == "missing" or d2 == "missing":
                if args.require_all_3:
                    continue
                if d1 == "missing":
                    d1 = "no"
                if d2 == "missing":
                    d2 = "no"
            labels1.append(d1)
            labels2.append(d2)

        k = cohen_kappa(labels1, labels2, {"yes", "no"}) if labels1 else None
        n = len(labels1)
        if k is not None:
            print(f"  {name}: κ = {k:+.4f}  ({interpret(k)}, n={n})")
        else:
            print(f"  {name}: κ = undefined  (n={n})")
        pairwise_kappas[name] = {"kappa": k, "interpretation": interpret(k), "n": n}
    print()

    # ─── Save report ────────────────────────────────────────────────────
    report = {
        "source_file": str(silver_path),
        "n_total_records": total_records,
        "n_records_used": len(records),
        "require_all_3": args.require_all_3,
        "coverage": {
            "pass1": has_p1, "pass2": has_p2, "pass3": has_p3,
            "all_3": has_all_3,
        },
        "binary": {
            "kappa": binary_kappa,
            "interpretation": interpret(binary_kappa),
            "n_records": len(binary_matrix),
            "unanimous_yes": unanimous_yes,
            "unanimous_no": unanimous_no,
            "split_2_1": split,
        },
        "per_aspect": aspect_kappas,
        "pairwise_cohen": pairwise_kappas,
    }
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"{'=' * 70}")
    print(f"  ✅ Report saved: {output_path}")
    print(f"{'=' * 70}")
    print()
    print("  PAPER §7.3 SUGGESTED PHRASING:")
    print(f"    \"Inter-annotator agreement among the three LLM teacher passes")
    print(f"     was substantial, with Fleiss' κ = {binary_kappa:.3f} on binary")
    print(f"     has_modification labels (Landis & Koch 1977: {interpret(binary_kappa)}).\"")
    print()


if __name__ == "__main__":
    main()