#!/usr/bin/env python3
"""
Compute Inter-Annotator Agreement (IAA) between Daniel and Roei
on the 50-thread overlap region (threads 225-274) of the gold set.

Outputs:
  - Binary Cohen's kappa (has_modification yes/no)
  - Span-level F1 between annotators
  - Per-aspect agreement

Usage:
  python -m scripts.local.compute_iaa \
      --gold-file data/gold_validation/gold_final.jsonl \
      --output-dir results/iaa
"""

import json
import argparse
import os
from collections import Counter


def cohen_kappa(labels1, labels2):
    """Compute Cohen's kappa for two lists of binary labels."""
    assert len(labels1) == len(labels2), "Label lists must have same length"
    n = len(labels1)
    if n == 0:
        return 0.0

    # Observed agreement
    agree = sum(1 for a, b in zip(labels1, labels2) if a == b)
    p_o = agree / n

    # Expected agreement
    categories = sorted(set(labels1) | set(labels2))
    p_e = 0.0
    for c in categories:
        p1 = sum(1 for x in labels1 if x == c) / n
        p2 = sum(1 for x in labels2 if x == c) / n
        p_e += p1 * p2

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def interpret_kappa(k):
    """Landis & Koch (1977) interpretation."""
    if k < 0:
        return "Poor"
    elif k < 0.20:
        return "Slight"
    elif k < 0.40:
        return "Fair"
    elif k < 0.60:
        return "Moderate"
    elif k < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


def extract_spans(modifications):
    """Extract set of (span_text_normalized, aspect) tuples."""
    if not modifications:
        return set()
    spans = set()
    for mod in modifications:
        text = mod.get("span", mod.get("text", "")).strip().lower()
        aspect = mod.get("aspect", "").upper()
        if text and aspect:
            spans.add((text, aspect))
    return spans


def span_f1(spans1, spans2):
    """Compute span-level P, R, F1 between two sets of spans."""
    if not spans1 and not spans2:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    tp = len(spans1 & spans2)
    fp = len(spans2 - spans1)
    fn = len(spans1 - spans2)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def relaxed_span_match(span1, span2):
    """Check if two spans have the same aspect and overlapping text."""
    text1, aspect1 = span1
    text2, aspect2 = span2
    if aspect1 != aspect2:
        return False
    # Substring overlap in either direction
    return text1 in text2 or text2 in text1


def relaxed_span_f1(spans1, spans2):
    """Compute relaxed span F1 (partial text overlap + same aspect)."""
    if not spans1 and not spans2:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = 0
    matched2 = set()
    for s1 in spans1:
        for i, s2 in enumerate(spans2):
            if i not in matched2 and relaxed_span_match(s1, s2):
                tp += 1
                matched2.add(i)
                break

    spans2_list = list(spans2)
    fp = len(spans2) - len(matched2)
    fn = len(spans1) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Compute IAA on gold overlap region")
    parser.add_argument("--gold-file", default="data/gold_validation/gold_final.jsonl")
    parser.add_argument("--output-dir", default="results/iaa")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load gold data
    threads = []
    with open(args.gold_file) as f:
        for line in f:
            threads.append(json.loads(line))

    # Filter overlap region (both annotators present)
    overlap = [t for t in threads if t.get("overlap", False)
               and t.get("annotator_daniel") is not None
               and t.get("annotator_roei") is not None]

    if not overlap:
        # Try alternative: check if annotator_roei exists
        overlap = [t for t in threads
                   if t.get("annotator_daniel") is not None
                   and t.get("annotator_roei") is not None]

    print(f"Total threads in gold: {len(threads)}")
    print(f"Overlap threads (both annotators): {len(overlap)}")

    if len(overlap) == 0:
        print("ERROR: No overlap threads found. Check gold file format.")
        print("Sample thread keys:", list(threads[0].keys()) if threads else "empty")
        return

    # 1. Binary Cohen's Kappa
    daniel_binary = []
    roei_binary = []
    for t in overlap:
        d = t["annotator_daniel"].get("has_modification", False)
        r = t["annotator_roei"].get("has_modification", False)
        daniel_binary.append(1 if d else 0)
        roei_binary.append(1 if r else 0)

    kappa = cohen_kappa(daniel_binary, roei_binary)
    binary_agree = sum(1 for a, b in zip(daniel_binary, roei_binary) if a == b)

    print(f"\n{'='*60}")
    print(f"BINARY AGREEMENT (has_modification)")
    print(f"{'='*60}")
    print(f"  Cohen's kappa:    {kappa:.4f} ({interpret_kappa(kappa)})")
    print(f"  Raw agreement:    {binary_agree}/{len(overlap)} ({binary_agree/len(overlap)*100:.1f}%)")
    print(f"  Daniel positives: {sum(daniel_binary)}/{len(overlap)}")
    print(f"  Roei positives:   {sum(roei_binary)}/{len(overlap)}")

    # 2. Span-level F1
    all_daniel_spans = set()
    all_roei_spans = set()
    per_thread_results = []

    for t in overlap:
        d_mods = t["annotator_daniel"].get("gold_modifications", [])
        r_mods = t["annotator_roei"].get("gold_modifications", [])

        d_spans = extract_spans(d_mods)
        r_spans = extract_spans(r_mods)

        all_daniel_spans |= d_spans
        all_roei_spans |= r_spans

        thread_f1 = span_f1(d_spans, r_spans)
        per_thread_results.append({
            "thread_id": t.get("thread_id", "?"),
            "daniel_spans": len(d_spans),
            "roei_spans": len(r_spans),
            **thread_f1
        })

    overall_exact = span_f1(all_daniel_spans, all_roei_spans)
    overall_relaxed = relaxed_span_f1(all_daniel_spans, all_roei_spans)

    print(f"\n{'='*60}")
    print(f"SPAN-LEVEL AGREEMENT")
    print(f"{'='*60}")
    print(f"  Daniel total spans: {len(all_daniel_spans)}")
    print(f"  Roei total spans:   {len(all_roei_spans)}")
    print(f"\n  Exact match:")
    print(f"    Precision: {overall_exact['precision']:.4f}")
    print(f"    Recall:    {overall_exact['recall']:.4f}")
    print(f"    F1:        {overall_exact['f1']:.4f}")
    print(f"    TP={overall_exact['tp']} FP={overall_exact['fp']} FN={overall_exact['fn']}")
    print(f"\n  Relaxed match (text overlap + same aspect):")
    print(f"    Precision: {overall_relaxed['precision']:.4f}")
    print(f"    Recall:    {overall_relaxed['recall']:.4f}")
    print(f"    F1:        {overall_relaxed['f1']:.4f}")

    # 3. Per-aspect agreement
    aspects = ["SUBSTITUTION", "QUANTITY", "TECHNIQUE", "ADDITION"]
    print(f"\n{'='*60}")
    print(f"PER-ASPECT AGREEMENT")
    print(f"{'='*60}")

    aspect_results = {}
    for aspect in aspects:
        d_asp = {(t, a) for (t, a) in all_daniel_spans if a == aspect}
        r_asp = {(t, a) for (t, a) in all_roei_spans if a == aspect}
        result = span_f1(d_asp, r_asp)
        aspect_results[aspect] = result
        print(f"  {aspect:15s}  F1={result['f1']:.4f}  P={result['precision']:.4f}  R={result['recall']:.4f}  "
              f"(D={len(d_asp)}, R={len(r_asp)}, TP={result['tp']})")

    # 4. Disagreement examples
    print(f"\n{'='*60}")
    print(f"DISAGREEMENT EXAMPLES")
    print(f"{'='*60}")

    disagree_count = 0
    for t in overlap:
        d_mods = t["annotator_daniel"].get("gold_modifications", [])
        r_mods = t["annotator_roei"].get("gold_modifications", [])
        d_spans = extract_spans(d_mods)
        r_spans = extract_spans(r_mods)

        if d_spans != r_spans and (d_spans or r_spans):
            disagree_count += 1
            if disagree_count <= 5:
                print(f"\n  Thread: {t.get('thread_id', '?')[:30]}")
                only_d = d_spans - r_spans
                only_r = r_spans - d_spans
                if only_d:
                    print(f"    Daniel only: {only_d}")
                if only_r:
                    print(f"    Roei only:   {only_r}")

    print(f"\n  Total disagreements: {disagree_count}/{len(overlap)}")

    # Save results
    results = {
        "overlap_size": len(overlap),
        "binary": {
            "cohens_kappa": round(kappa, 4),
            "interpretation": interpret_kappa(kappa),
            "raw_agreement": f"{binary_agree}/{len(overlap)}",
            "raw_agreement_pct": round(binary_agree / len(overlap) * 100, 1),
            "daniel_positives": sum(daniel_binary),
            "roei_positives": sum(roei_binary),
        },
        "span_exact": {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in overall_exact.items()},
        "span_relaxed": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in overall_relaxed.items()},
        "per_aspect": {asp: {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in res.items()}
                       for asp, res in aspect_results.items()},
    }

    output_file = os.path.join(args.output_dir, "iaa_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_file}")

    # Print LaTeX-ready values
    print(f"\n{'='*60}")
    print(f"LATEX-READY VALUES")
    print(f"{'='*60}")
    print(f"  Cohen's $\\kappa$ = {kappa:.2f} ({interpret_kappa(kappa)})")
    print(f"  Span F$_1$ (exact) = {overall_exact['f1']:.2f}")
    print(f"  Span F$_1$ (relaxed) = {overall_relaxed['f1']:.2f}")


if __name__ == "__main__":
    main()