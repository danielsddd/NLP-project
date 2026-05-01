#!/usr/bin/env python3
"""
run_diagnostics.py — One-shot diagnostic runner
================================================
Covers three things in a single pass:

  [D1] source_comment distribution in silver
       → Tells you whether the evaluate_teacher 'top' filter is correct
         or is silently discarding real predictions.

  [D2] 10 False Positive examples + 10 Span Mismatch examples
       → These become the few-shot examples for the new SYSTEM_PROMPT
         (Track A Day 1 input).

  [D0] Day 0 pre-flight checks (from MASTER_PLAN §11.1)
       → __init__.py files, class-weights wiring, and teacher F1 sanity.

Usage (run from inside recipe_project/):
    python run_diagnostics.py

    # Override paths:
    python run_diagnostics.py \
        --silver data/silver_labels/teacher_output.jsonl \
        --gold   data/gold_validation/gold_final.jsonl \
        --output results/diagnostics_output.json

    # Skip the teacher eval re-run (fast mode, D0 check 3 skipped):
    python run_diagnostics.py --skip-eval

Output:
    Prints a structured report to stdout.
    Writes results/diagnostics_output.json with all findings.
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# SEQEVAL — optional; only needed for D2 span comparison
# ---------------------------------------------------------------------------
try:
    from seqeval.metrics import f1_score as seq_f1
    HAS_SEQEVAL = True
except ImportError:
    HAS_SEQEVAL = False

VALID_ASPECTS = {"SUBSTITUTION", "QUANTITY", "TECHNIQUE", "ADDITION"}

# ============================================================================
# HELPERS
# ============================================================================

def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] JSON parse error at line {lineno}: {e}")
    return records


def load_silver_indexed(path: str) -> Dict[str, Dict]:
    """Index silver by thread_id."""
    silver = {}
    for rec in load_jsonl(path):
        tid = rec.get("thread_id", "")
        if tid:
            silver[tid] = rec
    return silver


def load_gold_indexed(path: str) -> Dict[str, Dict]:
    """Index gold by thread_id (gold only has top-level comments)."""
    gold = {}
    for rec in load_jsonl(path):
        tid = rec.get("thread_id", "")
        if tid:
            gold[tid] = rec
    return gold


def get_silver_label(rec: Dict) -> Dict:
    """Extract the final_label or teacher_output sub-dict."""
    return rec.get("final_label") or rec.get("teacher_output") or {}


def mods_to_word_bio(text: str, modifications: List[Dict]) -> List[str]:
    """
    Convert a list of {span, aspect} modifications into word-level BIO tags.
    Each word gets O / B-ASPECT / I-ASPECT.
    """
    words = text.split()
    bio = ["O"] * len(words)

    # Build character offset → word index map
    char_to_word = {}
    idx = 0
    for wi, word in enumerate(words):
        for ci in range(len(word)):
            char_to_word[idx + ci] = wi
        idx += len(word) + 1  # +1 for the space

    for mod in modifications:
        span = mod.get("span", "").strip()
        aspect = mod.get("aspect", "UNKNOWN")
        if not span or aspect not in VALID_ASPECTS:
            continue

        # Find span in text
        start = text.find(span)
        if start == -1:
            continue
        end = start + len(span) - 1

        # Map to words
        start_word = char_to_word.get(start)
        end_word   = char_to_word.get(end)
        if start_word is None or end_word is None:
            continue

        for wi in range(start_word, end_word + 1):
            if wi == start_word:
                bio[wi] = f"B-{aspect}"
            else:
                bio[wi] = f"I-{aspect}"

    return bio


def sep(title: str = "", width: int = 68):
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{'─' * 2} {title} {'─' * pad}")
    else:
        print("─" * width)


# ============================================================================
# DIAGNOSTIC 1 — source_comment distribution
# ============================================================================

def diagnostic_1_source_comment(silver_records: Dict[str, Dict]) -> Dict:
    """
    Count every distinct source_comment value across all modifications
    in the silver file.

    The evaluate_teacher.py script filters with:
        if src == position   (position defaults to "top" for gold records)

    If silver uses e.g. "top_comment" or "comment_0" instead of "top",
    ALL teacher predictions will be silently dropped → F1 = 0.

    Returns a summary dict.
    """
    sep("D1 · source_comment distribution in silver")

    all_src_values: Counter = Counter()
    threads_with_mods = 0
    threads_no_mods   = 0
    threads_total     = len(silver_records)

    for tid, rec in silver_records.items():
        label = get_silver_label(rec)
        mods  = label.get("modifications", [])
        if not mods:
            threads_no_mods += 1
            continue
        threads_with_mods += 1
        for mod in mods:
            src = mod.get("source_comment", "__MISSING__")
            all_src_values[src] += 1

    print(f"\n  Total silver records     : {threads_total:,}")
    print(f"  Threads with mods        : {threads_with_mods:,}")
    print(f"  Threads without mods     : {threads_no_mods:,}")
    print(f"\n  source_comment values found (modification-level counts):")

    if not all_src_values:
        print("  ⚠️  NO modifications found at all in silver file!")
    else:
        for val, count in all_src_values.most_common():
            marker = " ← MATCHES gold 'top' filter ✅" if val == "top" else ""
            print(f"    {repr(val):30s}  {count:6,}{marker}")

    # VERDICT
    top_count = all_src_values.get("top", 0)
    total_mods = sum(all_src_values.values())

    verdict = {}
    if total_mods == 0:
        verdict = {
            "status": "ERROR",
            "message": "Silver file has zero modifications. Labeling pipeline may have failed.",
        }
        print("\n  ❌ VERDICT: Silver has NO modifications. Labeling pipeline failed.")
    elif top_count == 0:
        non_top_vals = [v for v in all_src_values if v != "top"]
        verdict = {
            "status": "BUG",
            "message": (
                f"Silver uses {non_top_vals} — NOT 'top'. "
                "evaluate_teacher.py 'top' filter is discarding ALL predictions. "
                "F1=0.23 may be artificially low or zero due to this bug."
            ),
        }
        print(f"\n  ❌ VERDICT: 'top' is ABSENT from source_comment values.")
        print(f"     The evaluate_teacher filter `src == position` ('top') drops")
        print(f"     EVERY prediction. F1=0.23 is likely a filter bug.")
        print(f"     FIX: align source_comment values in silver to use 'top'")
        print(f"     OR update evaluate_teacher.py to match the actual values.")
    elif top_count < total_mods * 0.5:
        verdict = {
            "status": "PARTIAL_BUG",
            "message": (
                f"Only {top_count}/{total_mods} mods use 'top'. "
                "Most predictions are being silently dropped by the 'top' filter."
            ),
        }
        print(f"\n  ⚠️  VERDICT: Partial bug — only {top_count}/{total_mods} mods "
              f"have source_comment='top'.")
        print(f"     The rest are being silently dropped in evaluate_teacher.py.")
    else:
        verdict = {
            "status": "OK",
            "message": (
                f"{top_count}/{total_mods} mods use 'top'. "
                "Filter is working correctly. F1=0.23 is real."
            ),
        }
        print(f"\n  ✅ VERDICT: 'top' accounts for {top_count}/{total_mods} mods.")
        print(f"     The filter is correct. F1=0.23 is a real quality signal.")
        print(f"     → The teacher prompt needs rewriting.")

    return {
        "source_comment_counts": dict(all_src_values),
        "threads_total": threads_total,
        "threads_with_mods": threads_with_mods,
        "threads_without_mods": threads_no_mods,
        "verdict": verdict,
    }


# ============================================================================
# DIAGNOSTIC 2 — False Positives + Span Mismatches
# ============================================================================

def diagnostic_2_error_examples(
    silver_records: Dict[str, Dict],
    gold_records: Dict[str, Dict],
    n_fp: int = 10,
    n_mismatch: int = 10,
) -> Dict:
    """
    Collect two kinds of teacher errors against the gold set:

    1. FALSE POSITIVES (FP):
       Teacher says has_modification=True, gold says False.
       These are cases where the teacher hallucinates a modification.

    2. SPAN MISMATCHES:
       Both teacher and gold agree has_modification=True, but
       the teacher's extracted span does NOT match the gold span.
       Includes: wrong span text, wrong aspect, partial overlap.

    Returns dicts ready to be written to JSON and printed.
    """
    sep("D2 · Error examples for prompt rewriting")

    fp_examples: List[Dict]       = []
    mismatch_examples: List[Dict] = []

    matched_threads = 0
    skipped_threads = 0

    for tid, g_rec in gold_records.items():
        s_rec = silver_records.get(tid)
        if s_rec is None:
            skipped_threads += 1
            continue
        matched_threads += 1

        label      = get_silver_label(s_rec)
        g_has_mod  = g_rec.get("has_modification", False)
        s_has_mod  = label.get("has_modification", False)
        g_mods     = g_rec.get("gold_modifications", [])
        s_mods     = label.get("modifications", [])

        text = g_rec.get("comment_text", "")

        # ── FALSE POSITIVES ──────────────────────────────────────────────────
        if s_has_mod and not g_has_mod and len(fp_examples) < n_fp:
            # Extract what the teacher *thought* it saw
            teacher_spans = [
                {"span": m.get("span", ""), "aspect": m.get("aspect", ""), "source": m.get("source_comment", "")}
                for m in s_mods
            ]
            fp_examples.append({
                "thread_id"   : tid,
                "comment_text": text,
                "gold_label"  : "NO modification",
                "teacher_said": "HAS modification",
                "teacher_spans": teacher_spans,
                "note": "Teacher hallucinated a modification that does not exist.",
            })

        # ── SPAN MISMATCHES ──────────────────────────────────────────────────
        elif s_has_mod and g_has_mod and len(mismatch_examples) < n_mismatch:
            # Build sets of (span, aspect) for comparison
            gold_set    = {(m.get("span", "").strip(), m.get("aspect", "")) for m in g_mods}
            # Filter silver to 'top' source only (matching gold comment_position)
            silver_set  = {
                (m.get("span", "").strip(), m.get("aspect", ""))
                for m in s_mods
                if m.get("source_comment", "top") == g_rec.get("comment_position", "top")
            }

            # If the sets are identical, no mismatch — skip
            if gold_set == silver_set:
                continue

            only_in_gold   = list(gold_set - silver_set)   # FN at span level
            only_in_silver = list(silver_set - gold_set)   # FP at span level
            in_both        = list(gold_set & silver_set)

            if not only_in_gold and not only_in_silver:
                continue  # Perfect match at span level too, skip

            mismatch_examples.append({
                "thread_id"       : tid,
                "comment_text"    : text,
                "gold_spans"      : [{"span": s, "aspect": a} for s, a in sorted(gold_set)],
                "teacher_spans"   : [{"span": s, "aspect": a} for s, a in sorted(silver_set)],
                "correct_spans"   : [{"span": s, "aspect": a} for s, a in sorted(in_both)],
                "missed_by_teacher": [{"span": s, "aspect": a} for s, a in sorted(only_in_gold)],
                "hallucinated_by_teacher": [{"span": s, "aspect": a} for s, a in sorted(only_in_silver)],
                "note": "Span-level mismatch: teacher found mods but got boundaries or aspects wrong.",
            })

    # ── PRINT FALSE POSITIVES ────────────────────────────────────────────────
    print(f"\n  Matched gold↔silver threads : {matched_threads}")
    print(f"  Skipped (no silver match)   : {skipped_threads}")

    print(f"\n{'─'*68}")
    print(f"  FALSE POSITIVES ({len(fp_examples)} of up to {n_fp})")
    print(f"  Teacher said HAS MOD — gold says NO MOD")
    print(f"{'─'*68}")

    if not fp_examples:
        print("  ✅ No false positives found in the gold-silver overlap!")
    else:
        for i, ex in enumerate(fp_examples, 1):
            print(f"\n  FP #{i} — {ex['thread_id']}")
            print(f"  Text    : {ex['comment_text'][:180]}")
            for ts in ex["teacher_spans"]:
                print(f"  T-span  : [{ts['aspect']}] \"{ts['span']}\"  (source={ts['source']})")

    # ── PRINT SPAN MISMATCHES ────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"  SPAN MISMATCHES ({len(mismatch_examples)} of up to {n_mismatch})")
    print(f"  Both sides agree HAS MOD — but spans/aspects differ")
    print(f"{'─'*68}")

    if not mismatch_examples:
        print("  ✅ No span mismatches found! Perfect span alignment.")
    else:
        for i, ex in enumerate(mismatch_examples, 1):
            print(f"\n  MM #{i} — {ex['thread_id']}")
            print(f"  Text      : {ex['comment_text'][:180]}")
            if ex["missed_by_teacher"]:
                for s in ex["missed_by_teacher"]:
                    print(f"  MISSED    : [{s['aspect']}] \"{s['span']}\"   (gold has it, teacher didn't)")
            if ex["hallucinated_by_teacher"]:
                for s in ex["hallucinated_by_teacher"]:
                    print(f"  HALLUC.   : [{s['aspect']}] \"{s['span']}\"   (teacher added it, gold doesn't have it)")
            if ex["correct_spans"]:
                for s in ex["correct_spans"]:
                    print(f"  CORRECT   : [{s['aspect']}] \"{s['span']}\"")

    print(f"\n  ── Summary ──")
    print(f"  False positives collected  : {len(fp_examples)}")
    print(f"  Span mismatches collected  : {len(mismatch_examples)}")

    if len(fp_examples) < n_fp or len(mismatch_examples) < n_mismatch:
        print(f"\n  ℹ️  Note: Fewer examples than requested.")
        print(f"     This means the teacher is correct on most gold threads.")
        print(f"     If matched_threads is also small, the gold↔silver thread_id overlap")
        print(f"     is the bottleneck — check that silver covers the gold thread IDs.")

    return {
        "matched_threads": matched_threads,
        "skipped_threads": skipped_threads,
        "false_positives": fp_examples,
        "span_mismatches": mismatch_examples,
    }


# ============================================================================
# DAY 0 PRE-FLIGHT CHECKS (§11.1)
# ============================================================================

def preflight_checks(project_root: Path, skip_eval: bool) -> Dict:
    """
    Run the 5 Day-0 checks from MASTER_PLAN §11.1.
    All checks print PASS / FAIL with actionable messages.
    """
    sep("D0 · Day 0 Pre-flight Checks (§11.1)")
    results = {}

    # ── CHECK 1: src/utils/__init__.py ───────────────────────────────────────
    init_utils = project_root / "src" / "utils" / "__init__.py"
    if init_utils.exists():
        print(f"\n  [CHECK 1] src/utils/__init__.py         : ✅ EXISTS")
        results["utils_init"] = "PASS"
    else:
        print(f"\n  [CHECK 1] src/utils/__init__.py         : ❌ MISSING")
        print(f"            FIX: touch {init_utils}")
        results["utils_init"] = "FAIL"

    # ── CHECK 2: src/preprocessing/__init__.py has prepare_data_merged ───────
    init_pre = project_root / "src" / "preprocessing" / "__init__.py"
    if init_pre.exists():
        content = init_pre.read_text(encoding="utf-8", errors="replace")
        if "prepare_data_merged" in content:
            print(f"  [CHECK 2] preprocessing/__init__.py    : ✅ has prepare_data_merged")
            results["preprocessing_init"] = "PASS"
        else:
            print(f"  [CHECK 2] preprocessing/__init__.py    : ❌ missing 'prepare_data_merged'")
            print(f"            FIX: add 'from .prepare_data_merged import *' to {init_pre}")
            results["preprocessing_init"] = "FAIL"
    else:
        print(f"  [CHECK 2] preprocessing/__init__.py    : ❌ FILE MISSING")
        print(f"            FIX: touch {init_pre} and add the prepare_data_merged import")
        results["preprocessing_init"] = "FAIL"

    # ── CHECK 3: train_student.py mentions class.weights ─────────────────────
    train_student = project_root / "src" / "models" / "train_student.py"
    if train_student.exists():
        content = train_student.read_text(encoding="utf-8", errors="replace")
        import re
        matches = re.findall(r".{0,60}class.weights.{0,60}", content, re.IGNORECASE)
        if matches:
            print(f"  [CHECK 3] train_student.py class_weights : ✅ FOUND")
            for m in matches[:3]:
                print(f"            → {m.strip()}")
            results["train_student_class_weights"] = "PASS"
        else:
            print(f"  [CHECK 3] train_student.py class_weights : ❌ NO MATCH for 'class.weights'")
            print(f"            FIX: ensure class weights are loaded in train_student.py")
            results["train_student_class_weights"] = "FAIL"
    else:
        print(f"  [CHECK 3] train_student.py               : ❌ FILE NOT FOUND at {train_student}")
        results["train_student_class_weights"] = "FAIL"

    # ── CHECK 4: class_weights.py wiring ─────────────────────────────────────
    class_weights_py = project_root / "src" / "utils" / "class_weights.py"
    if class_weights_py.exists():
        content = class_weights_py.read_text(encoding="utf-8", errors="replace")
        has_load  = "load_class_weights" in content
        has_stats = "stats_merged"       in content
        markers = []
        if has_load:  markers.append("load_class_weights ✅")
        else:         markers.append("load_class_weights ❌")
        if has_stats: markers.append("stats_merged ✅")
        else:         markers.append("stats_merged ❌")
        status = "PASS" if (has_load or has_stats) else "FAIL"
        icon   = "✅" if status == "PASS" else "❌"
        print(f"  [CHECK 4] class_weights.py               : {icon} {', '.join(markers)}")
        if status == "FAIL":
            print(f"            FIX: add load_class_weights() or stats_merged reference to {class_weights_py}")
        results["class_weights_py"] = status
    else:
        print(f"  [CHECK 4] class_weights.py               : ❌ FILE NOT FOUND at {class_weights_py}")
        results["class_weights_py"] = "FAIL"

    # ── CHECK 5: Teacher eval re-run ─────────────────────────────────────────
    if skip_eval:
        print(f"  [CHECK 5] Teacher eval                   : ⏭  SKIPPED (--skip-eval)")
        results["teacher_eval"] = "SKIPPED"
    else:
        eval_script  = project_root / "scripts" / "evaluate_teacher.py"
        results_file = project_root / "results" / "teacher_upper_bound.json"

        if not eval_script.exists():
            print(f"  [CHECK 5] Teacher eval                   : ❌ Script not found at {eval_script}")
            results["teacher_eval"] = "FAIL"
        else:
            print(f"  [CHECK 5] Teacher eval                   : 🔄 Running (may take ~1 min)…")
            try:
                proc = subprocess.run(
                    [sys.executable, str(eval_script)],
                    cwd=str(project_root),
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if proc.returncode == 0:
                    # Try to read F1 from the results file
                    if results_file.exists():
                        with open(results_file) as f:
                            data = json.load(f)
                        f1 = data.get("entity_f1") or data.get("f1") or data.get("span_f1") or "?"
                        print(f"  [CHECK 5] Teacher eval                   : ✅ F1 = {f1}")
                    else:
                        print(f"  [CHECK 5] Teacher eval                   : ✅ Completed (results not at expected path)")
                        print(f"            Expected: {results_file}")
                    results["teacher_eval"] = "PASS"
                else:
                    print(f"  [CHECK 5] Teacher eval                   : ❌ Script exited with code {proc.returncode}")
                    if proc.stderr:
                        for line in proc.stderr.strip().splitlines()[-10:]:
                            print(f"            {line}")
                    results["teacher_eval"] = "FAIL"
            except subprocess.TimeoutExpired:
                print(f"  [CHECK 5] Teacher eval                   : ⚠️  TIMEOUT after 180s")
                results["teacher_eval"] = "TIMEOUT"
            except Exception as e:
                print(f"  [CHECK 5] Teacher eval                   : ❌ Error: {e}")
                results["teacher_eval"] = "FAIL"

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    sep("Pre-flight Summary")
    labels = {
        "utils_init"               : "src/utils/__init__.py",
        "preprocessing_init"       : "src/preprocessing/__init__.py has prepare_data_merged",
        "train_student_class_weights": "train_student.py class_weights wiring",
        "class_weights_py"         : "class_weights.py load_class_weights/stats_merged",
        "teacher_eval"             : "Teacher eval F1 sanity check",
    }
    all_pass = True
    for key, label in labels.items():
        status = results.get(key, "UNKNOWN")
        icon   = "✅" if status == "PASS" else ("⏭" if status == "SKIPPED" else "❌")
        if status not in ("PASS", "SKIPPED"):
            all_pass = False
        print(f"  {icon}  {label}")

    if all_pass:
        print("\n  🟢 All pre-flight checks passed. Safe to proceed to Track A Day 1.")
    else:
        print("\n  🔴 Some checks failed. Fix the ❌ items above before running training.")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="run_diagnostics.py — D1 + D2 + D0 in one pass")
    parser.add_argument(
        "--silver",
        default="data/silver_labels/teacher_output.jsonl",
        help="Path to silver labels JSONL (default: data/silver_labels/teacher_output.jsonl)",
    )
    parser.add_argument(
        "--gold",
        default="data/gold_validation/gold_final.jsonl",
        help="Path to gold annotations JSONL (default: data/gold_validation/gold_final.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="results/diagnostics_output.json",
        help="Where to write the JSON summary (default: results/diagnostics_output.json)",
    )
    parser.add_argument(
        "--n-fp",
        type=int,
        default=10,
        help="Max false-positive examples to collect (default: 10)",
    )
    parser.add_argument(
        "--n-mismatch",
        type=int,
        default=10,
        help="Max span-mismatch examples to collect (default: 10)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the teacher eval re-run in D0 (fast mode)",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Path to recipe_project/ root (default: current directory)",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    print("=" * 68)
    print("  DIAGNOSTIC RUNNER — D1 + D2 + D0 Pre-flight")
    print("=" * 68)

    # ── Resolve and validate paths ───────────────────────────────────────────
    silver_path = project_root / args.silver
    gold_path   = project_root / args.gold
    output_path = project_root / args.output

    print(f"\n  Project root : {project_root}")
    print(f"  Silver file  : {silver_path}")
    print(f"  Gold file    : {gold_path}")
    print(f"  Output       : {output_path}")

    for label, p in [("Silver", silver_path), ("Gold", gold_path)]:
        if not p.exists():
            print(f"\n  ❌ {label} file not found: {p}")
            print(f"     Adjust --silver / --gold flags or run from inside recipe_project/")
            sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\n  Loading files…")
    silver_records = load_silver_indexed(str(silver_path))
    gold_records   = load_gold_indexed(str(gold_path))
    print(f"  Silver threads loaded : {len(silver_records):,}")
    print(f"  Gold threads loaded   : {len(gold_records):,}")

    # ── Run diagnostics ──────────────────────────────────────────────────────
    d1_results = diagnostic_1_source_comment(silver_records)

    d2_results = diagnostic_2_error_examples(
        silver_records,
        gold_records,
        n_fp=args.n_fp,
        n_mismatch=args.n_mismatch,
    )

    d0_results = preflight_checks(project_root, skip_eval=args.skip_eval)

    # ── Write JSON output ────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        "diagnostic_1_source_comment": d1_results,
        "diagnostic_2_error_examples": d2_results,
        "preflight_checks": d0_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    sep("Done")
    print(f"\n  Full results written to: {output_path}")
    print(f"\n  Next steps depending on D1 verdict:")
    print(f"    OK     → F1=0.23 is real. Rewrite SYSTEM_PROMPT using D2 examples.")
    print(f"    BUG    → Fix source_comment values in silver, re-run evaluate_teacher.py")
    print(f"    PARTIAL→ Fix evaluate_teacher.py filter to accept all relevant source values.")
    print()


if __name__ == "__main__":
    main()
