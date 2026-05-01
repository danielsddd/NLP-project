#!/usr/bin/env python3
"""
=============================================================================
validate_silver_spans.py — Post-3-Pass Silver Label Sanity Audit
=============================================================================
Run this AFTER all 3 passes + --finalize are complete, BEFORE calling
prepare_data_merged.py. It validates both the silver label file and the
underlying span quality.

Usage:
    python scripts/local/validate_silver_spans.py \\
        --input data/silver_labels/teacher_output_v2.jsonl \\
        --check-all-passes

    # Also works on the enriched file:
    python scripts/local/validate_silver_spans.py \\
        --input data/silver_labels/threads_positives_focus_labeled_v2.jsonl \\
        --check-all-passes

    # Save a JSON report:
    python scripts/local/validate_silver_spans.py \\
        --input data/silver_labels/teacher_output_v2.jsonl \\
        --check-all-passes \\
        --report results/validate_silver_report.json

Checks:
    §A  File integrity          — count, JSON validity, required fields
    §B  Pass coverage           — P1/P2/P3/final present for every record
    §C  Per-pass label stats    — has_mod rate, aspect dist, parse failures
    §D  Per-pass span validity  — hallucination check (span ∉ thread text)
    §E  Inter-pass agreement    — agreement_1v2 / 1v3 / 2v3 distributions
    §F  Final label quality     — vote_method dist, needs_review, flips
    §G  Summary                 — PASS / WARN / FAIL gate for preprocessing

Exit code:
    0  = all required checks pass (warnings are OK)
    1  = at least one FAIL (do not proceed to preprocessing)
=============================================================================
"""

import json
import re
import sys
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — must match src/preprocessing/prepare_data.py
# ─────────────────────────────────────────────────────────────────────────────
VALID_ASPECTS = {"SUBSTITUTION", "QUANTITY", "TECHNIQUE", "ADDITION"}

# Agreement levels used by compute_pairwise_agreement() in generate_labels.py
VALID_AGREEMENT_LEVELS = {"full_agreement", "partial_agreement", "no_agreement"}

# vote_method values set by compute_majority_vote() in generate_labels.py
VALID_VOTE_METHODS = {"unanimous", "majority", "no_majority", "gemini_unanimous"}

# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL COLORS
# ─────────────────────────────────────────────────────────────────────────────
RED    = "\033[0;31m"
GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN   = "\033[0;36m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
NC     = "\033[0m"


def _ok(msg):
    print(f"  {GREEN}[OK  ]{NC} {msg}")

def _warn(msg):
    print(f"  {YELLOW}[WARN]{NC} {msg}")

def _fail(msg):
    print(f"  {RED}[FAIL]{NC} {msg}")

def _info(msg):
    print(f"  {DIM}      {msg}{NC}")

def _section(title):
    width = 70
    print(f"\n{CYAN}{BOLD}{'─' * width}{NC}")
    print(f"{CYAN}{BOLD}  {title}{NC}")
    print(f"{CYAN}{BOLD}{'─' * width}{NC}")


# ─────────────────────────────────────────────────────────────────────────────
# SPAN HELPERS  (mirrors generate_labels._normalize_for_span_check exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """
    Mirror of generate_labels._normalize_for_span_check.
    Collapse whitespace + strip trailing/leading punct that LLMs add.
    Hebrew has no case, so we do NOT lowercase.
    """
    if not s:
        return ""
    s = " ".join(s.split())
    s = s.strip(" \t\n\r.,!?;:\"'`׳״()[]{}")
    return s


def _span_in_thread(span: str, thread_text: str) -> bool:
    """Return True iff span (after normalization) is a substring of thread_text."""
    span_n = _normalize(span)
    if not span_n:
        return False          # empty after norm → hallucinated
    text_n = _normalize(thread_text)
    return span_n in text_n


def _build_thread_text(rec: dict) -> str:
    """
    Reconstruct the full thread text a teacher saw.
    Tries the pre-built fields first (top_comment_text / replies_texts),
    then falls back to the nested top_comment / replies objects.
    """
    # Pre-built text fields (set by generate_labels during labeling)
    top = rec.get("top_comment_text", "") or ""
    replies_texts = rec.get("replies_texts", []) or []
    if top:
        return top + "\n" + "\n".join(replies_texts)

    # Fallback: reconstruct from raw structure
    top_obj = rec.get("top_comment", {}) or {}
    top = top_obj.get("text", "") or ""
    replies = rec.get("replies", []) or []
    reply_texts = [r.get("text", "") or "" for r in replies if isinstance(r, dict)]
    return top + "\n" + "\n".join(reply_texts)


# ─────────────────────────────────────────────────────────────────────────────
# COUNTER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pct(num: int, den: int, decimals: int = 1) -> str:
    if den == 0:
        return "—"
    return f"{100.0 * num / den:.{decimals}f}%"


def _counter_table(counter: Counter, total: int, indent: int = 6) -> None:
    pad = " " * indent
    for key, cnt in counter.most_common():
        print(f"{pad}{str(key):<25s}  {cnt:6d}  ({_pct(cnt, total)})")


# ─────────────────────────────────────────────────────────────────────────────
# RESULT ACCUMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class AuditResult:
    def __init__(self):
        self.passes  = []   # list of (section, message)
        self.warns   = []
        self.fails   = []
        self.data    = {}   # for JSON report

    def ok(self, section: str, msg: str):
        _ok(msg)
        self.passes.append((section, msg))

    def warn(self, section: str, msg: str):
        _warn(msg)
        self.warns.append((section, msg))

    def fail(self, section: str, msg: str):
        _fail(msg)
        self.fails.append((section, msg))

    @property
    def exit_code(self) -> int:
        return 1 if self.fails else 0


# =============================================================================
# §A  FILE INTEGRITY
# =============================================================================

def check_file_integrity(path: Path, result: AuditResult) -> list:
    """Load records, check JSON validity, check required top-level fields."""
    _section("§A  FILE INTEGRITY")

    records = []
    bad_json = 0
    empty_lines = 0

    if not path.exists():
        result.fail("§A", f"File not found: {path}")
        return records

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                empty_lines += 1
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as e:
                bad_json += 1
                if bad_json <= 3:
                    _warn(f"  Line {lineno}: JSON parse error — {e}")

    result.data["path"] = str(path)
    result.data["total_records"] = len(records)

    if bad_json > 0:
        result.fail("§A", f"{bad_json} lines could not be parsed as JSON")
    if empty_lines > 0:
        _info(f"Skipped {empty_lines} blank lines")

    if len(records) == 0:
        result.fail("§A", "File is empty or contains no valid JSON lines")
        return records

    result.ok("§A", f"Loaded {len(records):,} records  ({bad_json} bad lines)")

    # ── Required top-level fields ────────────────────────────────────────────
    required_fields = ["thread_id"]
    missing_counts = Counter()
    empty_top_comment = 0
    dup_tids = Counter()

    for rec in records:
        tid = rec.get("thread_id", "")
        if tid:
            dup_tids[tid] += 1
        for f in required_fields:
            if not rec.get(f):
                missing_counts[f] += 1
        if not _build_thread_text(rec).strip():
            empty_top_comment += 1

    for field, cnt in missing_counts.items():
        result.fail("§A", f"Missing '{field}' in {cnt:,} records")

    dups = {tid: cnt for tid, cnt in dup_tids.items() if cnt > 1}
    if dups:
        result.fail("§A", f"{len(dups):,} duplicate thread_ids  "
                          f"(e.g. {next(iter(dups))} appears {next(iter(dups.values()))}×)")
    else:
        result.ok("§A", "No duplicate thread_ids")

    if empty_top_comment > 0:
        result.warn("§A", f"{empty_top_comment:,} records have empty thread text "
                          f"(top_comment_text + replies_texts both empty)")
    else:
        result.ok("§A", "All records have non-empty thread text")

    return records


# =============================================================================
# §B  PASS COVERAGE
# =============================================================================

def check_pass_coverage(records: list, check_all: bool, result: AuditResult) -> dict:
    """Verify P1/P2/P3/final coverage across every record."""
    _section("§B  PASS COVERAGE")

    n = len(records)
    has_p1     = sum(1 for r in records if r.get("teacher_output")        is not None)
    has_p2     = sum(1 for r in records if r.get("second_teacher_output") is not None)
    has_p3     = sum(1 for r in records if r.get("third_teacher_output")  is not None)
    p3_skipped = sum(1 for r in records if r.get("pass3_skipped"))
    has_final  = sum(1 for r in records if r.get("final_label")           is not None)

    # Pass 3 effective coverage = labeled + skipped (unanimous)
    p3_effective = has_p3 + p3_skipped

    coverage = {
        "pass1": has_p1,
        "pass2": has_p2,
        "pass3_labeled": has_p3,
        "pass3_skipped_unanimous": p3_skipped,
        "pass3_effective": p3_effective,
        "finalized": has_final,
        "total": n,
    }
    result.data["coverage"] = coverage

    _info(f"Total records:          {n:>7,}")
    _info(f"Pass 1 (teacher_output):     {has_p1:>7,}  ({_pct(has_p1, n)})")
    _info(f"Pass 2 (2nd_teacher_output): {has_p2:>7,}  ({_pct(has_p2, n)})")
    _info(f"Pass 3 (Qwen labeled):       {has_p3:>7,}  ({_pct(has_p3, n)})")
    _info(f"Pass 3 (skipped — unanimous):{p3_skipped:>7,}  ({_pct(p3_skipped, n)})")
    _info(f"Pass 3 effective total:      {p3_effective:>7,}  ({_pct(p3_effective, n)})")
    _info(f"Finalized (final_label):     {has_final:>7,}  ({_pct(has_final, n)})")

    # Pass 1 must be 100%
    if has_p1 < n:
        result.fail("§B", f"Pass 1 incomplete: {n - has_p1:,} records missing "
                          f"teacher_output — labeling did not finish")
    else:
        result.ok("§B", "Pass 1 complete (100%)")

    if check_all:
        # Pass 2 must be 100%
        if has_p2 < n:
            result.fail("§B", f"Pass 2 incomplete: {n - has_p2:,} records missing "
                              f"second_teacher_output")
        else:
            result.ok("§B", "Pass 2 complete (100%)")

        # Pass 3 effective (labeled + skipped) must be 100%
        if p3_effective < n:
            result.fail("§B", f"Pass 3 incomplete: {n - p3_effective:,} records have "
                              f"neither third_teacher_output nor pass3_skipped=True")
        else:
            result.ok("§B", f"Pass 3 complete (100%): "
                            f"{has_p3:,} Qwen-labeled + {p3_skipped:,} gemini_unanimous")

        # Finalized must be 100%
        if has_final < n:
            result.fail("§B", f"--finalize not run: {n - has_final:,} records "
                              f"missing final_label — run: "
                              f"python -m src.teacher_labeling.generate_labels "
                              f"--finalize -o {result.data.get('path', '<file>')}")
        else:
            result.ok("§B", "All records finalized (final_label present)")

    return coverage


# =============================================================================
# §C  PER-PASS LABEL STATS
# =============================================================================

def _check_one_pass_labels(
    records: list,
    pass_field: str,
    pass_name: str,
    result: AuditResult,
) -> dict:
    """Check has_mod rate, aspect distribution, and parse failures for one pass."""
    has_mod_count  = 0
    no_mod_count   = 0
    no_label_count = 0
    aspect_counter = Counter()
    bad_aspect     = 0
    empty_span     = 0
    parse_fail     = 0
    missing_field  = 0

    for rec in records:
        lbl = rec.get(pass_field)
        if lbl is None:
            no_label_count += 1
            continue
        if rec.get("parse_failure") and pass_field == "teacher_output":
            parse_fail += 1

        has_mod = lbl.get("has_modification")
        if has_mod is True:
            has_mod_count += 1
        elif has_mod is False:
            no_mod_count += 1
        else:
            no_label_count += 1

        for mod in lbl.get("modifications", []) or []:
            asp = mod.get("aspect", "UNKNOWN")
            aspect_counter[asp] += 1
            if asp not in VALID_ASPECTS:
                bad_aspect += 1
            if not mod.get("span", "").strip():
                empty_span += 1

    total_labeled = has_mod_count + no_mod_count
    pos_rate = 100.0 * has_mod_count / total_labeled if total_labeled > 0 else 0.0

    print(f"\n  {BOLD}{pass_name}{NC}")
    _info(f"has_modification=True:  {has_mod_count:>7,}  ({_pct(has_mod_count, total_labeled)})")
    _info(f"has_modification=False: {no_mod_count:>7,}  ({_pct(no_mod_count, total_labeled)})")
    if no_label_count:
        _info(f"unlabeled / None:       {no_label_count:>7,}")

    if aspect_counter:
        _info("Aspect distribution:")
        for asp, cnt in aspect_counter.most_common():
            _info(f"  {asp:<20s} {cnt:>6,}")

    # Gate: positive rate sanity
    if total_labeled > 0:
        if pos_rate < 3.0:
            result.warn("§C", f"{pass_name}: positive rate {pos_rate:.1f}% is suspiciously low")
        elif pos_rate > 80.0:
            result.warn("§C", f"{pass_name}: positive rate {pos_rate:.1f}% is suspiciously high")

    if bad_aspect > 0:
        result.warn("§C", f"{pass_name}: {bad_aspect:,} mods have unrecognized aspect "
                          f"(valid: {', '.join(sorted(VALID_ASPECTS))})")
    if empty_span > 0:
        result.warn("§C", f"{pass_name}: {empty_span:,} mods have empty span text")
    if parse_fail > 0:
        result.warn("§C", f"{pass_name}: {parse_fail:,} records flagged parse_failure=True")

    return {
        "has_mod": has_mod_count,
        "no_mod": no_mod_count,
        "unlabeled": no_label_count,
        "positive_rate_pct": round(pos_rate, 2),
        "aspect_distribution": dict(aspect_counter),
        "bad_aspect_count": bad_aspect,
        "empty_span_count": empty_span,
    }


def check_per_pass_label_stats(
    records: list, check_all: bool, result: AuditResult
) -> dict:
    _section("§C  PER-PASS LABEL STATS")

    stats = {}
    passes_to_check = [
        ("teacher_output",        "Pass 1 (Gemini primary)"),
        ("second_teacher_output", "Pass 2 (Gemini temp=0.3)"),
    ]
    if check_all:
        passes_to_check.append(("third_teacher_output", "Pass 3 (Qwen-235B)"))
        passes_to_check.append(("final_label",          "Final (majority vote)"))

    for field, name in passes_to_check:
        s = _check_one_pass_labels(records, field, name, result)
        stats[field] = s
        # If total_labeled == 0 for a pass that should exist, flag it
        if s["has_mod"] + s["no_mod"] == 0 and s["unlabeled"] > 0:
            result.warn("§C", f"{name}: no labeled records found — "
                              f"pass may not have run yet")

    result.data["per_pass_label_stats"] = stats

    # Cross-pass consistency: positive rate should be within ±20pp between passes
    rates = {}
    for field, name in passes_to_check:
        if stats[field]["has_mod"] + stats[field]["no_mod"] > 0:
            rates[name] = stats[field]["positive_rate_pct"]

    if len(rates) >= 2:
        rate_values = list(rates.values())
        spread = max(rate_values) - min(rate_values)
        if spread > 20:
            result.warn("§C",
                f"Positive rate spread across passes is {spread:.1f}pp — "
                f"large divergence may indicate a prompt regression between passes")
        else:
            result.ok("§C",
                f"Positive rate is consistent across passes "
                f"(spread = {spread:.1f}pp ≤ 20pp)")

    return stats


# =============================================================================
# §D  PER-PASS SPAN VALIDITY  (hallucination detection)
# =============================================================================

def _check_one_pass_spans(
    records: list,
    pass_field: str,
    pass_name: str,
    result: AuditResult,
    sample_limit: int = 5,
) -> dict:
    """
    For every modification in pass_field, check that span appears as a
    substring in the actual thread text the teacher received.
    Mirrors _span_in_thread() from generate_labels.py exactly.
    """
    total_spans  = 0
    found_spans  = 0
    hallucinated = 0
    empty_span   = 0
    examples     = []      # sample hallucinated spans for display
    aspects_hall = Counter()

    for rec in records:
        lbl = rec.get(pass_field)
        if lbl is None:
            continue
        if not lbl.get("has_modification"):
            continue

        thread_text = _build_thread_text(rec)
        tid = rec.get("thread_id", "?")

        for mod in lbl.get("modifications", []) or []:
            span = mod.get("span", "") or ""
            asp  = mod.get("aspect", "UNKNOWN")
            total_spans += 1

            if not span.strip():
                empty_span += 1
                hallucinated += 1
                aspects_hall[asp] += 1
                continue

            if _span_in_thread(span, thread_text):
                found_spans += 1
            else:
                hallucinated += 1
                aspects_hall[asp] += 1
                if len(examples) < sample_limit:
                    examples.append({
                        "thread_id": tid,
                        "span": span[:80],
                        "aspect": asp,
                    })

    hall_rate = 100.0 * hallucinated / total_spans if total_spans > 0 else 0.0

    print(f"\n  {BOLD}{pass_name}{NC}")
    _info(f"Total spans checked:  {total_spans:>7,}")
    _info(f"Found in thread text: {found_spans:>7,}  ({_pct(found_spans, total_spans)})")
    _info(f"Hallucinated:         {hallucinated:>7,}  ({_pct(hallucinated, total_spans)})")
    if empty_span:
        _info(f"  ↳ of which empty span: {empty_span:>5,}")

    if hallucinated > 0 and aspects_hall:
        _info(f"Hallucination by aspect:")
        for asp, cnt in aspects_hall.most_common():
            _info(f"  {asp:<20s} {cnt:>5,}  ({_pct(cnt, hallucinated)})")

    if examples:
        _info(f"Sample hallucinated spans:")
        for ex in examples:
            _info(f"  [{ex['aspect']:<14s}] '{ex['span']}'  (thread {ex['thread_id'][:30]})")

    # Gate
    if hall_rate > 15.0:
        result.fail("§D",
            f"{pass_name}: hallucination rate {hall_rate:.1f}% > 15% threshold — "
            f"check prompt or model (these spans will be dropped by --finalize)")
    elif hall_rate > 5.0:
        result.warn("§D",
            f"{pass_name}: hallucination rate {hall_rate:.1f}% > 5% — "
            f"acceptable but worth reviewing the sample above")
    elif total_spans == 0:
        _info("No spans to validate (all has_modification=False?)")
    else:
        result.ok("§D",
            f"{pass_name}: hallucination rate {hall_rate:.1f}% ≤ 5% ✓")

    return {
        "total_spans": total_spans,
        "found": found_spans,
        "hallucinated": hallucinated,
        "hallucination_rate_pct": round(hall_rate, 2),
        "by_aspect": dict(aspects_hall),
        "sample_hallucinated": examples,
    }


def check_per_pass_span_validity(
    records: list, check_all: bool, result: AuditResult
) -> dict:
    _section("§D  PER-PASS SPAN VALIDITY  (hallucination check)")
    print(f"  {DIM}Span must appear as a substring of top_comment_text + replies_texts.{NC}")
    print(f"  {DIM}Normalization: collapse whitespace, strip leading/trailing punctuation.{NC}")

    passes_to_check = [
        ("teacher_output",        "Pass 1 (Gemini primary)"),
        ("second_teacher_output", "Pass 2 (Gemini temp=0.3)"),
    ]
    if check_all:
        passes_to_check.append(("third_teacher_output", "Pass 3 (Qwen-235B)"))
        passes_to_check.append(("final_label",          "Final label (post-finalize)"))

    span_stats = {}
    for field, name in passes_to_check:
        s = _check_one_pass_spans(records, field, name, result)
        span_stats[field] = s

    result.data["per_pass_span_validity"] = span_stats

    # Also check span_validation block written by --finalize
    if check_all:
        print(f"\n  {BOLD}Span validation blocks (written by --finalize){NC}")
        has_sv = sum(1 for r in records if r.get("span_validation") is not None)
        total_dropped  = sum(
            (r.get("span_validation") or {}).get("dropped", 0) for r in records
        )
        total_flipped  = sum(
            1 for r in records
            if (r.get("span_validation") or {}).get("flipped_to_no_mod")
        )
        _info(f"Records with span_validation block: {has_sv:,} / {len(records):,}")
        _info(f"Total hallucinated spans dropped:   {total_dropped:,}")
        _info(f"Threads flipped to no_mod:          {total_flipped:,}")
        if has_sv < len(records):
            result.warn("§D",
                f"span_validation block missing on {len(records) - has_sv:,} records — "
                f"re-run --finalize to regenerate")
        span_stats["finalize_block"] = {
            "records_with_block": has_sv,
            "total_dropped": total_dropped,
            "threads_flipped": total_flipped,
        }

    return span_stats


# =============================================================================
# §E  INTER-PASS AGREEMENT
# =============================================================================

def check_agreement(records: list, check_all: bool, result: AuditResult) -> dict:
    _section("§E  INTER-PASS AGREEMENT")

    pairs = [
        ("agreement_1v2", "Pass1 vs Pass2"),
    ]
    if check_all:
        pairs += [
            ("agreement_1v3", "Pass1 vs Pass3 (Qwen subset only)"),
            ("agreement_2v3", "Pass2 vs Pass3 (Qwen subset only)"),
        ]

    agr_stats = {}
    for field, name in pairs:
        counter = Counter()
        for rec in records:
            val = rec.get(field)
            if val:
                counter[val] += 1
        total = sum(counter.values())
        agr_stats[field] = {"name": name, "counts": dict(counter), "total": total}

        print(f"\n  {BOLD}{name}{NC}  (n={total:,})")
        if total == 0:
            _info("No agreement values found for this pair")
            if check_all:
                result.warn("§E", f"{name}: no agreement values — "
                                  f"pass may not have completed or field not written")
            continue

        _counter_table(counter, total)

        # Check for unrecognized agreement levels
        bad = {k for k in counter if k not in VALID_AGREEMENT_LEVELS}
        if bad:
            result.warn("§E", f"{name}: unrecognized agreement levels: {bad}")

        full = counter.get("full_agreement", 0)
        full_pct = 100.0 * full / total
        if full_pct < 40.0:
            result.warn("§E",
                f"{name}: full_agreement = {full_pct:.1f}% < 40% — "
                f"label quality may be low; consider reviewing prompt")
        else:
            result.ok("§E", f"{name}: full_agreement = {full_pct:.1f}%")

    result.data["agreement"] = agr_stats
    return agr_stats


# =============================================================================
# §F  FINAL LABEL QUALITY
# =============================================================================

def check_final_label_quality(records: list, result: AuditResult) -> dict:
    _section("§F  FINAL LABEL QUALITY")

    finalized = [r for r in records if r.get("final_label") is not None]
    n = len(finalized)
    total = len(records)

    if n == 0:
        result.warn("§F", "No finalized records — run --finalize before preprocessing")
        return {}

    _info(f"Finalized records: {n:,} / {total:,}")

    # ── Vote method distribution ─────────────────────────────────────────────
    vote_counter = Counter()
    for rec in finalized:
        vm = rec.get("vote_method", "MISSING")
        vote_counter[vm] += 1

    print(f"\n  {BOLD}Vote method distribution{NC}")
    _counter_table(vote_counter, n)

    bad_vm = {k for k in vote_counter if k not in VALID_VOTE_METHODS and k != "MISSING"}
    if bad_vm:
        result.warn("§F", f"Unrecognized vote_method values: {bad_vm}")
    if vote_counter.get("MISSING", 0) > 0:
        result.warn("§F", f"{vote_counter['MISSING']:,} finalized records are missing "
                          f"vote_method field")

    # Gemini unanimous rate (expected ~30-70% for tiebreaker mode)
    g_unani = vote_counter.get("gemini_unanimous", 0)
    _info(f"\ngemini_unanimous = {g_unani:,}  ({_pct(g_unani, n)}) "
          f"— records where Pass1≡Pass2 so Pass3 was skipped")

    # ── needs_review ─────────────────────────────────────────────────────────
    needs_review = sum(1 for r in finalized if r.get("needs_review"))
    _info(f"needs_review=True: {needs_review:,}  ({_pct(needs_review, n)})")
    if needs_review > 0:
        result.warn("§F",
            f"{needs_review:,} records flagged needs_review (3-way disagreement) — "
            f"inspect with: python -m src.teacher_labeling.generate_labels "
            f"--export-review -o <file>")

    # ── parse failures ───────────────────────────────────────────────────────
    parse_fail = sum(1 for r in records if r.get("parse_failure"))
    _info(f"parse_failure=True: {parse_fail:,}  ({_pct(parse_fail, total)})")
    if parse_fail > 0:
        result.warn("§F", f"{parse_fail:,} records have parse_failure — "
                          f"LLM output could not be parsed; review these before training")

    # ── Final has_mod rate ───────────────────────────────────────────────────
    final_has_mod = sum(
        1 for r in finalized
        if (r.get("final_label") or {}).get("has_modification")
    )
    final_no_mod = n - final_has_mod
    pos_pct = 100.0 * final_has_mod / n if n > 0 else 0.0

    print(f"\n  {BOLD}Final label has_modification{NC}")
    _info(f"has_modification=True:   {final_has_mod:>7,}  ({_pct(final_has_mod, n)})")
    _info(f"has_modification=False:  {final_no_mod:>7,}  ({_pct(final_no_mod, n)})")

    if pos_pct < 3.0:
        result.fail("§F",
            f"Final positive rate {pos_pct:.1f}% is critically low — "
            f"training signal will be insufficient")
    elif pos_pct < 8.0:
        result.warn("§F",
            f"Final positive rate {pos_pct:.1f}% is low — "
            f"expect severe class imbalance; use focal loss or class weighting")
    else:
        result.ok("§F", f"Final positive rate {pos_pct:.1f}%")

    # ── Aspect distribution in final_label ──────────────────────────────────
    final_aspects = Counter()
    final_empty_span = 0
    for rec in finalized:
        lbl = rec.get("final_label") or {}
        for mod in lbl.get("modifications", []) or []:
            final_aspects[mod.get("aspect", "UNKNOWN")] += 1
            if not mod.get("span", "").strip():
                final_empty_span += 1

    if final_aspects:
        print(f"\n  {BOLD}Final label aspect distribution{NC}")
        _counter_table(final_aspects, sum(final_aspects.values()))

    if final_empty_span > 0:
        result.warn("§F",
            f"{final_empty_span:,} final modifications have empty span — "
            f"these will be treated as hallucinated during preprocessing")

    stats = {
        "finalized": n,
        "has_mod": final_has_mod,
        "no_mod": final_no_mod,
        "positive_rate_pct": round(pos_pct, 2),
        "vote_method_dist": dict(vote_counter),
        "needs_review": needs_review,
        "parse_failures": parse_fail,
        "final_aspect_dist": dict(final_aspects),
    }
    result.data["final_label_quality"] = stats
    return stats


# =============================================================================
# §G  SUMMARY
# =============================================================================

def print_summary(result: AuditResult, input_path: Path) -> None:
    _section("§G  SUMMARY")

    n_pass = len(result.passes)
    n_warn = len(result.warns)
    n_fail = len(result.fails)

    print(f"  File:      {input_path}")
    print(f"  Run at:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"  {GREEN}{BOLD}PASS{NC}  {n_pass:3d}")
    print(f"  {YELLOW}{BOLD}WARN{NC}  {n_warn:3d}")
    print(f"  {RED}{BOLD}FAIL{NC}  {n_fail:3d}")
    print()

    if n_fail > 0:
        print(f"  {RED}{BOLD}FAILURES (must fix before preprocessing):{NC}")
        for section, msg in result.fails:
            print(f"    [{section}] {msg}")
        print()

    if n_warn > 0:
        print(f"  {YELLOW}Warnings:{NC}")
        for section, msg in result.warns:
            print(f"    [{section}] {msg}")
        print()

    if n_fail == 0:
        print(f"  {GREEN}{BOLD}✅  ALL CHECKS PASSED — safe to run prepare_data_merged.py{NC}")
    else:
        print(f"  {RED}{BOLD}❌  AUDIT FAILED — fix FAILs before preprocessing{NC}")
        print(f"       Do NOT run prepare_data_merged.py with a broken silver file.")

    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-3-pass silver label sanity audit for recipe modification extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        default="data/silver_labels/teacher_output_v2.jsonl",
        help="Path to silver label JSONL file (default: data/silver_labels/teacher_output_v2.jsonl)",
    )
    parser.add_argument(
        "--check-all-passes",
        action="store_true",
        default=False,
        help="Also validate Pass 2, Pass 3, and final_label (use after all 3 passes are done)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path to write a JSON audit report (e.g. results/validate_silver_report.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    print(f"\n{BOLD}{'=' * 70}{NC}")
    print(f"{BOLD}  SILVER LABEL SANITY AUDIT{NC}")
    print(f"{BOLD}{'=' * 70}{NC}")
    print(f"  Input:          {input_path}")
    print(f"  check-all-passes: {args.check_all_passes}")
    print(f"  Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{BOLD}{'=' * 70}{NC}")

    result = AuditResult()

    # ── Run all sections ─────────────────────────────────────────────────────
    records = check_file_integrity(input_path, result)

    if not records:
        print_summary(result, input_path)
        sys.exit(result.exit_code)

    check_pass_coverage(records, args.check_all_passes, result)
    check_per_pass_label_stats(records, args.check_all_passes, result)
    check_per_pass_span_validity(records, args.check_all_passes, result)
    check_agreement(records, args.check_all_passes, result)
    check_final_label_quality(records, result)

    # ── Summary ──────────────────────────────────────────────────────────────
    print_summary(result, input_path)

    # ── Optional JSON report ─────────────────────────────────────────────────
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "meta": {
                "input_file": str(input_path),
                "check_all_passes": args.check_all_passes,
                "timestamp": datetime.now().isoformat(),
                "exit_code": result.exit_code,
            },
            "summary": {
                "pass": len(result.passes),
                "warn": len(result.warns),
                "fail": len(result.fails),
            },
            "failures": [{"section": s, "message": m} for s, m in result.fails],
            "warnings": [{"section": s, "message": m} for s, m in result.warns],
            "data": result.data,
        }
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"  JSON report saved → {report_path}\n")

    sys.exit(result.exit_code)


if __name__ == "__main__":
    main()