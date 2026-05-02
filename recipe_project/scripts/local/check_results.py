#!/usr/bin/env python3
"""
check_results.py  —  Quick F1 / status scanner for all SLURM jobs.

Run from recipe_project root:
    python scripts/local/check_results.py

What it does:
  1. Scans results/<model>/<variant>/gold|silver/evaluation_results.json
  2. Scans models/checkpoints/<model>/<variant>/training_summary.json
  3. Scans SLURM .out log files for inline F1 prints
  4. Shows a ranked summary table + flags issues

Usage:
    python scripts/local/check_results.py [--log-dir PATH] [--results-dir PATH]
                                          [--ckpt-dir PATH] [--top N]
                                          [--model FILTER] [--verbose]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

# ─── ANSI colours ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ─── Known job-id → friendly label mapping (update with each squeue dump) ────
KNOWN_JOBS = {
    "305156": "Large+CRF P2 (dlarge_c)",
    "305157": "Large+CRF P2 dup? (dlarge_c)",
    "305158": "P10v3 fixed weights (dict_P10)",
    "305159": "P10v4 focal (dict_P10)",
    "305160": "Large+CRF P2 dup (dlarge_c)",
    "305161": "P4v2 fixed weights (dict_P4v)",
    "305162": "P5v2 fixed weights (dict_P5v)",
    "305163": "Large+CRF P10 IO+threads (dlarge_c)",
    "305164": "P4v3 thread+focal (dict_P4v)",
    "305170": "Large+CRF P10v4 focal (dlrg_P10) [PD]",
    "305171": "P10v3 lr=1e-5 (dict_P10) [PD]",
    "305172": "Large+CRF P4 BIO+threads (dlrg_P4_) [PD]",
    "305173": "P10v3 seed=123 [PD]",
    "305174": "P10v3 seed=2026 [PD]",
    "305175": "P10v3 seed=7777 [PD]",
    "305176": "Phase 7 baselines [PD]",
    "305177": "Large+CRF A1a no weights [PD]",
    "305178": "Large+CRF A1c weights only [PD]",
    "305179": "Large+CRF+IO+threads lr=3e-5 [PD]",
}

# ─── Regex patterns for inline F1 in .out logs ───────────────────────────────
F1_PATTERNS = [
    # evaluation_results.json printed inline: "f1": 0.4321
    re.compile(r'"f1"\s*:\s*([0-9]+\.[0-9]+)', re.IGNORECASE),
    # seqeval output: "F1 : 0.4321" or "f1-score  0.4321"
    re.compile(r'\bF1[\s\-_:]+([0-9]+\.[0-9]+)', re.IGNORECASE),
    re.compile(r'f1[_\-]?score[\s:=]+([0-9]+\.[0-9]+)', re.IGNORECASE),
    # span_f1 / micro_f1 / macro_f1
    re.compile(r'(?:span|micro|macro)?_?f1[\s:=]+([0-9]+\.[0-9]+)', re.IGNORECASE),
    # "Overall F1: 0.43"
    re.compile(r'[Oo]verall[\s_]F1[\s:=]+([0-9]+\.[0-9]+)'),
    # training_summary style: "eval_f1": 0.43
    re.compile(r'"eval_f1"\s*:\s*([0-9]+\.[0-9]+)', re.IGNORECASE),
    # best_val_f1 / best_f1
    re.compile(r'best[_\s](?:val[_\s])?f1[\s:=]+([0-9]+\.[0-9]+)', re.IGNORECASE),
]

PRECISION_PATTERNS = [
    re.compile(r'"precision"\s*:\s*([0-9]+\.[0-9]+)', re.IGNORECASE),
    re.compile(r'\bprecision[\s:=]+([0-9]+\.[0-9]+)', re.IGNORECASE),
]

RECALL_PATTERNS = [
    re.compile(r'"recall"\s*:\s*([0-9]+\.[0-9]+)', re.IGNORECASE),
    re.compile(r'\brecall[\s:=]+([0-9]+\.[0-9]+)', re.IGNORECASE),
]

FAILED_PATTERNS = [
    re.compile(r'\bFAILED\b'),
    re.compile(r'Traceback \(most recent call last\)'),
    re.compile(r'CUDA out of memory'),
    re.compile(r'Error:'),
    re.compile(r'exit code: [^0]'),
    re.compile(r'RuntimeError'),
    re.compile(r'AssertionError'),
]

SUCCESS_PATTERNS = [
    re.compile(r'\bSUCCESS\b'),
    re.compile(r'ALL EVALUATIONS COMPLETE'),
    re.compile(r'Training complete'),
    re.compile(r'Saved best model'),
]


# ─── Data class ──────────────────────────────────────────────────────────────
@dataclass
class RunResult:
    model: str = ""
    variant: str = ""
    source: str = ""          # "eval_json" | "training_summary" | "log"
    split: str = ""           # "gold" | "silver" | "val"
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    failed: bool = False
    success: bool = False
    job_id: str = ""
    log_path: str = ""
    notes: list = field(default_factory=list)

    @property
    def label(self):
        return f"{self.model}/{self.variant}"

    def flag(self):
        if self.failed:
            return f"{RED}✗ FAILED{RESET}"
        if self.f1 is None:
            return f"{YELLOW}? NO F1{RESET}"
        if self.f1 >= 0.50:
            return f"{GREEN}★ {self.f1:.4f}{RESET}"
        if self.f1 >= 0.35:
            return f"{CYAN}● {self.f1:.4f}{RESET}"
        return f"{RED}▼ {self.f1:.4f}{RESET}"


# ─── Helpers ──────────────────────────────────────────────────────────────────
def extract_first_match(patterns, text):
    """Return first float match from any pattern in text."""
    for pat in patterns:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def extract_all_f1(patterns, text):
    """Return the LAST (most recent / final) float match."""
    matches = []
    for pat in patterns:
        for m in pat.finditer(text):
            try:
                matches.append((m.start(), float(m.group(1))))
            except ValueError:
                continue
    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[-1][1]
    return None


def check_failed(text):
    return any(p.search(text) for p in FAILED_PATTERNS)


def check_success(text):
    return any(p.search(text) for p in SUCCESS_PATTERNS)


def read_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ─── Scanners ────────────────────────────────────────────────────────────────
def scan_evaluation_results(results_dir: Path):
    """Scan results/<model>/<variant>/gold|silver/evaluation_results.json"""
    found = []
    if not results_dir.exists():
        return found

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for variant_dir in sorted(model_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            variant = variant_dir.name
            for split in ["gold", "silver"]:
                json_path = variant_dir / split / "evaluation_results.json"
                log_path  = variant_dir / f"{split}_eval.log"
                if not json_path.exists():
                    continue
                data = read_json_safe(json_path)
                if data is None:
                    continue
                r = RunResult(
                    model=model,
                    variant=variant,
                    source="eval_json",
                    split=split,
                    log_path=str(json_path),
                )
                # Try common key names
                for key in ["f1", "span_f1", "micro_f1", "overall_f1", "eval_f1"]:
                    if key in data and data[key] is not None:
                        r.f1 = float(data[key])
                        break
                # Nested: {"overall": {"f1": ...}}
                if r.f1 is None and "overall" in data:
                    for key in ["f1", "f1-score"]:
                        if key in data["overall"]:
                            r.f1 = float(data["overall"][key])
                            break
                for key in ["precision", "span_precision"]:
                    if key in data:
                        r.precision = float(data[key])
                        break
                for key in ["recall", "span_recall"]:
                    if key in data:
                        r.recall = float(data[key])
                        break
                r.success = r.f1 is not None
                found.append(r)
    return found


def scan_training_summaries(ckpt_dir: Path):
    """Scan models/checkpoints/<model>/<variant>/training_summary.json"""
    found = []
    if not ckpt_dir.exists():
        return found

    for model_dir in sorted(ckpt_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for variant_dir in sorted(model_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            variant = variant_dir.name
            for fname in ["training_summary.json", "all_results.json", "trainer_state.json"]:
                json_path = variant_dir / fname
                if not json_path.exists():
                    continue
                data = read_json_safe(json_path)
                if data is None:
                    continue
                r = RunResult(
                    model=model,
                    variant=variant,
                    source="training_summary",
                    split="val",
                    log_path=str(json_path),
                )
                # Look for best f1 in training log history
                if fname == "trainer_state.json" and "log_history" in data:
                    best_f1 = None
                    for entry in data["log_history"]:
                        for key in ["eval_f1", "f1"]:
                            if key in entry:
                                v = float(entry[key])
                                if best_f1 is None or v > best_f1:
                                    best_f1 = v
                    r.f1 = best_f1
                else:
                    for key in ["best_val_f1", "best_f1", "eval_f1", "f1",
                                "span_f1", "val_f1", "final_f1"]:
                        if key in data and data[key] is not None:
                            r.f1 = float(data[key])
                            break
                    # Nested under "best_metrics" or similar
                    if r.f1 is None:
                        for nested_key in ["best_metrics", "eval_metrics", "metrics"]:
                            if nested_key in data and isinstance(data[nested_key], dict):
                                for key in ["f1", "eval_f1", "span_f1"]:
                                    if key in data[nested_key]:
                                        r.f1 = float(data[nested_key][key])
                                        break
                            if r.f1 is not None:
                                break
                if r.f1 is not None:
                    r.success = True
                    found.append(r)
                break  # Only use first found summary file per variant
    return found


def scan_slurm_logs(log_dir: Path, model_filter: str = ""):
    """Scan all .out files in logs/slurm_output/<model>/"""
    found = []
    if not log_dir.exists():
        return found

    for model_dir in sorted(log_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        if model_filter and model_filter.lower() not in model.lower():
            continue

        for log_file in sorted(model_dir.glob("*.out")):
            # Parse job_id and variant from filename: {jobid}_{variant}.out
            stem = log_file.stem
            parts = stem.split("_", 1)
            job_id = parts[0] if parts[0].isdigit() else ""
            variant = parts[1] if len(parts) > 1 else stem

            try:
                text = log_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            r = RunResult(
                model=model,
                variant=variant,
                source="log",
                split="log",
                job_id=job_id,
                log_path=str(log_file),
            )

            r.failed  = check_failed(text)
            r.success = check_success(text)
            r.f1        = extract_all_f1(F1_PATTERNS, text)
            r.precision = extract_first_match(PRECISION_PATTERNS, text)
            r.recall    = extract_first_match(RECALL_PATTERNS, text)

            # Annotate from known jobs table
            if job_id in KNOWN_JOBS:
                r.notes.append(f"Job: {KNOWN_JOBS[job_id]}")

            # Warn if completed but no F1
            if r.success and r.f1 is None:
                r.notes.append("⚠ SUCCESS but no F1 found in log")

            # Detect partial run (log has content but no SUCCESS/FAILED)
            if not r.failed and not r.success and len(text.strip()) > 200:
                r.notes.append("⏳ still running or incomplete")

            found.append(r)

    return found


# ─── Dedup & rank ─────────────────────────────────────────────────────────────
def deduplicate(results):
    """
    For (model, variant, split) keep the result with highest priority:
    eval_json > training_summary > log
    """
    source_priority = {"eval_json": 0, "training_summary": 1, "log": 2}
    seen = {}
    for r in results:
        key = (r.model, r.variant, r.split)
        if key not in seen:
            seen[key] = r
        else:
            existing = seen[key]
            if source_priority[r.source] < source_priority[existing.source]:
                seen[key] = r
            elif (source_priority[r.source] == source_priority[existing.source]
                  and r.f1 is not None
                  and (existing.f1 is None or r.f1 > existing.f1)):
                seen[key] = r
    return list(seen.values())


# ─── Display ──────────────────────────────────────────────────────────────────
def print_table(results, top_n=None, show_all_splits=False):
    if not results:
        print(f"{YELLOW}No results found.{RESET}")
        return

    # Group by model/variant, pick best split (gold > silver > val > log)
    split_prio = {"gold": 0, "silver": 1, "val": 2, "log": 3}
    best = {}
    for r in results:
        key = (r.model, r.variant)
        if key not in best:
            best[key] = r
        else:
            cur = best[key]
            # Prefer gold split, then higher F1
            if split_prio.get(r.split, 99) < split_prio.get(cur.split, 99):
                best[key] = r
            elif (split_prio.get(r.split, 99) == split_prio.get(cur.split, 99)
                  and r.f1 is not None
                  and (cur.f1 is None or r.f1 > cur.f1)):
                best[key] = r

    ranked = sorted(best.values(),
                    key=lambda r: (r.f1 is None, -(r.f1 or -1)))

    if top_n:
        ranked = ranked[:top_n]

    # Header
    col_model   = 28
    col_variant = 32
    col_f1      = 10
    col_pr      = 10
    col_re      = 10
    col_split   = 8
    col_src     = 16

    sep = "─" * (col_model + col_variant + col_f1 + col_pr + col_re + col_split + col_src + 14)
    header = (
        f"{BOLD}"
        f"{'Model':<{col_model}} "
        f"{'Variant':<{col_variant}} "
        f"{'F1':>{col_f1}} "
        f"{'Prec':>{col_pr}} "
        f"{'Recall':>{col_re}} "
        f"{'Split':<{col_split}} "
        f"{'Source':<{col_src}}"
        f"{RESET}"
    )

    print()
    print(f"{BOLD}{CYAN}{'═'*len(sep)}{RESET}")
    print(f"{BOLD}{CYAN} RESULTS SUMMARY — {datetime.now().strftime('%H:%M:%S')}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*len(sep)}{RESET}")
    print(header)
    print(sep)

    for i, r in enumerate(ranked):
        f1_str   = f"{r.f1:.4f}"  if r.f1  is not None else "—"
        pr_str   = f"{r.precision:.4f}" if r.precision is not None else "—"
        re_str   = f"{r.recall:.4f}"    if r.recall    is not None else "—"

        # Colour the F1 cell
        if r.failed:
            f1_disp = f"{RED}{f1_str:>{col_f1}}{RESET}"
        elif r.f1 is None:
            f1_disp = f"{YELLOW}{'—':>{col_f1}}{RESET}"
        elif r.f1 >= 0.50:
            f1_disp = f"{GREEN}{f1_str:>{col_f1}}{RESET}"
        elif r.f1 >= 0.35:
            f1_disp = f"{CYAN}{f1_str:>{col_f1}}{RESET}"
        else:
            f1_disp = f"{RED}{f1_str:>{col_f1}}{RESET}"

        status = ""
        if r.failed:
            status = f" {RED}[FAILED]{RESET}"
        elif r.f1 is None and not r.failed:
            status = f" {YELLOW}[NO RESULT]{RESET}"

        row = (
            f"{r.model:<{col_model}} "
            f"{r.variant:<{col_variant}} "
            f"{f1_disp} "
            f"{pr_str:>{col_pr}} "
            f"{re_str:>{col_re}} "
            f"{r.split:<{col_split}} "
            f"{r.source:<{col_src}}"
            f"{status}"
        )
        print(row)

        for note in r.notes:
            print(f"   {DIM}↳ {note}{RESET}")

    print(sep)
    print(f"{DIM}Total: {len(ranked)} variants | "
          f"With F1: {sum(1 for r in ranked if r.f1 is not None)} | "
          f"Failed: {sum(1 for r in ranked if r.failed)} | "
          f"No result: {sum(1 for r in ranked if r.f1 is None and not r.failed)}"
          f"{RESET}")
    print()


def print_gold_vs_silver(results):
    """Show gold vs silver side-by-side for each variant."""
    from collections import defaultdict
    grouped = defaultdict(dict)
    for r in results:
        if r.source == "log":
            continue
        key = (r.model, r.variant)
        grouped[key][r.split] = r

    if not grouped:
        return

    print(f"{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN} GOLD vs SILVER COMPARISON{RESET}")
    print(f"{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{'Variant':<38} {'Gold F1':>9} {'Silver F1':>10} {'Δ':>8}{RESET}")
    print("─" * 70)

    rows = []
    for (model, variant), splits in grouped.items():
        gold_f1   = splits.get("gold",   RunResult()).f1
        silver_f1 = splits.get("silver", RunResult()).f1
        delta = None
        if gold_f1 is not None and silver_f1 is not None:
            delta = gold_f1 - silver_f1
        rows.append((model, variant, gold_f1, silver_f1, delta))

    rows.sort(key=lambda x: (x[2] is None, -(x[2] or -1)))

    for model, variant, gold_f1, silver_f1, delta in rows:
        label = f"{model}/{variant}"[:37]
        g = f"{gold_f1:.4f}"   if gold_f1   is not None else "—"
        s = f"{silver_f1:.4f}" if silver_f1 is not None else "—"
        d = f"{delta:+.4f}"    if delta     is not None else "—"
        colour = RESET
        if delta is not None:
            colour = GREEN if delta >= 0 else YELLOW
        print(f"{label:<38} {g:>9} {s:>10} {colour}{d:>8}{RESET}")
    print()


def print_failed_jobs(results):
    """Print all failed jobs with their log path."""
    failed = [r for r in results if r.failed]
    if not failed:
        print(f"{GREEN}✓ No failed jobs found.{RESET}\n")
        return
    print(f"{BOLD}{RED}{'═'*60}{RESET}")
    print(f"{BOLD}{RED} FAILED JOBS ({len(failed)}){RESET}")
    print(f"{BOLD}{RED}{'═'*60}{RESET}")
    for r in failed:
        print(f"  {RED}✗{RESET} {r.model}/{r.variant}")
        print(f"       Log: {DIM}{r.log_path}{RESET}")
        for note in r.notes:
            print(f"       {DIM}↳ {note}{RESET}")
    print()


def print_no_result_jobs(results):
    """Print jobs with no F1 that are not failed (still running or broken)."""
    no_result = [r for r in results if r.f1 is None and not r.failed]
    if not no_result:
        return
    print(f"{BOLD}{YELLOW}{'═'*60}{RESET}")
    print(f"{BOLD}{YELLOW} NO F1 YET ({len(no_result)}) — still running or not started{RESET}")
    print(f"{BOLD}{YELLOW}{'═'*60}{RESET}")
    for r in no_result:
        label = f"{r.model}/{r.variant}" if r.model else r.variant
        src   = r.source if r.source else "—"
        print(f"  {YELLOW}?{RESET} {label:<45} [{src}]")
        for note in r.notes:
            print(f"       {DIM}↳ {note}{RESET}")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Check F1 scores across all SLURM jobs")
    parser.add_argument(
        "--log-dir",
        default="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output",
        help="Root of SLURM .out log files",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root of evaluation_results.json files (default: results/)",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="models/checkpoints",
        help="Root of model checkpoints (default: models/checkpoints/)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only top N variants by F1",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Filter by model name substring (e.g. dictabert_crf)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all splits (gold + silver) separately",
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Skip scanning SLURM .out files (faster)",
    )
    args = parser.parse_args()

    log_dir     = Path(args.log_dir)
    results_dir = Path(args.results_dir)
    ckpt_dir    = Path(args.ckpt_dir)

    print(f"\n{BOLD}Scanning directories...{RESET}")
    print(f"  SLURM logs  : {log_dir}")
    print(f"  Results     : {results_dir}")
    print(f"  Checkpoints : {ckpt_dir}")

    all_results = []

    # 1. Eval JSONs (most reliable)
    eval_results = scan_evaluation_results(results_dir)
    print(f"  Found {len(eval_results)} evaluation_results.json entries")
    all_results.extend(eval_results)

    # 2. Training summaries
    train_results = scan_training_summaries(ckpt_dir)
    print(f"  Found {len(train_results)} training summary entries")
    all_results.extend(train_results)

    # 3. SLURM logs (optional)
    if not args.no_logs:
        log_results = scan_slurm_logs(log_dir, model_filter=args.model)
        print(f"  Found {len(log_results)} SLURM .out files")
        all_results.extend(log_results)

    # Apply model filter
    if args.model:
        all_results = [r for r in all_results
                       if args.model.lower() in r.model.lower()
                       or args.model.lower() in r.variant.lower()]

    # Dedup
    deduped = deduplicate(all_results)

    # Separate eval-JSON and log results for different displays
    structured = [r for r in deduped if r.source != "log"]
    log_only   = [r for r in deduped if r.source == "log"]

    # Print main table (structured results = most accurate)
    if structured:
        print()
        print(f"{BOLD}=== STRUCTURED RESULTS (eval JSON + training summary) ==={RESET}")
        print_table(structured, top_n=args.top)
        if args.verbose or True:
            print_gold_vs_silver(structured)
    else:
        print(f"\n{YELLOW}No structured results found yet — only log files available.{RESET}\n")

    # Log-only table (for running jobs)
    if log_only:
        print(f"{BOLD}=== LOG-ONLY RESULTS (from .out files — less reliable) ==={RESET}")
        print_table(log_only, top_n=args.top)

    # Status sections
    print_failed_jobs(all_results)
    print_no_result_jobs(all_results)

    # Quick best-model line
    with_f1 = [r for r in deduped if r.f1 is not None and not r.failed]
    if with_f1:
        # Prefer gold split
        gold_only = [r for r in with_f1 if r.split == "gold"]
        candidates = gold_only if gold_only else with_f1
        best = max(candidates, key=lambda r: r.f1)
        print(f"{BOLD}{GREEN}★  BEST SO FAR: {best.model}/{best.variant} "
              f"F1={best.f1:.4f} [{best.split}]{RESET}\n")

    # Summary of the known running jobs from the pasted squeue
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN} KNOWN RUNNING JOBS (from last squeue){RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")
    running_ids = [
        "305156","305158","305159","305160",
        "305161","305162","305163","305164",
    ]
    pending_ids = [
        "305170","305171","305172","305173",
        "305174","305175","305176","305177",
        "305178","305179",
    ]
    print(f"  {GREEN}RUNNING ({len(running_ids)}){RESET}")
    for jid in running_ids:
        label = KNOWN_JOBS.get(jid, "unknown")
        print(f"    [{jid}] {label}")
    print(f"  {YELLOW}PENDING ({len(pending_ids)}){RESET}")
    for jid in pending_ids:
        label = KNOWN_JOBS.get(jid, "unknown")
        print(f"    [{jid}] {label}")
    print()

    # Note about 305156 vs 305160 duplicate question
    print(f"{DIM}NOTE: You asked about 305156 vs 305160 (both named 'dlarge_c').{RESET}")
    print(f"{DIM}      Run: grep -h 'VARIANT\\|OUTPUT' "
          f"/path/to/logs/slurm_output/dictabert_large/305156*.out "
          f"/path/to/logs/slurm_output/dictabert_large/305160*.out{RESET}")
    print(f"{DIM}      to see their actual VARIANT/OUTPUT lines and confirm.{RESET}\n")


if __name__ == "__main__":
    main()