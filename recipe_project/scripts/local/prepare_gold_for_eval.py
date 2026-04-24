#!/usr/bin/env python3
"""
Gold Adapter: tokenize gold_final.jsonl for each model's tokenizer.
====================================================================
Converts the raw adjudicated gold annotations into per-model
pre-tokenized JSONL files that src/evaluation/evaluate.py can consume
unchanged.

Input schema (gold_final.jsonl):
    {thread_id, comment_text, comment_position, has_modification,
     gold_modifications: [{span, aspect}],
     annotator_daniel, annotator_roei, overlap, resolution}

Output schema (gold_tokenized_<slug>.jsonl) — matches data/processed/test.jsonl:
    {input_ids, attention_mask, labels, text,
     thread_id, has_modification, comment_position,
     gold_modifications, annotator_daniel, annotator_roei,
     overlap, resolution}

Usage:
    # Tokenize for all 6 models at once (recommended before submitting sbatches):
    python scripts/prepare_gold_for_eval.py --all

    # Tokenize for one model only:
    python scripts/prepare_gold_for_eval.py --model dicta-il/dictabert

    # Custom paths / options:
    python scripts/prepare_gold_for_eval.py \
        --gold data/gold_validation/gold_final.jsonl \
        --model onlplab/alephbert-base \
        --output data/gold_validation/gold_tokenized_alephbert.jsonl \
        --max-length 128 \
        --scheme BIO
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

# Find project root by walking up until we hit the folder containing src/
# Works regardless of whether the script lives in scripts/, scripts/local/, etc.
def _find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "src" / "preprocessing" / "prepare_data.py").exists():
            return parent
    raise RuntimeError(
        f"Could not locate project root (no src/preprocessing/prepare_data.py found) "
        f"starting from {start}"
    )

PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer

from src.preprocessing.prepare_data import (
    align_example,
    BIO_LABEL2ID, IO_LABEL2ID,
    VALID_ASPECTS,
)


# =============================================================================
# MODEL REGISTRY
# =============================================================================
# Maps HuggingFace model IDs to short slugs used in output filenames.
# Keep this in sync with the 6 encoders in the main comparison table.

MODEL_REGISTRY = {
    "bert-base-multilingual-cased": "mbert",
    "avichr/heBERT":                "hebert",
    "xlm-roberta-base":             "xlmr",
    "onlplab/alephbert-base":       "alephbert",
    "dicta-il/dictabert":           "dictabert",
    "dicta-il/dictabert-large":     "dictabert_large",
}


# =============================================================================
# CORE ADAPTER
# =============================================================================

def adapt_gold_for_model(
    gold_path: Path,
    model_name: str,
    output_path: Path,
    scheme: str = "BIO",
    max_length: int = 128,
    verbose: bool = True,
):
    """Tokenize the gold file using one specific model's tokenizer.

    Args:
        gold_path:   Path to raw gold_final.jsonl.
        model_name:  HuggingFace model ID (e.g. "dicta-il/dictabert").
        output_path: Where to write the pre-tokenized JSONL.
        scheme:      "BIO" or "IO".
        max_length:  Max token sequence length (must match training).
        verbose:     Print progress.

    Returns:
        dict with summary counts.
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  GOLD ADAPTER")
        print(f"{'=' * 70}")
        print(f"  Gold file:   {gold_path}")
        print(f"  Tokenizer:   {model_name}")
        print(f"  Output:      {output_path}")
        print(f"  Scheme:      {scheme}   Max length: {max_length}")

    # ── 1. Load tokenizer + label map ────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label2id = BIO_LABEL2ID if scheme == "BIO" else IO_LABEL2ID

    # ── 2. Read gold records (robust to malformed lines) ─────────────────
    records = []
    malformed = 0
    with gold_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                malformed += 1
                if malformed <= 3 and verbose:
                    print(f"  WARN: line {i} malformed — {e}")

    if verbose:
        print(f"\n  Loaded {len(records)} gold records "
              f"({malformed} malformed lines skipped)")

    # ── 3. Tokenize + align each record ──────────────────────────────────
    output_records = []
    stats = Counter()
    aspect_counts = Counter()

    for rec in records:
        text = (rec.get("comment_text") or "").strip()
        if not text:
            stats["skipped_empty_text"] += 1
            continue

        gold_mods_raw = rec.get("gold_modifications") or []

        # Normalize + validate (defensive — gold should already be clean)
        gold_mods = []
        for m in gold_mods_raw:
            span = (m.get("span") or "").strip()
            aspect = (m.get("aspect") or "").upper().strip()
            if not span:
                stats["skipped_empty_span"] += 1
                continue
            if aspect not in VALID_ASPECTS:
                stats["skipped_invalid_aspect"] += 1
                continue
            gold_mods.append({"span": span, "aspect": aspect})
            aspect_counts[aspect] += 1

        # Canonical alignment (same function used during training)
        result, align_stats = align_example(
            text, gold_mods, tokenizer, label2id, max_length
        )
        if result is None:
            stats["skipped_alignment_failed"] += 1
            continue

        # Merge all gold metadata into the tokenized record
        result["thread_id"]          = rec.get("thread_id", "")
        result["comment_position"]   = rec.get("comment_position", "top")
        result["has_modification"]   = rec.get("has_modification", False)
        result["gold_modifications"] = gold_mods
        # Inter-annotator metadata — required for Cohen's kappa later
        result["annotator_daniel"]   = rec.get("annotator_daniel")
        result["annotator_roei"]     = rec.get("annotator_roei")
        result["overlap"]            = rec.get("overlap", False)
        result["resolution"]         = rec.get("resolution", "")

        output_records.append(result)
        stats["success"]            += 1
        stats["aligned_spans"]      += align_stats.get("aligned", 0)
        stats["hallucinated_spans"] += align_stats.get("hallucinated", 0)
        stats["truncated_spans"]    += align_stats.get("truncated", 0)

    # ── 4. Write output ──────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in output_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pos = sum(1 for r in output_records if r.get("has_modification"))
    positive_rate = n_pos / max(len(output_records), 1)

    # ── 5. Report ────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  Written:                 {len(output_records)} records")
        print(f"  Positive rate:           {n_pos} ({positive_rate * 100:.1f}%)")
        print(f"  Skipped — empty text:    {stats['skipped_empty_text']}")
        print(f"  Skipped — empty span:    {stats['skipped_empty_span']}")
        print(f"  Skipped — bad aspect:    {stats['skipped_invalid_aspect']}")
        print(f"  Skipped — align failed:  {stats['skipped_alignment_failed']}")
        print(f"\n  Span alignment breakdown:")
        print(f"    Aligned (found in text):        {stats['aligned_spans']}")
        print(f"    Hallucinated (span not found):  {stats['hallucinated_spans']}")
        print(f"    Truncated (past max_length):    {stats['truncated_spans']}")
        print(f"\n  Aspect distribution: {dict(aspect_counts)}")

    return {
        "model":              model_name,
        "output":             str(output_path),
        "n_records_in":       len(records),
        "n_written":          len(output_records),
        "n_positive":         n_pos,
        "positive_rate":      positive_rate,
        "aligned_spans":      stats["aligned_spans"],
        "hallucinated_spans": stats["hallucinated_spans"],
        "truncated_spans":    stats["truncated_spans"],
        "aspects":            dict(aspect_counts),
        "malformed_lines":    malformed,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tokenize gold annotations for each model's tokenizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/prepare_gold_for_eval.py --all\n"
            "  python scripts/prepare_gold_for_eval.py --model dicta-il/dictabert\n"
        ),
    )
    parser.add_argument(
        "--gold",
        default="data/gold_validation/gold_final.jsonl",
        help="Path to the raw gold JSONL (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID (omit and use --all for batch mode).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (auto-generated from slug if omitted).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Tokenize for all 6 registered models in MODEL_REGISTRY.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/gold_validation",
        help="Output directory for --all mode (default: %(default)s).",
    )
    parser.add_argument(
        "--max-length", type=int, default=128,
        help="Max token sequence length (default: 128 — must match training).",
    )
    parser.add_argument(
        "--scheme",
        default="BIO",
        choices=["BIO", "IO"],
        help="Tagging scheme (default: BIO).",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"ERROR: gold file not found: {gold_path}")
        print(f"       Current dir:          {Path.cwd()}")
        print(f"       Did you rename Gold_lables.jsonl -> gold_final.jsonl?")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # ── Batch mode ───────────────────────────────────────────────────────
    if args.all:
        print(f"Tokenizing gold for all {len(MODEL_REGISTRY)} models...")
        summaries = {}
        for model_name, slug in MODEL_REGISTRY.items():
            out_path = output_dir / f"gold_tokenized_{slug}.jsonl"
            try:
                summaries[slug] = adapt_gold_for_model(
                    gold_path, model_name, out_path,
                    scheme=args.scheme, max_length=args.max_length,
                )
            except Exception as e:
                print(f"\n  FAILED for {model_name}: {type(e).__name__}: {e}")
                summaries[slug] = {"error": f"{type(e).__name__}: {e}"}

        # Summary report
        print(f"\n{'=' * 70}")
        print(f"  FINAL SUMMARY ({len(summaries)} models)")
        print(f"{'=' * 70}")
        header = f"  {'Slug':<18s}  {'Written':>8s}  {'Positive':>11s}  {'Aligned':>8s}  {'Missed':>7s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for slug, s in summaries.items():
            if "error" in s:
                print(f"  {slug:<18s}  ERROR: {s['error'][:45]}")
                continue
            print(
                f"  {slug:<18s}  {s['n_written']:>8d}  "
                f"{s['n_positive']:>4d} ({s['positive_rate'] * 100:4.1f}%)   "
                f"{s['aligned_spans']:>8d}  {s['hallucinated_spans']:>7d}"
            )
        print(f"{'=' * 70}")
        print(f"  Files written to: {output_dir}/gold_tokenized_*.jsonl")
        return

    # ── Single-model mode ────────────────────────────────────────────────
    if args.model is None:
        print("ERROR: --model is required (or use --all).")
        print(f"Known models: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    slug = MODEL_REGISTRY.get(args.model, args.model.replace("/", "_"))
    out_path = Path(args.output) if args.output else (output_dir / f"gold_tokenized_{slug}.jsonl")

    adapt_gold_for_model(
        gold_path, args.model, out_path,
        scheme=args.scheme, max_length=args.max_length,
    )


if __name__ == "__main__":
    main()