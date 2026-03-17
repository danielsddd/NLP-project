"""
Preprocessing Module
====================
Converts silver labels (spans) to BIO tags aligned with AlephBERT tokenization.

Key design principle (v2):
  - We do NOT feed the full comment/thread to AlephBERT.
  - We find the specific sentence that CONTAINS the modification span, and feed
    only that sentence. This gives AlephBERT a clean, focused training signal.
  - signal_type and output_note from the teacher go directly to aggregate.py;
    AlephBERT only needs to learn 9 span labels.

Usage:
    python -m src.preprocessing.prepare_data \
        --input data/silver_labels/teacher_output.jsonl \
        --output-dir data/processed
"""

import json
import logging
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter

from tqdm import tqdm

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not installed. Run: pip install transformers")


# =============================================================================
# LABEL SCHEMA  (9 labels — intentionally minimal)
# =============================================================================

BIO_LABEL2ID = {
    "O": 0,
    "B-SUBSTITUTION": 1,
    "I-SUBSTITUTION": 2,
    "B-QUANTITY": 3,
    "I-QUANTITY": 4,
    "B-TECHNIQUE": 5,
    "I-TECHNIQUE": 6,
    "B-ADDITION": 7,
    "I-ADDITION": 8,
}
BIO_ID2LABEL = {v: k for k, v in BIO_LABEL2ID.items()}

# IO tagging scheme (5 labels) — ablation only
IO_LABEL2ID = {
    "O": 0,
    "I-SUBSTITUTION": 1,
    "I-QUANTITY": 2,
    "I-TECHNIQUE": 3,
    "I-ADDITION": 4,
}
IO_ID2LABEL = {v: k for k, v in IO_LABEL2ID.items()}

VALID_ASPECTS = {"SUBSTITUTION", "QUANTITY", "TECHNIQUE", "ADDITION"}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ProcessedExample:
    """A single processed example ready for training."""
    comment_id: str
    original_text: str           # Full original comment text
    training_text: str           # Sentence actually fed to AlephBERT

    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]            # BIO tag IDs, aligned to training_text tokens

    num_modifications: int = 0
    aspects_found: List[str] = field(default_factory=list)

    # Passthrough fields from teacher (used by aggregate.py, not AlephBERT)
    signal_type: Optional[str] = None
    output_note: Optional[str] = None
    like_count: int = 0
    video_id: Optional[str] = None

    # Debug
    tokens: Optional[List[str]] = None
    label_strings: Optional[List[str]] = None


@dataclass
class AlignmentStats:
    """Statistics from the alignment process."""
    total_examples: int = 0
    examples_with_mods: int = 0
    total_modifications: int = 0
    successful_alignments: int = 0
    failed_alignments: int = 0
    hallucinated_spans: int = 0
    used_full_text_fallback: int = 0   # Times we fell back to full text
    aspect_counts: Dict[str, int] = field(default_factory=lambda: {a: 0 for a in VALID_ASPECTS})
    avg_tokens_per_example: float = 0.0
    avg_tagged_tokens_per_example: float = 0.0


# =============================================================================
# SENTENCE EXTRACTION  (the core new logic)
# =============================================================================

def split_into_sentences(text: str) -> List[Tuple[str, int, int]]:
    """
    Split Hebrew/mixed text into sentences.

    Returns a list of (sentence, start_char, end_char) tuples.
    Hebrew uses periods, exclamation marks, question marks, and newlines
    as sentence boundaries. We also handle comma-separated clauses as
    separate units for very long texts.
    """
    # Split on sentence-ending punctuation or newlines
    pattern = r'(?<=[.!?])\s+|(?<=\n)'
    parts = re.split(pattern, text)

    sentences = []
    cursor = 0
    for part in parts:
        part = part.strip()
        if not part:
            # advance cursor past whitespace
            idx = text.find('\n', cursor)
            if idx >= 0:
                cursor = idx + 1
            continue
        start = text.find(part, cursor)
        if start == -1:
            start = cursor
        end = start + len(part)
        sentences.append((part, start, end))
        cursor = end

    return sentences if sentences else [(text, 0, len(text))]


def extract_training_sentence(
    text: str,
    modifications: List[Dict],
) -> Tuple[str, bool]:
    """
    Find the sentence (or minimal contiguous substring) that contains ALL
    modification spans for this example.

    Args:
        text: Full comment text
        modifications: List of modification dicts with 'span' key

    Returns:
        (training_text, used_fallback)
        - training_text: The extracted sentence or full text if fallback needed
        - used_fallback: True if we had to fall back to the full text
    """
    if not modifications:
        return text, False

    # Find character ranges for all spans
    span_ranges: List[Tuple[int, int]] = []
    for mod in modifications:
        span = mod.get("span", "")
        if not span:
            continue

        start = mod.get("start_char")
        end = mod.get("end_char")

        # If pre-computed indices are valid, use them
        if start is not None and end is not None and start >= 0 and end > start:
            span_ranges.append((start, end))
        else:
            # Search for span in text
            idx = text.find(span)
            if idx >= 0:
                span_ranges.append((idx, idx + len(span)))
            else:
                # Normalized whitespace fallback
                norm_span = " ".join(span.split())
                norm_text = " ".join(text.split())
                idx = norm_text.find(norm_span)
                if idx >= 0:
                    span_ranges.append((idx, idx + len(norm_span)))

    if not span_ranges:
        return text, True  # No spans found — full text fallback

    # Overall span extent across all modifications in this example
    all_start = min(r[0] for r in span_ranges)
    all_end = max(r[1] for r in span_ranges)

    # Try to find a sentence that contains all spans
    sentences = split_into_sentences(text)

    covering_sentence = None
    for sent_text, sent_start, sent_end in sentences:
        if sent_start <= all_start and sent_end >= all_end:
            covering_sentence = sent_text
            break

    if covering_sentence:
        return covering_sentence, False

    # No single sentence covers all spans — find the minimal covering window
    # (Take from the start of the first sentence containing any span to the
    #  end of the last sentence containing any span.)
    containing_sentences = []
    for sent_text, sent_start, sent_end in sentences:
        for r_start, r_end in span_ranges:
            if sent_start <= r_start < sent_end or sent_start < r_end <= sent_end:
                containing_sentences.append((sent_text, sent_start, sent_end))
                break

    if containing_sentences:
        window_start = containing_sentences[0][1]
        window_end = containing_sentences[-1][2]
        return text[window_start:window_end].strip(), False

    # Complete fallback
    return text, True


# =============================================================================
# SPAN ALIGNMENT  (unchanged from v1 — correct algorithm)
# =============================================================================

def find_span_in_text(text: str, span: str) -> Optional[Tuple[int, int]]:
    """Find character indices of a span in text (direct + normalized fallback)."""
    idx = text.find(span)
    if idx >= 0:
        return (idx, idx + len(span))
    # Normalized whitespace
    norm_span = " ".join(span.split())
    norm_text = " ".join(text.split())
    idx = norm_text.find(norm_span)
    if idx >= 0:
        return (idx, idx + len(norm_span))
    return None


def token_overlaps_span(
    token_start: int,
    token_end: int,
    span_start: int,
    span_end: int,
) -> bool:
    """True if a token's character range overlaps a span's character range."""
    if token_end <= span_start:
        return False
    if token_start >= span_end:
        return False
    return True


def align_spans_to_bio(
    training_text: str,
    modifications: List[Dict],
    tokenizer: "PreTrainedTokenizer",
    max_length: int = 128,
    tagging_scheme: str = "BIO",
) -> Optional[ProcessedExample]:
    """
    Align character-level spans to token-level BIO tags.

    Segal et al. (2020) algorithm adapted for our task.

    NOTE: training_text is the EXTRACTED sentence, not the full comment.
    The span character indices in modifications are relative to the original
    full text, so we re-locate them within training_text here.

    Args:
        training_text: The sentence to tokenize and tag
        modifications:  List of modification dicts (span, aspect, start_char, end_char)
        tokenizer:      AlephBERT tokenizer
        max_length:     Max sequence length
        tagging_scheme: "BIO" or "IO"

    Returns:
        ProcessedExample (comment_id is "" and will be set by caller)
        or None if tokenization fails
    """
    if not training_text.strip():
        return None

    # Tokenize with offset mapping
    encoding = tokenizer(
        training_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors=None,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offsets = encoding["offset_mapping"]

    label2id = BIO_LABEL2ID if tagging_scheme == "BIO" else IO_LABEL2ID
    id2label = BIO_ID2LABEL if tagging_scheme == "BIO" else IO_ID2LABEL
    labels = [label2id["O"]] * len(input_ids)

    aspects_found = []
    successful_alignments = 0

    for mod in modifications:
        span_text = mod.get("span", "")
        aspect = mod.get("aspect", "").upper()

        if aspect not in VALID_ASPECTS or not span_text:
            continue

        # Re-locate the span within training_text
        # (original char indices may be relative to the full comment text)
        found = find_span_in_text(training_text, span_text)
        if found is None:
            continue  # Span not in this training sentence — skip
        start_char, end_char = found

        # Find overlapping tokens
        span_token_indices = []
        for token_idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end == 0:
                continue  # Special token
            if token_overlaps_span(tok_start, tok_end, start_char, end_char):
                span_token_indices.append(token_idx)

        if span_token_indices:
            if tagging_scheme == "BIO":
                b_tag = f"B-{aspect}"
                i_tag = f"I-{aspect}"
                labels[span_token_indices[0]] = label2id[b_tag]
                for token_idx in span_token_indices[1:]:
                    if labels[token_idx] == label2id["O"]:
                        labels[token_idx] = label2id[i_tag]
            else:
                i_tag = f"I-{aspect}"
                for token_idx in span_token_indices:
                    if labels[token_idx] == label2id["O"]:
                        labels[token_idx] = label2id[i_tag]

            aspects_found.append(aspect)
            successful_alignments += 1

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    label_strings = [id2label[l] for l in labels]

    return ProcessedExample(
        comment_id="",
        original_text="",        # Set by caller
        training_text=training_text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        num_modifications=len(modifications),
        aspects_found=aspects_found,
        tokens=tokens,
        label_strings=label_strings,
    )


# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

class DataPreprocessor:
    """
    Main preprocessing pipeline.

    Reads teacher_output.jsonl → extracts training sentences →
    aligns spans to BIO tags → saves train/val/test splits.
    """

    def __init__(
        self,
        model_name: str = "onlplab/alephbert-base",
        max_length: int = 128,
        tagging_scheme: str = "BIO",
        output_dir: str = "data/processed",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.tagging_scheme = tagging_scheme
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed")

        self.logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logger.info("Tokenizer loaded ✓")

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def process_file(
        self,
        input_file: str,
        output_prefix: str = "dataset",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        include_no_modification: bool = True,
    ) -> AlignmentStats:
        """
        Process teacher_output.jsonl → train/val/test JSONL splits.

        Args:
            input_file:              Path to teacher_output.jsonl
            output_prefix:           Prefix for output files
            train_ratio:             Fraction for training
            val_ratio:               Fraction for validation
            test_ratio:              Fraction for test
            seed:                    Random seed
            include_no_modification: Include examples without modifications

        Returns:
            AlignmentStats with processing statistics
        """
        random.seed(seed)

        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        self.logger.info(f"Processing: {input_file}")
        self.logger.info(f"Tagging scheme: {self.tagging_scheme}")

        examples = []
        stats = AlignmentStats()

        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Processing examples"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # ── Pull fields from the JSONL record ──────────────────────────
            comment_id = data.get("comment_id", "")
            original_text = data.get("text", "")
            like_count = data.get("like_count", 0)
            video_id = data.get("video_id", "")

            teacher_output = data.get("teacher_output", {})
            modifications = teacher_output.get("modifications", [])
            has_modification = teacher_output.get("has_modification", False)

            # Passthrough fields (v3 generate_labels output; safe to default)
            signal_type = teacher_output.get("signal_type", None)
            output_note = teacher_output.get("output_note", None)

            stats.total_examples += 1

            if not has_modification and not include_no_modification:
                continue

            if has_modification:
                stats.examples_with_mods += 1
                stats.total_modifications += len(modifications)

            # ── Extract training sentence ───────────────────────────────────
            training_text, used_fallback = extract_training_sentence(
                original_text, modifications
            )
            if used_fallback:
                stats.used_full_text_fallback += 1

            # ── Align spans → BIO tags (on training_text) ──────────────────
            processed = align_spans_to_bio(
                training_text=training_text,
                modifications=modifications,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                tagging_scheme=self.tagging_scheme,
            )

            if processed is None:
                continue

            # ── Fill in metadata ────────────────────────────────────────────
            processed.comment_id = comment_id
            processed.original_text = original_text
            processed.signal_type = signal_type
            processed.output_note = output_note
            processed.like_count = like_count
            processed.video_id = video_id

            # ── Update stats ────────────────────────────────────────────────
            for aspect in processed.aspects_found:
                stats.aspect_counts[aspect] += 1

            expected = len(modifications)
            actual = len(processed.aspects_found)
            stats.successful_alignments += actual
            stats.hallucinated_spans += max(0, expected - actual)

            examples.append(processed)

        self.logger.info(f"Processed {len(examples)} usable examples")

        # ── Compute averages ──────────────────────────────────────────────
        if examples:
            total_tokens = sum(sum(e.attention_mask) for e in examples)
            total_tagged = sum(sum(1 for l in e.labels if l != 0) for e in examples)
            stats.avg_tokens_per_example = total_tokens / len(examples)
            stats.avg_tagged_tokens_per_example = total_tagged / len(examples)

        # ── Shuffle and split ─────────────────────────────────────────────
        random.shuffle(examples)

        n = len(examples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_examples = examples[:train_end]
        val_examples = examples[train_end:val_end]
        test_examples = examples[val_end:]

        # ── Save splits ───────────────────────────────────────────────────
        self._save_split(train_examples, output_prefix + "_train.jsonl")
        self._save_split(val_examples,   output_prefix + "_val.jsonl")
        self._save_split(test_examples,  output_prefix + "_test.jsonl")

        # ── Save label mapping ────────────────────────────────────────────
        label_map = {
            "label2id": BIO_LABEL2ID if self.tagging_scheme == "BIO" else IO_LABEL2ID,
            "id2label": {str(k): v for k, v in (
                BIO_ID2LABEL if self.tagging_scheme == "BIO" else IO_ID2LABEL
            ).items()},
            "num_labels": len(BIO_LABEL2ID) if self.tagging_scheme == "BIO" else len(IO_LABEL2ID),
            "tagging_scheme": self.tagging_scheme,
            "model_name": self.model_name,
        }
        with open(self.output_dir / "label_map.json", "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2, ensure_ascii=False)

        self._print_summary(stats, train_examples, val_examples, test_examples)
        return stats

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _save_split(self, examples: List[ProcessedExample], filename: str):
        """Save a list of examples to a JSONL file."""
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                record = {
                    "comment_id": ex.comment_id,
                    "original_text": ex.original_text,
                    "training_text": ex.training_text,
                    "input_ids": ex.input_ids,
                    "attention_mask": ex.attention_mask,
                    "labels": ex.labels,
                    "num_modifications": ex.num_modifications,
                    "aspects_found": ex.aspects_found,
                    # Passthrough for aggregate.py
                    "signal_type": ex.signal_type,
                    "output_note": ex.output_note,
                    "like_count": ex.like_count,
                    "video_id": ex.video_id,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.logger.info(f"Saved {len(examples)} examples → {output_path}")

    def _print_summary(
        self,
        stats: AlignmentStats,
        train: list,
        val: list,
        test: list,
    ):
        """Print preprocessing summary."""
        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"  Total input examples:      {stats.total_examples}")
        print(f"  Examples with mods:        {stats.examples_with_mods}")
        print(f"  Total modifications:       {stats.total_modifications}")
        print(f"  Successful alignments:     {stats.successful_alignments}")
        print(f"  Hallucinated spans:        {stats.hallucinated_spans}")
        print(f"  Full-text fallbacks used:  {stats.used_full_text_fallback}")
        print(f"\n  Avg tokens/example:        {stats.avg_tokens_per_example:.1f}")
        print(f"  Avg tagged tokens/example: {stats.avg_tagged_tokens_per_example:.1f}")
        print("\n  Aspect distribution:")
        for aspect, count in stats.aspect_counts.items():
            print(f"    {aspect}: {count}")
        print("\n  Splits:")
        print(f"    Train: {len(train)}")
        print(f"    Val:   {len(val)}")
        print(f"    Test:  {len(test)}")
        print("=" * 60)


# =============================================================================
# DEBUG VISUALIZATION
# =============================================================================

def visualize_alignment(example: ProcessedExample, max_tokens: int = 50):
    """Print token-label alignment for a single example (debugging)."""
    print(f"\nOriginal:  {example.original_text[:100]}{'...' if len(example.original_text) > 100 else ''}")
    print(f"Training:  {example.training_text[:100]}{'...' if len(example.training_text) > 100 else ''}")
    print("-" * 60)
    for i, (token, label) in enumerate(
        zip(example.tokens[:max_tokens], example.label_strings[:max_tokens])
    ):
        marker = "  ← TAGGED" if label != "O" else ""
        if token not in ("[CLS]", "[SEP]", "[PAD]") or label != "O":
            print(f"  [{i:3d}] {token:20s}  {label}{marker}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess silver labels for AlephBERT training"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input teacher_output.jsonl file")
    parser.add_argument("--output-dir", "-o", default="data/processed",
                        help="Output directory for splits")
    parser.add_argument("--model", default="onlplab/alephbert-base",
                        help="Tokenizer model name")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max sequence length for tokenizer")
    parser.add_argument("--scheme", choices=["BIO", "IO"], default="BIO",
                        help="Tagging scheme (BIO=9 labels, IO=5 labels)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffle")
    parser.add_argument("--no-empty", action="store_true",
                        help="Exclude examples without modifications")

    args = parser.parse_args()

    preprocessor = DataPreprocessor(
        model_name=args.model,
        max_length=args.max_length,
        tagging_scheme=args.scheme,
        output_dir=args.output_dir,
    )

    preprocessor.process_file(
        input_file=args.input,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        seed=args.seed,
        include_no_modification=not args.no_empty,
    )


if __name__ == "__main__":
    main()