"""
Preprocessing Module
====================
Converts silver labels (spans) to BIO tags aligned with AlephBERT tokenization.

This is the critical alignment step from Segal et al. (2020).

Usage:
    python -m src.preprocessing.prepare_data --input data/silver_labels/teacher_output.jsonl
"""

import json
import logging
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
# LABEL SCHEMA
# =============================================================================

# BIO tagging scheme (9 labels)
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

# IO tagging scheme (5 labels) - for ablation study
IO_LABEL2ID = {
    "O": 0,
    "I-SUBSTITUTION": 1,
    "I-QUANTITY": 2,
    "I-TECHNIQUE": 3,
    "I-ADDITION": 4,
}

IO_ID2LABEL = {v: k for k, v in IO_LABEL2ID.items()}

# Valid aspects
VALID_ASPECTS = {"SUBSTITUTION", "QUANTITY", "TECHNIQUE", "ADDITION"}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ProcessedExample:
    """A single processed example ready for training."""
    # Identifiers
    comment_id: str
    original_text: str
    
    # Tokenized inputs
    input_ids: List[int]
    attention_mask: List[int]
    
    # Labels
    labels: List[int]  # BIO tag IDs
    
    # Metadata
    num_modifications: int = 0
    aspects_found: List[str] = field(default_factory=list)
    
    # For debugging
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
    hallucinated_spans: int = 0  # Spans not found in text
    aspect_counts: Dict[str, int] = field(default_factory=lambda: {a: 0 for a in VALID_ASPECTS})
    avg_tokens_per_example: float = 0.0
    avg_tagged_tokens_per_example: float = 0.0


# =============================================================================
# ALIGNMENT FUNCTIONS
# =============================================================================

def find_span_in_text(text: str, span: str) -> Optional[Tuple[int, int]]:
    """
    Find the character indices of a span in text.
    
    Args:
        text: Original text
        span: Span to find
        
    Returns:
        (start_char, end_char) or None if not found
    """
    # Direct search
    idx = text.find(span)
    if idx >= 0:
        return (idx, idx + len(span))
    
    # Try with normalized whitespace
    normalized_span = " ".join(span.split())
    normalized_text = " ".join(text.split())
    idx = normalized_text.find(normalized_span)
    if idx >= 0:
        # This is approximate - may need refinement
        return (idx, idx + len(normalized_span))
    
    return None


def token_overlaps_span(
    token_start: int, 
    token_end: int, 
    span_start: int, 
    span_end: int
) -> bool:
    """Check if a token overlaps with a span."""
    # No overlap if token ends before span starts
    if token_end <= span_start:
        return False
    # No overlap if token starts after span ends
    if token_start >= span_end:
        return False
    return True


def align_spans_to_bio(
    text: str,
    modifications: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    tagging_scheme: str = "BIO",
) -> Optional[ProcessedExample]:
    """
    Align character-level spans to token-level BIO tags.
    
    This is the core algorithm from Segal et al. (2020) adapted for our task.
    
    Args:
        text: Original comment text
        modifications: List of modification dicts with span, aspect, start_char, end_char
        tokenizer: HuggingFace tokenizer (AlephBERT)
        max_length: Maximum sequence length
        tagging_scheme: "BIO" or "IO"
        
    Returns:
        ProcessedExample or None if alignment fails
    """
    # Tokenize with offset mapping
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors=None,
    )
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offsets = encoding["offset_mapping"]  # List of (start_char, end_char) per token
    
    # Initialize all labels as "O"
    label2id = BIO_LABEL2ID if tagging_scheme == "BIO" else IO_LABEL2ID
    labels = [label2id["O"]] * len(input_ids)
    
    # Track which aspects were found
    aspects_found = []
    successful_alignments = 0
    
    # Process each modification
    for mod in modifications:
        span_text = mod.get("span", "")
        aspect = mod.get("aspect", "").upper()
        
        # Validate aspect
        if aspect not in VALID_ASPECTS:
            continue
        
        # Get span character indices
        start_char = mod.get("start_char")
        end_char = mod.get("end_char")
        
        # If not provided, find in text
        if start_char is None or end_char is None or start_char < 0:
            found = find_span_in_text(text, span_text)
            if found is None:
                continue  # Hallucinated span
            start_char, end_char = found
        
        # Find all tokens that overlap with this span
        span_token_indices = []
        
        for token_idx, (tok_start, tok_end) in enumerate(offsets):
            # Skip special tokens (offset = (0, 0) for [CLS], [SEP], [PAD])
            if tok_start == tok_end == 0:
                continue
            
            if token_overlaps_span(tok_start, tok_end, start_char, end_char):
                span_token_indices.append(token_idx)
        
        # Assign BIO tags
        if span_token_indices:
            if tagging_scheme == "BIO":
                # First token gets "B" tag
                b_tag = f"B-{aspect}"
                i_tag = f"I-{aspect}"
                
                labels[span_token_indices[0]] = label2id[b_tag]
                
                # Subsequent tokens get "I" tag
                for token_idx in span_token_indices[1:]:
                    # Only assign if not already tagged (handle overlaps)
                    if labels[token_idx] == label2id["O"]:
                        labels[token_idx] = label2id[i_tag]
            else:
                # IO scheme - all tokens get "I" tag
                i_tag = f"I-{aspect}"
                for token_idx in span_token_indices:
                    if labels[token_idx] == label2id["O"]:
                        labels[token_idx] = label2id[i_tag]
            
            aspects_found.append(aspect)
            successful_alignments += 1
    
    # Get token strings for debugging
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    id2label = BIO_ID2LABEL if tagging_scheme == "BIO" else IO_ID2LABEL
    label_strings = [id2label[l] for l in labels]
    
    return ProcessedExample(
        comment_id="",  # Will be set by caller
        original_text=text,
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
    Main preprocessing pipeline for converting silver labels to training data.
    
    Usage:
        preprocessor = DataPreprocessor()
        preprocessor.process_file("data/silver_labels/teacher_output.jsonl")
    """
    
    def __init__(
        self,
        model_name: str = "onlplab/alephbert-base",
        max_length: int = 128,
        tagging_scheme: str = "BIO",
        output_dir: str = "data/processed",
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.tagging_scheme = tagging_scheme
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.label2id = BIO_LABEL2ID if tagging_scheme == "BIO" else IO_LABEL2ID
        self.id2label = BIO_ID2LABEL if tagging_scheme == "BIO" else IO_ID2LABEL
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
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
        Process a silver labels file and create train/val/test splits.
        
        Args:
            input_file: Path to teacher_output.jsonl
            output_prefix: Prefix for output files
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed for reproducibility
            include_no_modification: Include examples without modifications
            
        Returns:
            AlignmentStats with processing statistics
        """
        import random
        random.seed(seed)
        
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        self.logger.info(f"Processing {input_file}")
        self.logger.info(f"Tagging scheme: {self.tagging_scheme}")
        
        # Load and process examples
        examples = []
        stats = AlignmentStats()
        
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc="Processing examples"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            text = data.get("text", "")
            teacher_output = data.get("teacher_output", {})
            modifications = teacher_output.get("modifications", [])
            has_modification = teacher_output.get("has_modification", False)
            
            stats.total_examples += 1
            
            # Skip examples without modifications if configured
            if not has_modification and not include_no_modification:
                continue
            
            if has_modification:
                stats.examples_with_mods += 1
                stats.total_modifications += len(modifications)
            
            # Align spans to BIO tags
            processed = align_spans_to_bio(
                text=text,
                modifications=modifications,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                tagging_scheme=self.tagging_scheme,
            )
            
            if processed is None:
                continue
            
            processed.comment_id = data.get("comment_id", "")
            
            # Count statistics
            for aspect in processed.aspects_found:
                stats.aspect_counts[aspect] += 1
            
            if len(processed.aspects_found) > 0:
                stats.successful_alignments += len(processed.aspects_found)
            
            # Count hallucinated spans
            expected = len(modifications)
            actual = len(processed.aspects_found)
            stats.hallucinated_spans += (expected - actual)
            
            examples.append(processed)
        
        self.logger.info(f"Processed {len(examples)} examples")
        
        # Calculate averages
        if examples:
            total_tokens = sum(sum(e.attention_mask) for e in examples)
            total_tagged = sum(sum(1 for l in e.labels if l != 0) for e in examples)
            stats.avg_tokens_per_example = total_tokens / len(examples)
            stats.avg_tagged_tokens_per_example = total_tagged / len(examples)
        
        # Shuffle and split
        random.shuffle(examples)
        
        n = len(examples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_examples = examples[:train_end]
        val_examples = examples[train_end:val_end]
        test_examples = examples[val_end:]
        
        self.logger.info(f"Split: train={len(train_examples)}, val={len(val_examples)}, test={len(test_examples)}")
        
        # Save splits
        self._save_split(train_examples, "train", output_prefix)
        self._save_split(val_examples, "val", output_prefix)
        self._save_split(test_examples, "test", output_prefix)
        
        # Save metadata
        metadata = {
            "model_name": self.tokenizer.name_or_path,
            "max_length": self.max_length,
            "tagging_scheme": self.tagging_scheme,
            "label2id": self.label2id,
            "id2label": {str(k): v for k, v in self.id2label.items()},
            "num_labels": len(self.label2id),
            "train_size": len(train_examples),
            "val_size": len(val_examples),
            "test_size": len(test_examples),
            "stats": {
                "total_examples": stats.total_examples,
                "examples_with_mods": stats.examples_with_mods,
                "total_modifications": stats.total_modifications,
                "successful_alignments": stats.successful_alignments,
                "hallucinated_spans": stats.hallucinated_spans,
                "aspect_counts": stats.aspect_counts,
                "avg_tokens_per_example": stats.avg_tokens_per_example,
                "avg_tagged_tokens_per_example": stats.avg_tagged_tokens_per_example,
            }
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self._print_summary(stats)
        return stats
    
    def _save_split(self, examples: List[ProcessedExample], split_name: str, prefix: str):
        """Save a data split to JSONL."""
        output_path = self.output_dir / f"{prefix}_{split_name}.jsonl"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for ex in examples:
                data = {
                    "comment_id": ex.comment_id,
                    "text": ex.original_text,
                    "input_ids": ex.input_ids,
                    "attention_mask": ex.attention_mask,
                    "labels": ex.labels,
                    "num_modifications": ex.num_modifications,
                    "aspects_found": ex.aspects_found,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        self.logger.info(f"Saved {len(examples)} examples to {output_path}")
    
    def _print_summary(self, stats: AlignmentStats):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"  Total examples:          {stats.total_examples}")
        print(f"  Examples with mods:      {stats.examples_with_mods}")
        print(f"  Total modifications:     {stats.total_modifications}")
        print(f"  Successful alignments:   {stats.successful_alignments}")
        print(f"  Hallucinated spans:      {stats.hallucinated_spans}")
        print(f"\n  Avg tokens/example:      {stats.avg_tokens_per_example:.1f}")
        print(f"  Avg tagged tokens/example: {stats.avg_tagged_tokens_per_example:.1f}")
        print("\n  Aspect distribution:")
        for aspect, count in stats.aspect_counts.items():
            print(f"    {aspect}: {count}")
        print("=" * 60)


def visualize_alignment(example: ProcessedExample, max_tokens: int = 50):
    """
    Visualize token-label alignment for debugging.
    
    Args:
        example: ProcessedExample to visualize
        max_tokens: Maximum tokens to show
    """
    print(f"\nOriginal: {example.original_text[:100]}...")
    print("-" * 60)
    
    for i, (token, label) in enumerate(zip(example.tokens[:max_tokens], example.label_strings[:max_tokens])):
        if label != "O":
            print(f"  [{i:3d}] {token:20s} → {label}")
        elif token not in ["[CLS]", "[SEP]", "[PAD]"]:
            print(f"  [{i:3d}] {token:20s}   O")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess silver labels for training")
    parser.add_argument("--input", "-i", required=True, help="Input teacher_output.jsonl file")
    parser.add_argument("--output-dir", "-o", default="data/processed", help="Output directory")
    parser.add_argument("--model", default="onlplab/alephbert-base", help="Tokenizer model")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--scheme", choices=["BIO", "IO"], default="BIO", help="Tagging scheme")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-empty", action="store_true", help="Exclude examples without modifications")
    
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
