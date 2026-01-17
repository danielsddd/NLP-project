"""
Baseline Models
===============
Simple baselines for comparison (required by grading rubric).

Implements:
1. Majority baseline (all O)
2. Random baseline  
3. Keyword-based baseline (FIXED: properly locates keywords in tokenized text)

Usage:
    python -m src.baselines.run_baselines --data-dir data/processed

Note: For mBERT baseline, use train_student.py with --model bert-base-multilingual-cased
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from dataclasses import dataclass

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not installed. Keyword baseline will be limited.")

try:
    from seqeval.metrics import f1_score, precision_score, recall_score
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False
    print("⚠️  seqeval not installed. Run: pip install seqeval")


# =============================================================================
# LABEL SCHEMA
# =============================================================================

BIO_LABEL2ID = {
    "O": 0,
    "B-SUBSTITUTION": 1, "I-SUBSTITUTION": 2,
    "B-QUANTITY": 3, "I-QUANTITY": 4,
    "B-TECHNIQUE": 5, "I-TECHNIQUE": 6,
    "B-ADDITION": 7, "I-ADDITION": 8,
}

BIO_ID2LABEL = {v: k for k, v in BIO_LABEL2ID.items()}

ALL_LABELS = list(BIO_LABEL2ID.keys())


# =============================================================================
# KEYWORD DICTIONARIES (Hebrew)
# =============================================================================

ASPECT_KEYWORDS = {
    "SUBSTITUTION": [
        "במקום",
        "החלפתי",
        "השתמשתי ב",
        "אפשר גם",
        "עובד גם עם",
        "ולא",
        "החלפה",
        "תחליף",
        "שיניתי",
        "החלפתי את",
    ],
    "QUANTITY": [
        "יותר",
        "פחות",
        "כפול",
        "חצי",
        "הוספתי",
        "הורדתי",
        "כפלתי",
        "הגדלתי",
        "הקטנתי",
        "עוד",
        "כמות",
        "פי שניים",
        "פי שלוש",
        "כפית",
        "כף",
        "גרם",
    ],
    "TECHNIQUE": [
        "דקות",
        "שעות",
        "מעלות",
        "תנור",
        "כיריים",
        "לערבב",
        "לקפל",
        "לאפות",
        "לטגן",
        "לבשל",
        "טמפרטורה",
        "חימום",
        "אפיתי",
        "בישלתי",
        "טיגנתי",
        "חיממתי",
    ],
    "ADDITION": [
        "הוספתי גם",
        "שמתי גם",
        "אפשר להוסיף",
        "מומלץ להוסיף",
        "כדאי להוסיף",
        "נתתי גם",
        "הוספתי עוד",
        "גם שמתי",
    ],
}


# =============================================================================
# BASE CLASSES
# =============================================================================

@dataclass
class BaselineResult:
    """Result from a baseline evaluation."""
    name: str
    f1: float
    precision: float
    recall: float
    token_accuracy: float
    predictions: Optional[List[List[str]]] = None


class BaselineModel:
    """Base class for baseline models."""
    
    name: str = "base"
    
    def predict(self, example: Dict) -> List[int]:
        """
        Predict labels for an example.
        
        Args:
            example: Dict with input_ids, attention_mask, labels
            
        Returns:
            List of predicted label IDs
        """
        raise NotImplementedError
    
    def evaluate(self, examples: List[Dict]) -> BaselineResult:
        """Evaluate on a list of examples."""
        all_true = []
        all_pred = []
        
        total_tokens = 0
        correct_tokens = 0
        
        for example in examples:
            true_labels = example["labels"]
            pred_labels = self.predict(example)
            
            # Convert to strings for seqeval
            true_str = []
            pred_str = []
            
            for t, p in zip(true_labels, pred_labels):
                if t != -100:  # Ignore padding
                    true_str.append(BIO_ID2LABEL.get(t, "O"))
                    pred_str.append(BIO_ID2LABEL.get(p, "O"))
                    
                    total_tokens += 1
                    if t == p:
                        correct_tokens += 1
            
            all_true.append(true_str)
            all_pred.append(pred_str)
        
        # Calculate metrics
        if SEQEVAL_AVAILABLE and all_true:
            f1 = f1_score(all_true, all_pred)
            precision = precision_score(all_true, all_pred)
            recall = recall_score(all_true, all_pred)
        else:
            f1 = precision = recall = 0.0
        
        token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        
        return BaselineResult(
            name=self.name,
            f1=f1,
            precision=precision,
            recall=recall,
            token_accuracy=token_acc,
            predictions=all_pred,
        )


# =============================================================================
# BASELINE IMPLEMENTATIONS
# =============================================================================

class MajorityBaseline(BaselineModel):
    """
    Majority baseline: predict "O" for all tokens.
    
    This establishes the lower bound and shows that high token accuracy
    is misleading for NER tasks (most tokens are "O").
    """
    
    name = "majority"
    
    def predict(self, example: Dict) -> List[int]:
        """Predict all O."""
        return [BIO_LABEL2ID["O"]] * len(example["input_ids"])


class RandomBaseline(BaselineModel):
    """
    Random baseline: randomly assign labels based on training distribution.
    """
    
    name = "random"
    
    def __init__(self, label_distribution: Optional[Dict[int, float]] = None, seed: int = 42):
        """
        Args:
            label_distribution: Dict mapping label ID to probability.
                               If None, uses default distribution.
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        
        if label_distribution:
            self.labels = list(label_distribution.keys())
            self.weights = list(label_distribution.values())
        else:
            # Default: heavily weighted toward "O"
            self.labels = list(BIO_LABEL2ID.values())
            self.weights = [0.85] + [0.15 / 8] * 8  # 85% O, rest uniform
    
    def predict(self, example: Dict) -> List[int]:
        """Random prediction based on distribution."""
        return random.choices(self.labels, weights=self.weights, k=len(example["input_ids"]))


class KeywordBaseline(BaselineModel):
    """
    Keyword-based baseline: use Hebrew keywords to identify modifications.
    
    FIXED VERSION: Properly locates keywords in the tokenized text using
    offset_mapping to align character positions with token indices.
    
    This represents traditional rule-based NLP approaches.
    """
    
    name = "keyword"
    
    def __init__(
        self, 
        tokenizer=None, 
        keywords: Dict[str, List[str]] = None,
        model_name: str = "onlplab/alephbert-base"
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer (will load if None)
            keywords: Dict mapping aspect to keyword list
            model_name: Model name for tokenizer if not provided
        """
        self.keywords = keywords or ASPECT_KEYWORDS
        
        # Load tokenizer if not provided (needed for proper keyword location)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                print(f"⚠️  Could not load tokenizer: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
        
        # Sort keywords by length (longer first) to match longer phrases first
        for aspect in self.keywords:
            self.keywords[aspect] = sorted(self.keywords[aspect], key=len, reverse=True)
    
    def _find_keyword_positions(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Find all keyword occurrences in text.
        
        Returns:
            List of (start_char, end_char, aspect) tuples
        """
        found = []
        used_ranges = []  # Track used character ranges to avoid overlaps
        
        for aspect, keywords in self.keywords.items():
            for keyword in keywords:
                # Find all occurrences of this keyword
                start = 0
                while True:
                    idx = text.find(keyword, start)
                    if idx == -1:
                        break
                    
                    end_idx = idx + len(keyword)
                    
                    # Check if this range overlaps with already found keywords
                    overlaps = False
                    for used_start, used_end in used_ranges:
                        if not (end_idx <= used_start or idx >= used_end):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        found.append((idx, end_idx, aspect))
                        used_ranges.append((idx, end_idx))
                    
                    start = idx + 1
        
        # Sort by start position
        found.sort(key=lambda x: x[0])
        return found
    
    def _get_token_char_mapping(
        self, 
        text: str, 
        input_ids: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Get character offset mapping for tokens.
        
        Returns:
            List of (start_char, end_char) for each token
        """
        if self.tokenizer is None:
            return None
        
        # Re-tokenize with offset mapping
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=len(input_ids),
            padding="max_length",
            return_offsets_mapping=True,
        )
        
        return encoding.get("offset_mapping", None)
    
    def predict(self, example: Dict) -> List[int]:
        """
        Predict using keyword matching with proper token alignment.
        
        This FIXED version:
        1. Finds keywords in the original text
        2. Uses offset_mapping to locate which tokens contain those keywords
        3. Properly assigns B- and I- tags to the correct tokens
        """
        input_ids = example["input_ids"]
        labels = [BIO_LABEL2ID["O"]] * len(input_ids)
        
        # Get original text
        text = example.get("original_text", example.get("text", ""))
        
        # If no text available, try to decode from input_ids
        if not text and self.tokenizer is not None:
            text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        
        if not text:
            return labels
        
        # Find keyword positions in text
        keyword_positions = self._find_keyword_positions(text)
        
        if not keyword_positions:
            return labels
        
        # Get token-to-character mapping
        offsets = self._get_token_char_mapping(text, input_ids)
        
        if offsets is None:
            # Fallback: simple heuristic if we can't get proper offsets
            return self._fallback_predict(example, keyword_positions)
        
        # Map keywords to tokens
        for kw_start, kw_end, aspect in keyword_positions:
            b_tag = BIO_LABEL2ID[f"B-{aspect}"]
            i_tag = BIO_LABEL2ID[f"I-{aspect}"]
            
            first_token = True
            
            for token_idx, (tok_start, tok_end) in enumerate(offsets):
                # Skip special tokens (offset 0,0)
                if tok_start == 0 and tok_end == 0:
                    continue
                
                # Check if this token overlaps with the keyword
                if tok_start < kw_end and tok_end > kw_start:
                    # This token is part of the keyword span
                    if labels[token_idx] == BIO_LABEL2ID["O"]:  # Don't overwrite
                        if first_token:
                            labels[token_idx] = b_tag
                            first_token = False
                        else:
                            labels[token_idx] = i_tag
        
        return labels
    
    def _fallback_predict(
        self, 
        example: Dict, 
        keyword_positions: List[Tuple[int, int, str]]
    ) -> List[int]:
        """
        Fallback prediction when offset mapping is not available.
        Uses approximate token matching.
        """
        input_ids = example["input_ids"]
        labels = [BIO_LABEL2ID["O"]] * len(input_ids)
        
        if self.tokenizer is None:
            return labels
        
        # Get tokens as strings
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        for _, _, aspect in keyword_positions:
            b_tag = BIO_LABEL2ID[f"B-{aspect}"]
            i_tag = BIO_LABEL2ID[f"I-{aspect}"]
            
            # Find keyword tokens in the token list
            for kw in self.keywords[aspect]:
                kw_tokens = self.tokenizer.tokenize(kw)
                
                if not kw_tokens:
                    continue
                
                # Search for the keyword token sequence
                for i in range(len(tokens) - len(kw_tokens) + 1):
                    # Check if tokens match (handle ## prefix)
                    match = True
                    for j, kw_tok in enumerate(kw_tokens):
                        tok = tokens[i + j]
                        if tok is None or tok in ['[CLS]', '[SEP]', '[PAD]']:
                            match = False
                            break
                        # Normalize: remove ## prefix for comparison
                        tok_norm = tok.replace('##', '')
                        kw_tok_norm = kw_tok.replace('##', '')
                        if tok_norm != kw_tok_norm:
                            match = False
                            break
                    
                    if match:
                        # Tag these tokens
                        for j in range(len(kw_tokens)):
                            if labels[i + j] == BIO_LABEL2ID["O"]:
                                if j == 0:
                                    labels[i + j] = b_tag
                                else:
                                    labels[i + j] = i_tag
                        break  # Found this keyword, move to next
        
        return labels


# =============================================================================
# EVALUATION RUNNER
# =============================================================================

class BaselineEvaluator:
    """Run all baselines and compare results."""
    
    def __init__(
        self, 
        data_dir: str = "data/processed",
        model_name: str = "onlplab/alephbert-base"
    ):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.test_examples = self._load_test_data()
        
        # Load tokenizer for keyword baseline
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                print(f"⚠️  Could not load tokenizer: {e}")
    
    def _load_test_data(self) -> List[Dict]:
        """Load test dataset."""
        test_file = self.data_dir / "dataset_test.jsonl"
        
        if not test_file.exists():
            # Fallback to validation
            test_file = self.data_dir / "dataset_val.jsonl"
        
        if not test_file.exists():
            raise FileNotFoundError(f"No test data found in {self.data_dir}")
        
        examples = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(examples)} test examples from {test_file}")
        return examples
    
    def _get_label_distribution(self) -> Optional[Dict[int, float]]:
        """Calculate label distribution from training data."""
        train_file = self.data_dir / "dataset_train.jsonl"
        
        if not train_file.exists():
            return None
        
        counter = Counter()
        total = 0
        
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    for label in example["labels"]:
                        if label != -100:
                            counter[label] += 1
                            total += 1
                except:
                    continue
        
        if total > 0:
            dist = {k: v / total for k, v in counter.items()}
            print(f"Label distribution: O={dist.get(0, 0):.2%}, non-O={1-dist.get(0, 0):.2%}")
            return dist
        return None
    
    def run_all_baselines(self) -> Dict[str, BaselineResult]:
        """Run all baselines and return results."""
        results = {}
        
        print("\n" + "=" * 70)
        print("BASELINE EVALUATION")
        print("=" * 70)
        print(f"Test set size: {len(self.test_examples)} examples")
        print(f"Model: {self.model_name}")
        
        # 1. Majority baseline
        print("\n1. Running Majority baseline...")
        majority = MajorityBaseline()
        results["majority"] = majority.evaluate(self.test_examples)
        self._print_result(results["majority"])
        
        # 2. Random baseline
        print("\n2. Running Random baseline...")
        label_dist = self._get_label_distribution()
        random_bl = RandomBaseline(label_dist, seed=42)
        results["random"] = random_bl.evaluate(self.test_examples)
        self._print_result(results["random"])
        
        # 3. Keyword baseline (FIXED)
        print("\n3. Running Keyword baseline (fixed version)...")
        keyword_bl = KeywordBaseline(tokenizer=self.tokenizer)
        results["keyword"] = keyword_bl.evaluate(self.test_examples)
        self._print_result(results["keyword"])
        
        # Summary table
        self._print_summary(results)
        
        return results
    
    def _print_result(self, result: BaselineResult):
        """Print a single result."""
        print(f"   Token Accuracy: {result.token_accuracy:.4f}")
        print(f"   Entity F1:      {result.f1:.4f}")
        print(f"   Precision:      {result.precision:.4f}")
        print(f"   Recall:         {result.recall:.4f}")
    
    def _print_summary(self, results: Dict[str, BaselineResult]):
        """Print summary table."""
        print("\n" + "=" * 70)
        print("BASELINE SUMMARY")
        print("=" * 70)
        print(f"{'Model':<15} {'Token Acc.':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 70)
        
        for name, result in results.items():
            print(f"{name:<15} {result.token_accuracy:.4f}       {result.precision:.4f}       {result.recall:.4f}       {result.f1:.4f}")
        
        print("=" * 70)
        
        # Save results
        output_path = self.data_dir.parent / "results" / "baseline_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = {
            name: {
                "token_accuracy": r.token_accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
            }
            for name, r in results.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
        # Print note about mBERT baseline
        print("\n" + "-" * 70)
        print("NOTE: For mBERT baseline, run:")
        print("  python -m src.models.train_student \\")
        print("      --model bert-base-multilingual-cased \\")
        print("      --output-dir models/baselines/mbert \\")
        print("      --data-dir data/processed")
        print("-" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with processed data")
    parser.add_argument("--model", default="onlplab/alephbert-base", help="Tokenizer model name")
    
    args = parser.parse_args()
    
    if not SEQEVAL_AVAILABLE:
        print("ERROR: seqeval is required. Install with: pip install seqeval")
        return 1
    
    evaluator = BaselineEvaluator(args.data_dir, args.model)
    evaluator.run_all_baselines()
    
    return 0


if __name__ == "__main__":
    exit(main())