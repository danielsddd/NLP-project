"""
Evaluation Module
=================
Comprehensive evaluation metrics and error analysis.

Usage:
    python -m src.evaluation.evaluate --model-path models/checkpoints/student/best_model
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, field

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from seqeval.metrics import (
        f1_score, 
        precision_score, 
        recall_score, 
        classification_report,
    )
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False

try:
    import numpy as np
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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

ASPECTS = ["SUBSTITUTION", "QUANTITY", "TECHNIQUE", "ADDITION"]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EvaluationResult:
    """Comprehensive evaluation results."""
    # Overall metrics
    token_accuracy: float = 0.0
    entity_f1: float = 0.0
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    
    # Per-aspect metrics
    per_aspect_f1: Dict[str, float] = field(default_factory=dict)
    per_aspect_precision: Dict[str, float] = field(default_factory=dict)
    per_aspect_recall: Dict[str, float] = field(default_factory=dict)
    
    # Confusion analysis
    confusion_matrix: Optional[Dict] = None
    
    # Error analysis
    error_counts: Dict[str, int] = field(default_factory=dict)
    error_examples: List[Dict] = field(default_factory=list)
    
    # Confidence intervals (if bootstrap performed)
    f1_ci_lower: Optional[float] = None
    f1_ci_upper: Optional[float] = None


@dataclass
class PredictionExample:
    """A single prediction for analysis."""
    text: str
    true_labels: List[str]
    pred_labels: List[str]
    tokens: List[str]
    true_spans: List[Dict]
    pred_spans: List[Dict]
    is_correct: bool


# =============================================================================
# EVALUATION CLASS
# =============================================================================

class ModelEvaluator:
    """
    Comprehensive model evaluation.
    
    Usage:
        evaluator = ModelEvaluator(model_path="models/checkpoints/student/best_model")
        results = evaluator.evaluate("data/processed/dataset_test.jsonl")
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Args:
            model_path: Path to saved model (loads from here)
            model: Pre-loaded model (alternative to model_path)
            tokenizer: Pre-loaded tokenizer
        """
        if model_path:
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.model = None
            self.tokenizer = None
        
        if self.model:
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def evaluate(
        self,
        test_file: str,
        output_dir: Optional[str] = None,
        compute_bootstrap: bool = False,
        bootstrap_iterations: int = 1000,
        max_error_examples: int = 50,
    ) -> EvaluationResult:
        """
        Run comprehensive evaluation.
        
        Args:
            test_file: Path to test JSONL file
            output_dir: Directory to save results
            compute_bootstrap: Whether to compute bootstrap confidence intervals
            bootstrap_iterations: Number of bootstrap samples
            max_error_examples: Maximum error examples to store
            
        Returns:
            EvaluationResult with all metrics
        """
        self.logger.info(f"Evaluating on {test_file}")
        
        # Load test data
        examples = self._load_test_data(test_file)
        self.logger.info(f"Loaded {len(examples)} test examples")
        
        # Get predictions
        all_true_labels = []
        all_pred_labels = []
        prediction_examples = []
        
        for example in examples:
            if self.model:
                pred_labels = self._predict_example(example)
            else:
                # Use gold labels for analysis (when no model)
                pred_labels = example["labels"]
            
            true_labels = example["labels"]
            
            # Convert to strings
            true_str = [BIO_ID2LABEL.get(l, "O") for l in true_labels if l != -100]
            pred_str = [BIO_ID2LABEL.get(l, "O") for l in pred_labels[:len(true_str)]]
            
            all_true_labels.append(true_str)
            all_pred_labels.append(pred_str)
            
            # Store for error analysis
            tokens = self.tokenizer.convert_ids_to_tokens(example["input_ids"]) if self.tokenizer else []
            prediction_examples.append(PredictionExample(
                text=example.get("text", ""),
                true_labels=true_str,
                pred_labels=pred_str,
                tokens=tokens,
                true_spans=self._labels_to_spans(true_str),
                pred_spans=self._labels_to_spans(pred_str),
                is_correct=true_str == pred_str,
            ))
        
        # Compute metrics
        result = self._compute_metrics(all_true_labels, all_pred_labels)
        
        # Error analysis
        result.error_counts, result.error_examples = self._analyze_errors(
            prediction_examples, max_examples=max_error_examples
        )
        
        # Bootstrap confidence intervals
        if compute_bootstrap and SCIPY_AVAILABLE:
            result.f1_ci_lower, result.f1_ci_upper = self._bootstrap_ci(
                all_true_labels, all_pred_labels, bootstrap_iterations
            )
        
        # Save results
        if output_dir:
            self._save_results(result, output_dir)
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _load_test_data(self, test_file: str) -> List[Dict]:
        """Load test data from JSONL."""
        examples = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return examples
    
    def _predict_example(self, example: Dict) -> List[int]:
        """Get model predictions for an example."""
        with torch.no_grad():
            input_ids = torch.tensor([example["input_ids"]], device=self.device)
            attention_mask = torch.tensor([example["attention_mask"]], device=self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
        
        return predictions
    
    def _compute_metrics(
        self, 
        all_true: List[List[str]], 
        all_pred: List[List[str]]
    ) -> EvaluationResult:
        """Compute all evaluation metrics."""
        result = EvaluationResult()
        
        # Token-level accuracy
        total_tokens = 0
        correct_tokens = 0
        for true_seq, pred_seq in zip(all_true, all_pred):
            for t, p in zip(true_seq, pred_seq):
                total_tokens += 1
                if t == p:
                    correct_tokens += 1
        result.token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # Entity-level metrics (seqeval)
        if SEQEVAL_AVAILABLE:
            result.entity_f1 = f1_score(all_true, all_pred)
            result.entity_precision = precision_score(all_true, all_pred)
            result.entity_recall = recall_score(all_true, all_pred)
            
            # Per-aspect metrics
            report = classification_report(all_true, all_pred, output_dict=True)
            
            for aspect in ASPECTS:
                if aspect in report:
                    result.per_aspect_f1[aspect] = report[aspect].get("f1-score", 0.0)
                    result.per_aspect_precision[aspect] = report[aspect].get("precision", 0.0)
                    result.per_aspect_recall[aspect] = report[aspect].get("recall", 0.0)
                else:
                    result.per_aspect_f1[aspect] = 0.0
                    result.per_aspect_precision[aspect] = 0.0
                    result.per_aspect_recall[aspect] = 0.0
        
        # Confusion matrix
        result.confusion_matrix = self._compute_confusion_matrix(all_true, all_pred)
        
        return result
    
    def _compute_confusion_matrix(
        self, 
        all_true: List[List[str]], 
        all_pred: List[List[str]]
    ) -> Dict:
        """Compute confusion matrix for aspects."""
        # Map to aspects (ignore B/I distinction)
        def to_aspect(label):
            if label == "O":
                return "O"
            return label.split("-")[1] if "-" in label else label
        
        matrix = defaultdict(lambda: defaultdict(int))
        
        for true_seq, pred_seq in zip(all_true, all_pred):
            for t, p in zip(true_seq, pred_seq):
                true_aspect = to_aspect(t)
                pred_aspect = to_aspect(p)
                matrix[true_aspect][pred_aspect] += 1
        
        return {k: dict(v) for k, v in matrix.items()}
    
    def _labels_to_spans(self, labels: List[str]) -> List[Dict]:
        """Convert BIO labels to spans."""
        spans = []
        current_span = None
        
        for i, label in enumerate(labels):
            if label.startswith("B-"):
                if current_span:
                    spans.append(current_span)
                current_span = {
                    "aspect": label[2:],
                    "start": i,
                    "end": i + 1,
                }
            elif label.startswith("I-"):
                if current_span and label[2:] == current_span["aspect"]:
                    current_span["end"] = i + 1
                else:
                    if current_span:
                        spans.append(current_span)
                    current_span = {
                        "aspect": label[2:],
                        "start": i,
                        "end": i + 1,
                    }
            else:
                if current_span:
                    spans.append(current_span)
                    current_span = None
        
        if current_span:
            spans.append(current_span)
        
        return spans
    
    def _analyze_errors(
        self, 
        examples: List[PredictionExample],
        max_examples: int = 50,
    ) -> Tuple[Dict[str, int], List[Dict]]:
        """Analyze prediction errors."""
        error_counts = {
            "span_boundary": 0,      # Wrong span boundaries
            "aspect_confusion": 0,   # Wrong aspect
            "false_negative": 0,     # Missed modification
            "false_positive": 0,     # Hallucinated modification
            "correct": 0,
        }
        
        error_examples = []
        
        for ex in examples:
            if ex.is_correct:
                error_counts["correct"] += 1
                continue
            
            true_set = set((s["aspect"], s["start"], s["end"]) for s in ex.true_spans)
            pred_set = set((s["aspect"], s["start"], s["end"]) for s in ex.pred_spans)
            
            # Categorize errors
            for span in ex.true_spans:
                key = (span["aspect"], span["start"], span["end"])
                if key not in pred_set:
                    # Check if partial match
                    partial = any(
                        p["aspect"] == span["aspect"] and 
                        (p["start"] <= span["end"] and p["end"] >= span["start"])
                        for p in ex.pred_spans
                    )
                    if partial:
                        error_counts["span_boundary"] += 1
                    else:
                        # Check aspect confusion
                        aspect_match = any(
                            p["start"] <= span["end"] and p["end"] >= span["start"]
                            for p in ex.pred_spans
                        )
                        if aspect_match:
                            error_counts["aspect_confusion"] += 1
                        else:
                            error_counts["false_negative"] += 1
            
            for span in ex.pred_spans:
                key = (span["aspect"], span["start"], span["end"])
                if key not in true_set:
                    # Check if it's a complete hallucination
                    overlap = any(
                        t["start"] <= span["end"] and t["end"] >= span["start"]
                        for t in ex.true_spans
                    )
                    if not overlap:
                        error_counts["false_positive"] += 1
            
            # Store example
            if len(error_examples) < max_examples:
                error_examples.append({
                    "text": ex.text[:200],
                    "true_spans": ex.true_spans,
                    "pred_spans": ex.pred_spans,
                })
        
        return error_counts, error_examples
    
    def _bootstrap_ci(
        self,
        all_true: List[List[str]],
        all_pred: List[List[str]],
        n_iterations: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for F1."""
        n = len(all_true)
        f1_scores = []
        
        for _ in range(n_iterations):
            # Sample with replacement
            indices = np.random.choice(n, n, replace=True)
            sample_true = [all_true[i] for i in indices]
            sample_pred = [all_pred[i] for i in indices]
            
            f1 = f1_score(sample_true, sample_pred)
            f1_scores.append(f1)
        
        # Compute percentiles
        alpha = 1 - confidence
        lower = np.percentile(f1_scores, alpha / 2 * 100)
        upper = np.percentile(f1_scores, (1 - alpha / 2) * 100)
        
        return lower, upper
    
    def _save_results(self, result: EvaluationResult, output_dir: str):
        """Save evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_dict = {
            "token_accuracy": result.token_accuracy,
            "entity_f1": result.entity_f1,
            "entity_precision": result.entity_precision,
            "entity_recall": result.entity_recall,
            "per_aspect_f1": result.per_aspect_f1,
            "per_aspect_precision": result.per_aspect_precision,
            "per_aspect_recall": result.per_aspect_recall,
            "error_counts": result.error_counts,
            "confusion_matrix": result.confusion_matrix,
        }
        
        if result.f1_ci_lower is not None:
            results_dict["f1_ci_95"] = [result.f1_ci_lower, result.f1_ci_upper]
        
        with open(output_path / "evaluation_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save error examples
        with open(output_path / "error_examples.json", 'w', encoding='utf-8') as f:
            json.dump(result.error_examples, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def _print_summary(self, result: EvaluationResult):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\nOVERALL METRICS:")
        print(f"  Token Accuracy:    {result.token_accuracy:.4f}")
        print(f"  Entity F1:         {result.entity_f1:.4f}")
        print(f"  Entity Precision:  {result.entity_precision:.4f}")
        print(f"  Entity Recall:     {result.entity_recall:.4f}")
        
        if result.f1_ci_lower is not None:
            print(f"  F1 95% CI:         [{result.f1_ci_lower:.4f}, {result.f1_ci_upper:.4f}]")
        
        print(f"\nPER-ASPECT F1:")
        for aspect in ASPECTS:
            f1 = result.per_aspect_f1.get(aspect, 0.0)
            print(f"  {aspect:<15} {f1:.4f}")
        
        print(f"\nERROR ANALYSIS:")
        for error_type, count in result.error_counts.items():
            print(f"  {error_type:<20} {count}")
        
        print("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--test-file", default="data/processed/dataset_test.jsonl", help="Test file")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--bootstrap", action="store_true", help="Compute bootstrap CI")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(model_path=args.model_path)
    evaluator.evaluate(
        test_file=args.test_file,
        output_dir=args.output_dir,
        compute_bootstrap=args.bootstrap,
    )


if __name__ == "__main__":
    main()
