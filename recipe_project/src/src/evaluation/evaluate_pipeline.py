#!/usr/bin/env python3
"""
Two-Step Pipeline Evaluation
============================
Chains the binary classifier (Step 1) with the token extractor (Step 2)
to compute the TRUE combined pipeline F1 score.

Flow:
    For each test example:
        1. Classifier predicts: has_modification? (yes/no)
        2. If YES → Extractor predicts BIO tags
        3. If NO  → All tokens get "O" (no extraction attempted)

Usage:
    python -m src.evaluation.evaluate_pipeline \
        --classifier-path models/checkpoints/classifier_dictabert/best_model \
        --extractor-path models/checkpoints/dictabert_twostep/best_model \
        --test-file data/processed_dictabert/test.jsonl \
        --output-dir results/pipeline_dictabert \
        --bootstrap
"""

import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# =============================================================================
# LABEL SCHEMA
# =============================================================================

BIO_ID2LABEL = {
    0: "O", 1: "B-SUBSTITUTION", 2: "I-SUBSTITUTION",
    3: "B-QUANTITY", 4: "I-QUANTITY",
    5: "B-TECHNIQUE", 6: "I-TECHNIQUE",
    7: "B-ADDITION", 8: "I-ADDITION",
}
ASPECTS = ["SUBSTITUTION", "QUANTITY", "TECHNIQUE", "ADDITION"]

# =============================================================================
# INFERENCE HELPERS
# =============================================================================

def classify_batch(model, examples, device, batch_size=64):
    """Run classifier on examples. Returns list of 0/1 predictions."""
    model.eval()
    all_preds = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], device=device)
        attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
        all_preds.extend(preds)
    return all_preds


def extract_batch(model, examples, device, batch_size=32):
    """Run token classifier on examples. Returns list of predicted label ID lists."""
    model.eval()
    all_preds = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], device=device)
        attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
        all_preds.extend(preds)
    return all_preds


def to_tag_sequences(pred_ids, label_ids, id2label):
    """Convert ID sequences to tag strings, skipping -100."""
    true_seq, pred_seq = [], []
    for p, l in zip(pred_ids, label_ids):
        if l == -100:
            continue
        true_seq.append(id2label.get(l, "O"))
        pred_seq.append(id2label.get(p, "O"))
    return true_seq, pred_seq


def make_all_o(label_ids):
    """Create all-O prediction for examples the classifier rejects."""
    return [0 if l != -100 else -100 for l in label_ids]


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# =============================================================================

def bootstrap_f1_ci(all_true, all_pred, n_bootstrap=1000, seed=42):
    """Compute 95% bootstrap CI for entity F1."""
    rng = np.random.RandomState(seed)
    n = len(all_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        t = [all_true[i] for i in idx]
        p = [all_pred[i] for i in idx]
        try:
            scores.append(f1_score(t, p))
        except Exception:
            scores.append(0.0)
    scores.sort()
    lo = scores[int(0.025 * n_bootstrap)]
    hi = scores[int(0.975 * n_bootstrap)]
    return lo, hi


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_pipeline(classifier_path, extractor_path, test_file,
                      output_dir, compute_bootstrap=False):
    """Run the full two-step pipeline evaluation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print(f"\nLoading classifier: {classifier_path}")
    classifier = AutoModelForSequenceClassification.from_pretrained(
        classifier_path, trust_remote_code=True
    ).to(device)

    print(f"Loading extractor: {extractor_path}")
    extractor = AutoModelForTokenClassification.from_pretrained(
        extractor_path, trust_remote_code=True
    ).to(device)

    # Determine label mapping from extractor config
    id2label = extractor.config.id2label if hasattr(extractor.config, "id2label") else BIO_ID2LABEL
    if isinstance(id2label, dict):
        id2label = {int(k): v for k, v in id2label.items()}

    # Load test data
    examples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} test examples")

    # Compute ground truth
    true_has_mod = []
    for ex in examples:
        has_entity = any(l not in (-100, 0) for l in ex["labels"])
        true_has_mod.append(1 if has_entity else 0)

    n_pos = sum(true_has_mod)
    n_neg = len(true_has_mod) - n_pos
    print(f"  Positive (has modification): {n_pos}")
    print(f"  Negative (no modification):  {n_neg}")

    # =========================================================================
    # STEP 1: Classify all examples
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 1: Binary Classification")
    print(f"{'='*60}")

    clf_preds = classify_batch(classifier, examples, device)

    tp = sum(1 for p, t in zip(clf_preds, true_has_mod) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(clf_preds, true_has_mod) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(clf_preds, true_has_mod) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(clf_preds, true_has_mod) if p == 0 and t == 0)

    clf_acc = (tp + tn) / len(clf_preds)
    clf_p = tp / (tp + fp) if (tp + fp) > 0 else 0
    clf_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    clf_f1 = 2 * clf_p * clf_r / (clf_p + clf_r) if (clf_p + clf_r) > 0 else 0

    predicted_positive = sum(clf_preds)
    print(f"  Classified as modification:     {predicted_positive}")
    print(f"  Classified as no modification:  {len(clf_preds) - predicted_positive}")
    print(f"  Accuracy:   {clf_acc:.4f}")
    print(f"  Precision:  {clf_p:.4f}")
    print(f"  Recall:     {clf_r:.4f}")
    print(f"  F1:         {clf_f1:.4f}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # =========================================================================
    # STEP 2: Extract only from classifier-positive examples
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 2: Token Extraction (classifier-positive only)")
    print(f"{'='*60}")

    pos_indices = [i for i, p in enumerate(clf_preds) if p == 1]
    neg_indices = [i for i, p in enumerate(clf_preds) if p == 0]

    print(f"  Sending {len(pos_indices)} examples to extractor")
    print(f"  Auto-rejecting {len(neg_indices)} examples (all-O)")

    pos_examples = [examples[i] for i in pos_indices]
    if pos_examples:
        ext_preds = extract_batch(extractor, pos_examples, device)
    else:
        ext_preds = []

    # =========================================================================
    # COMBINE: Merge predictions
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"COMBINED PIPELINE RESULTS")
    print(f"{'='*60}")

    all_true_tags = []
    all_pred_tags = []

    ext_idx = 0
    for i, ex in enumerate(examples):
        gold_labels = ex["labels"]

        if clf_preds[i] == 1 and ext_idx < len(ext_preds):
            pred_labels = ext_preds[ext_idx]
            ext_idx += 1
        else:
            pred_labels = make_all_o(gold_labels)

        true_seq, pred_seq = to_tag_sequences(pred_labels, gold_labels, id2label)
        all_true_tags.append(true_seq)
        all_pred_tags.append(pred_seq)

    combined_f1 = f1_score(all_true_tags, all_pred_tags)
    combined_p = precision_score(all_true_tags, all_pred_tags)
    combined_r = recall_score(all_true_tags, all_pred_tags)

    correct, total = 0, 0
    for ts, ps in zip(all_true_tags, all_pred_tags):
        for t, pr in zip(ts, ps):
            total += 1
            if t == pr:
                correct += 1
    token_acc = correct / total if total else 0

    print(f"\n  *** COMBINED PIPELINE F1: {combined_f1:.4f} ***")
    print(f"  Entity P:        {combined_p:.4f}")
    print(f"  Entity R:        {combined_r:.4f}")
    print(f"  Token Accuracy:  {token_acc:.4f}")

    report_dict = classification_report(all_true_tags, all_pred_tags, output_dict=True)
    per_aspect = {}
    print(f"\n  Per-Aspect F1:")
    for aspect in ASPECTS:
        if aspect in report_dict:
            per_aspect[aspect] = {
                "f1": report_dict[aspect]["f1-score"],
                "precision": report_dict[aspect]["precision"],
                "recall": report_dict[aspect]["recall"],
            }
            print(f"    {aspect:<15s}  F1={report_dict[aspect]['f1-score']:.3f}  "
                  f"P={report_dict[aspect]['precision']:.3f}  "
                  f"R={report_dict[aspect]['recall']:.3f}")

    print(f"\n  Full Classification Report:")
    print(classification_report(all_true_tags, all_pred_tags))

    ci = None
    if compute_bootstrap:
        lo, hi = bootstrap_f1_ci(all_true_tags, all_pred_tags)
        ci = {"lower": lo, "upper": hi}
        print(f"  Bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")

    # =========================================================================
    # ERROR ANALYSIS
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS")
    print(f"{'='*60}")

    print(f"  Classifier false negatives: {fn}")
    print(f"    (modifications missed because classifier said NO)")
    print(f"  Classifier false positives: {fp}")
    print(f"    (empty comments sent to extractor unnecessarily)")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "pipeline": "two_step",
        "classifier_path": str(classifier_path),
        "extractor_path": str(extractor_path),
        "test_file": str(test_file),
        "num_examples": len(examples),
        "classifier": {
            "accuracy": clf_acc, "precision": clf_p,
            "recall": clf_r, "f1": clf_f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "combined": {
            "entity_f1": combined_f1, "entity_precision": combined_p,
            "entity_recall": combined_r, "token_accuracy": token_acc,
            "per_aspect": per_aspect, "bootstrap_ci": ci,
        },
    }

    with open(out / "pipeline_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {out / 'pipeline_results.json'}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate two-step pipeline")
    parser.add_argument("--classifier-path", required=True)
    parser.add_argument("--extractor-path", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", default="results/pipeline")
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()

    evaluate_pipeline(
        args.classifier_path, args.extractor_path,
        args.test_file, args.output_dir, args.bootstrap,
    )

if __name__ == "__main__":
    main()
