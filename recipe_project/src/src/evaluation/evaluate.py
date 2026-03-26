#!/usr/bin/env python3
"""
Evaluation — v4
Evaluates a trained model with entity F1, per-aspect breakdown, error analysis, bootstrap CI.

Usage:
    python -m src.evaluation.evaluate \
        --model-path models/checkpoints/student/best_model \
        --test-file data/processed/test.jsonl \
        --bootstrap
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

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
# INFERENCE
# =============================================================================

def predict_batch(model, examples, device, batch_size=32):
    """Run model inference on all examples. Returns list of predicted label ID lists."""
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

def to_tag_sequences(pred_ids_list, label_ids_list, id2label):
    """Convert ID sequences to tag string sequences, skipping -100."""
    all_true, all_pred = [], []
    for preds, labels in zip(pred_ids_list, label_ids_list):
        true_seq, pred_seq = [], []
        for p, l in zip(preds, labels):
            if l == -100:
                continue
            true_seq.append(id2label.get(l, "O"))
            pred_seq.append(id2label.get(p, "O"))
        all_true.append(true_seq)
        all_pred.append(pred_seq)
    return all_true, all_pred

# =============================================================================
# SPAN EXTRACTION (for error analysis)
# =============================================================================

def labels_to_spans(tag_seq):
    """Convert BIO tag sequence to list of {aspect, start, end}."""
    spans, current = [], None
    for i, tag in enumerate(tag_seq):
        if tag.startswith("B-"):
            if current:
                spans.append(current)
            current = {"aspect": tag[2:], "start": i, "end": i}
        elif tag.startswith("I-"):
            if current and current["aspect"] == tag[2:]:
                current["end"] = i
            else:
                if current:
                    spans.append(current)
                current = {"aspect": tag[2:], "start": i, "end": i}
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans

# =============================================================================
# ERROR ANALYSIS
# =============================================================================

def analyze_errors(all_true, all_pred, texts=None, max_examples=20):
    """Categorize prediction errors into types."""
    counts = Counter({"span_boundary": 0, "aspect_confusion": 0,
                      "false_negative": 0, "false_positive": 0, "correct": 0})
    examples = []

    for idx, (true_seq, pred_seq) in enumerate(zip(all_true, all_pred)):
        true_spans = labels_to_spans(true_seq)
        pred_spans = labels_to_spans(pred_seq)

        true_set = {(s["aspect"], s["start"], s["end"]) for s in true_spans}
        pred_set = {(s["aspect"], s["start"], s["end"]) for s in pred_spans}

        if true_set == pred_set:
            counts["correct"] += 1
            continue

        for ts in true_spans:
            tk = (ts["aspect"], ts["start"], ts["end"])
            if tk in pred_set:
                continue
            # Check partial overlap
            overlap = [ps for ps in pred_spans
                       if ps["start"] <= ts["end"] and ps["end"] >= ts["start"]]
            if overlap:
                if any(ps["aspect"] == ts["aspect"] for ps in overlap):
                    counts["span_boundary"] += 1
                else:
                    counts["aspect_confusion"] += 1
            else:
                counts["false_negative"] += 1

        for ps in pred_spans:
            pk = (ps["aspect"], ps["start"], ps["end"])
            if pk not in true_set:
                overlap = [ts for ts in true_spans
                           if ts["start"] <= ps["end"] and ts["end"] >= ps["start"]]
                if not overlap:
                    counts["false_positive"] += 1

        if len(examples) < max_examples:
            examples.append({
                "index": idx,
                "text": (texts[idx][:150] + "...") if texts and idx < len(texts) else "",
                "true_spans": true_spans,
                "pred_spans": pred_spans,
            })

    return dict(counts), examples

# =============================================================================
# BOOTSTRAP CI
# =============================================================================

def bootstrap_f1_ci(all_true, all_pred, n_iter=1000, confidence=0.95):
    """Compute bootstrap confidence interval for entity F1."""
    n = len(all_true)
    scores = []
    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        sample_t = [all_true[i] for i in idx]
        sample_p = [all_pred[i] for i in idx]
        scores.append(f1_score(sample_t, sample_p))

    alpha = 1 - confidence
    lo = float(np.percentile(scores, alpha / 2 * 100))
    hi = float(np.percentile(scores, (1 - alpha / 2) * 100))
    return lo, hi

# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate(model_path, test_file, output_dir="results", compute_bootstrap=False):
    """Full evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Determine label mapping from model config
    id2label = model.config.id2label if hasattr(model.config, "id2label") else BIO_ID2LABEL
    if isinstance(id2label, dict):
        id2label = {int(k): v for k, v in id2label.items()}

    # Load test data
    examples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} test examples")

    # Predict
    pred_ids = predict_batch(model, examples, device)
    label_ids = [ex["labels"] for ex in examples]
    texts = [ex.get("text", "") for ex in examples]

    # Convert to tags
    all_true, all_pred = to_tag_sequences(pred_ids, label_ids, id2label)

    # Compute metrics
    f1 = f1_score(all_true, all_pred)
    p = precision_score(all_true, all_pred)
    r = recall_score(all_true, all_pred)

    # Token accuracy
    correct, total = 0, 0
    for ts, ps in zip(all_true, all_pred):
        for t, pr in zip(ts, ps):
            total += 1
            if t == pr:
                correct += 1
    acc = correct / total if total else 0

    # Per-aspect
    report = classification_report(all_true, all_pred, output_dict=True)
    per_aspect = {}
    for aspect in ASPECTS:
        if aspect in report:
            per_aspect[aspect] = {
                "f1": report[aspect]["f1-score"],
                "precision": report[aspect]["precision"],
                "recall": report[aspect]["recall"],
            }

    # Error analysis
    error_counts, error_examples = analyze_errors(all_true, all_pred, texts)

    # Bootstrap
    ci = None
    if compute_bootstrap:
        lo, hi = bootstrap_f1_ci(all_true, all_pred)
        ci = {"lower": lo, "upper": hi}
        print(f"Bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")

    # Print results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Token Accuracy:  {acc:.4f}")
    print(f"  Entity F1:       {f1:.4f}")
    print(f"  Entity P:        {p:.4f}")
    print(f"  Entity R:        {r:.4f}")
    if ci:
        print(f"  F1 95% CI:       [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    print(f"\n  Per-Aspect F1:")
    for aspect, m in per_aspect.items():
        print(f"    {aspect:<15s}  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
    print(f"\n  Error Analysis:")
    for etype, count in error_counts.items():
        print(f"    {etype:<20s}  {count}")
    print(f"{'='*60}")

    # Full classification report
    print("\nFull classification report:")
    print(classification_report(all_true, all_pred))

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "token_accuracy": acc,
        "entity_f1": f1, "entity_precision": p, "entity_recall": r,
        "per_aspect": per_aspect,
        "error_counts": error_counts,
        "bootstrap_ci": ci,
        "num_examples": len(examples),
        "model_path": str(model_path),
    }
    with open(out / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    with open(out / "error_examples.json", 'w', encoding='utf-8') as f:
        json.dump(error_examples, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out}")

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--test-file", default="data/processed/test.jsonl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--bootstrap", action="store_true", help="Compute bootstrap CI")
    args = parser.parse_args()

    evaluate(args.model_path, args.test_file, args.output_dir, args.bootstrap)

if __name__ == "__main__":
    main()