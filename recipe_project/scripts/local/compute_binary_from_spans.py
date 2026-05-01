#!/usr/bin/env python3
"""
Compute binary classification metrics (has_modification yes/no)
from existing span model predictions.

If a model predicted ANY entity span in an example → has_modification=True.
This derives thread-level binary F1 without training a separate classifier.

Usage:
    python scripts/local/compute_binary_from_spans.py \
        --ckpt-dir models/checkpoints/dictabert_crf/P10v3_fixed_weights \
        --test-file data/processed_thread_aware_io/test.jsonl \
        --output-dir results/dictabert_crf/P10v3_fixed_weights/binary \
        --model-name dicta-il/dictabert

    # Or batch mode (all results):
    python scripts/local/compute_binary_from_spans.py --batch
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np

try:
    from sklearn.metrics import (
        classification_report, f1_score, precision_score,
        recall_score, accuracy_score, confusion_matrix
    )
except ImportError:
    print("ERROR: sklearn required. pip install scikit-learn")
    sys.exit(1)


# =====================================================================
# SINGLE MODEL EVALUATION
# =====================================================================

def compute_binary_from_checkpoint(ckpt_dir, test_file, output_dir, model_name):
    """Run inference and derive binary metrics from span predictions."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from src.models.joint_model import BertCRFModel

    IO_ID2LABEL = {0: "O", 1: "I-SUBSTITUTION", 2: "I-QUANTITY", 3: "I-TECHNIQUE", 4: "I-ADDITION"}
    BIO_ID2LABEL = {
        0: "O", 1: "B-SUBSTITUTION", 2: "I-SUBSTITUTION",
        3: "B-QUANTITY", 4: "I-QUANTITY", 5: "B-TECHNIQUE",
        6: "I-TECHNIQUE", 7: "B-ADDITION", 8: "I-ADDITION",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt_path = Path(ckpt_dir) / "best_model.pt"
    if not ckpt_path.exists():
        print(f"  SKIP: no best_model.pt at {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Auto-detect num_labels
    state_dict = ckpt["model_state_dict"]
    detected_labels = None
    for key in state_dict:
        if "classifier" in key or "emission" in key or "hidden2tag" in key:
            shape = state_dict[key].shape
            if len(shape) >= 1:
                detected_labels = shape[0]
                break
    if detected_labels is None:
        for key in state_dict:
            if "crf" in key and "transitions" in key:
                detected_labels = state_dict[key].shape[0]
                break
    num_labels = detected_labels if detected_labels else 9

    model = BertCRFModel(model_name=model_name, num_labels=num_labels, dropout_rate=0.1)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    # Load test data
    examples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    # Run inference
    gold_binary = []
    pred_binary = []

    batch_size = 32
    for start in range(0, len(examples), batch_size):
        batch_exs = examples[start:start + batch_size]

        input_ids = torch.tensor([ex["input_ids"] for ex in batch_exs], dtype=torch.long).to(device)
        attention_mask = torch.tensor([ex["attention_mask"] for ex in batch_exs], dtype=torch.long).to(device)

        with torch.no_grad():
            pred_sequences = model(input_ids, attention_mask)

        for i, (pred_seq, ex) in enumerate(zip(pred_sequences, batch_exs)):
            labels = ex["labels"]

            # Gold: any non-O, non-special label
            has_gold_entity = any(l not in (-100, 0) for l in labels)
            gold_binary.append(1 if has_gold_entity else 0)

            # Pred: any non-O prediction at a valid (non -100) position
            has_pred_entity = False
            for j, (p, g) in enumerate(zip(pred_seq, labels)):
                if g == -100:
                    continue
                if p != 0:
                    has_pred_entity = True
                    break
            pred_binary.append(1 if has_pred_entity else 0)

    # Compute metrics
    f1 = f1_score(gold_binary, pred_binary, pos_label=1)
    prec = precision_score(gold_binary, pred_binary, pos_label=1)
    rec = recall_score(gold_binary, pred_binary, pos_label=1)
    acc = accuracy_score(gold_binary, pred_binary)
    cm = confusion_matrix(gold_binary, pred_binary)
    report = classification_report(gold_binary, pred_binary,
                                   target_names=["no_modification", "has_modification"])

    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot_f1s = []
    indices = list(range(len(gold_binary)))
    for _ in range(1000):
        sample = rng.choice(indices, size=len(indices), replace=True)
        s_gold = [gold_binary[i] for i in sample]
        s_pred = [pred_binary[i] for i in sample]
        try:
            boot_f1s.append(f1_score(s_gold, s_pred, pos_label=1))
        except:
            boot_f1s.append(0.0)
    ci_low = float(np.percentile(boot_f1s, 2.5))
    ci_high = float(np.percentile(boot_f1s, 97.5))

    results = {
        "binary_f1": round(f1, 4),
        "binary_precision": round(prec, 4),
        "binary_recall": round(rec, 4),
        "binary_accuracy": round(acc, 4),
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "n_examples": len(gold_binary),
        "n_positive_gold": sum(gold_binary),
        "n_positive_pred": sum(pred_binary),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "ckpt_dir": str(ckpt_dir),
        "test_file": str(test_file),
    }

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "binary_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  BINARY RESULTS: {Path(ckpt_dir).name}")
    print(f"{'='*60}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  95% CI:    [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Gold pos:  {sum(gold_binary)}/{len(gold_binary)}")
    print(f"  Pred pos:  {sum(pred_binary)}/{len(gold_binary)}")
    print(f"\n{report}")
    print(f"  Saved to: {out_path / 'binary_results.json'}")
    print(f"{'='*60}")

    return results


# =====================================================================
# BATCH MODE
# =====================================================================

def batch_compute():
    """Compute binary metrics for all models that have checkpoints."""
    configs = [
        ("dictabert_crf/P2_crf_weighted",
         "models/checkpoints/dictabert_crf/P2_crf_weighted" if os.path.exists("models/checkpoints/dictabert_crf/P2_crf_weighted/best_model.pt")
         else "results/dictabert_crf/P2_crf_weighted",
         "data/processed_v2/test.jsonl", "dicta-il/dictabert"),
        ("dictabert_crf/P10v3_fixed_weights",
         "models/checkpoints/dictabert_crf/P10v3_fixed_weights",
         "data/processed_thread_aware_io/test.jsonl", "dicta-il/dictabert"),
        ("dictabert_crf/P10v4_focal",
         "models/checkpoints/dictabert_crf/P10v4_focal",
         "data/processed_thread_aware_io/test.jsonl", "dicta-il/dictabert"),
        ("dictabert_large_crf/P2_crf_weighted",
         "models/checkpoints/dictabert_large_crf/P2_crf_weighted",
         "data/processed_v2/test.jsonl", "dicta-il/dictabert-large"),
        ("dictabert_large_crf/P10_large_crf_thread_io",
         "models/checkpoints/dictabert_large_crf/P10_large_crf_thread_io",
         "data/processed_thread_aware_io/test.jsonl", "dicta-il/dictabert-large"),
    ]

    print("=" * 70)
    print("BINARY CLASSIFICATION FROM SPAN PREDICTIONS — BATCH MODE")
    print("=" * 70)

    summary = []
    for name, ckpt, test, model in configs:
        print(f"\n>>> {name}")
        result = compute_binary_from_checkpoint(
            ckpt, test, f"results/{name}/binary", model
        )
        if result:
            summary.append((name, result))

    if summary:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"{'Config':<45} {'F1':>6} {'Prec':>6} {'Rec':>6}")
        print("-" * 70)
        for name, r in summary:
            print(f"{name:<45} {r['binary_f1']:>6.4f} {r['binary_precision']:>6.4f} {r['binary_recall']:>6.4f}")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Derive binary (has_modification) metrics from span model predictions"
    )
    parser.add_argument("--ckpt-dir", help="Checkpoint directory (with best_model.pt)")
    parser.add_argument("--test-file", help="Test JSONL file")
    parser.add_argument("--output-dir", help="Where to save binary_results.json")
    parser.add_argument("--model-name", default="dicta-il/dictabert")
    parser.add_argument("--batch", action="store_true",
                        help="Run on all known model configs")
    args = parser.parse_args()

    if args.batch:
        batch_compute()
    elif args.ckpt_dir and args.test_file and args.output_dir:
        compute_binary_from_checkpoint(
            args.ckpt_dir, args.test_file, args.output_dir, args.model_name
        )
    else:
        parser.print_help()
        print("\nUse --batch for all models, or provide --ckpt-dir, --test-file, --output-dir")


if __name__ == "__main__":
    main()
