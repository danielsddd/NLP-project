#!/usr/bin/env python3
"""
Student Model Training — v4 (FIXED)
Fine-tunes AlephBERT (or any BERT) for token classification using HF Trainer.

Usage:
    python -m src.models.train_student \
        --model onlplab/alephbert-base \
        --data-dir data/processed \
        --output-dir models/checkpoints/student \
        --epochs 5 --batch-size 16 --lr 2e-5 --fp16

Fixes applied:
    - Added EarlyStoppingCallback (patience=3) per project guide
    - Added gradient clipping (max_grad_norm=1.0) per computing guide
    - Enabled TensorBoard logging
    - Added val set existence check before loading
    - Added overfit-test detection (skips early stopping when --max-examples is small)
    - Used eval_strategy instead of deprecated evaluation_strategy
    - Added overfit test warning about val metrics
"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

# =============================================================================
# DATASET
# =============================================================================

class TokenClassificationDataset(Dataset):
    """Loads preprocessed JSONL into a PyTorch Dataset."""

    def __init__(self, path, max_examples=None):
        self.examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))
                if max_examples and len(self.examples) >= max_examples:
                    break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
        }

# =============================================================================
# METRICS
# =============================================================================

def make_compute_metrics(id2label):
    """Returns a compute_metrics function for HF Trainer.

    NOTE: id2label loaded from JSON has string keys ("0", "1", ...).
    We look up with str(int(l)) to match those keys.
    """

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Convert to tag strings, skipping -100
        true_tags, pred_tags = [], []
        for pred_seq, label_seq in zip(predictions, labels):
            t_tags, p_tags = [], []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                t_tags.append(id2label.get(str(int(l)), "O"))
                p_tags.append(id2label.get(str(int(p)), "O"))
            true_tags.append(t_tags)
            pred_tags.append(p_tags)

        f1 = f1_score(true_tags, pred_tags)
        p = precision_score(true_tags, pred_tags)
        r = recall_score(true_tags, pred_tags)

        # Token accuracy (secondary)
        correct, total = 0, 0
        for ts, ps in zip(true_tags, pred_tags):
            for t, pr in zip(ts, ps):
                total += 1
                if t == pr:
                    correct += 1

        return {
            "f1": f1,
            "precision": p,
            "recall": r,
            "token_accuracy": correct / total if total else 0,
        }

    return compute_metrics

# =============================================================================
# TRAINING
# =============================================================================

def train(args):
    set_seed(args.seed)

    # Load label mapping
    data_dir = Path(args.data_dir)
    with open(data_dir / "id2label.json") as f:
        id2label = json.load(f)  # keys are strings: {"0": "O", "1": "B-SUB", ...}
    with open(data_dir / "label2id.json") as f:
        label2id = json.load(f)

    num_labels = len(label2id)
    print(f"Model: {args.model}, Labels: {num_labels}, Seed: {args.seed}")

    # Load datasets
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")

    train_ds = TokenClassificationDataset(train_path, args.max_examples)
    val_ds = TokenClassificationDataset(val_path)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Detect overfit test mode: small dataset, skip early stopping
    is_overfit_test = args.max_examples is not None and args.max_examples <= 200
    if is_overfit_test:
        print(f"\n{'='*60}")
        print(f"  *** OVERFIT TEST MODE (max_examples={args.max_examples}) ***")
        print(f"  Early stopping: DISABLED")
        print(f"  Val metrics will be low — this is expected.")
        print(f"  Check TRAIN loss to verify the model can memorize.")
        print(f"  If train loss doesn't decrease → you have a bug.")
        print(f"{'='*60}\n")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        id2label={int(k): v for k, v in id2label.items()},
        label2id={k: int(v) for k, v in label2id.items()},
    )

    # Training arguments
    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),

        # Training
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,

        # Optimization
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup,
        max_grad_norm=1.0,  # gradient clipping per computing guide

        # Mixed precision
        fp16=args.fp16,

        # Evaluation & Checkpointing
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # Logging
        logging_steps=50,
        logging_dir=str(output_dir / "logs"),
        report_to=["tensorboard"],

        # Reproducibility & Performance
        seed=args.seed,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    # Build callbacks
    callbacks = []
    if not is_overfit_test:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(id2label),
        callbacks=callbacks,
    )

    # Train
    result = trainer.train()
    print(f"\nTraining complete. Best F1 on val set.")

    # Save best model
    best_dir = output_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Best model saved to {best_dir}")

    # Save metrics
    metrics = trainer.evaluate()
    metrics["train_runtime"] = result.metrics.get("train_runtime", 0)
    metrics["train_loss"] = result.metrics.get("train_loss", 0)
    metrics["train_examples"] = len(train_ds)
    metrics["val_examples"] = len(val_ds)
    with open(output_dir / "training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Final eval: F1={metrics.get('eval_f1', 0):.4f}, "
          f"P={metrics.get('eval_precision', 0):.4f}, "
          f"R={metrics.get('eval_recall', 0):.4f}")

    # Print classification report
    print("\nPer-aspect breakdown:")
    val_preds = trainer.predict(val_ds)
    logits, labels = val_preds.predictions, val_preds.label_ids
    preds = np.argmax(logits, axis=-1)
    true_t, pred_t = [], []
    for ps, ls in zip(preds, labels):
        tt, pt = [], []
        for p, l in zip(ps, ls):
            if l == -100:
                continue
            tt.append(id2label.get(str(int(l)), "O"))
            pt.append(id2label.get(str(int(p)), "O"))
        true_t.append(tt)
        pred_t.append(pt)
    print(classification_report(true_t, pred_t))

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train AlephBERT student model")
    parser.add_argument("--model", default="onlplab/alephbert-base")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="models/checkpoints/student")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-examples", type=int, help="Limit training examples (for overfit test)")
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()