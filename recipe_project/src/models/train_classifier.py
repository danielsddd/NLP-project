#!/usr/bin/env python3
"""
Modification Classifier — Step 1 of Two-Step Pipeline
=====================================================
Binary classifier: does this comment contain a recipe modification?

Trained on ALL examples (positive + negative).
At inference, filters to only modification-containing sentences
before passing to the token classifier (Step 2).

Usage:
    # Train
    python -m src.models.train_classifier \
        --model dicta-il/dictabert \
        --data-dir data/processed_dictabert \
        --output-dir models/checkpoints/classifier_dictabert \
        --epochs 10 --batch-size 32 --lr 3e-5 --fp16

    # This is typically run via the two-step sbatch script.
"""

import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

# =============================================================================
# DATASET
# =============================================================================

class ClassificationDataset(Dataset):
    """Converts token-classification JSONL to binary classification.
    
    Label 1 = has at least one non-O entity tag (modification present)
    Label 0 = all tokens are O (no modification)
    """

    def __init__(self, path, max_examples=None):
        self.examples = []
        pos_count = 0
        neg_count = 0

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                ex = json.loads(line)
                # Check if any token has a non-O, non-special label
                has_entity = any(l not in (-100, 0) for l in ex["labels"])
                label = 1 if has_entity else 0

                self.examples.append({
                    "input_ids": ex["input_ids"],
                    "attention_mask": ex["attention_mask"],
                    "label": label,
                })

                if label == 1:
                    pos_count += 1
                else:
                    neg_count += 1

                if max_examples and len(self.examples) >= max_examples:
                    break

        print(f"  Loaded {len(self.examples)} examples: {pos_count} positive, {neg_count} negative")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
        }

# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary"),
        "precision": precision_score(labels, predictions, average="binary", zero_division=0),
        "recall": recall_score(labels, predictions, average="binary", zero_division=0),
    }

# =============================================================================
# TRAINING
# =============================================================================

def train(args):
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"{'='*60}")
    print(f"CLASSIFIER TRAINING (Step 1)")
    print(f"{'='*60}")
    print(f"  Model:      {args.model}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output:     {args.output_dir}")
    print(f"  LR:         {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs:     {args.epochs}")
    print(f"{'='*60}")

    # Load datasets
    print("\nLoading training data...")
    train_ds = ClassificationDataset(data_dir / "train.jsonl")
    print("Loading validation data...")
    val_ds = ClassificationDataset(data_dir / "val.jsonl")

    # Compute class weights for the binary task
    pos = sum(1 for ex in train_ds.examples if ex["label"] == 1)
    neg = sum(1 for ex in train_ds.examples if ex["label"] == 0)
    total = pos + neg
    weight_neg = total / (2 * neg) if neg > 0 else 1.0
    weight_pos = total / (2 * pos) if pos > 0 else 1.0
    print(f"\n  Class weights: neg={weight_neg:.2f}, pos={weight_pos:.2f}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label={0: "NO_MOD", 1: "HAS_MOD"},
        label2id={"NO_MOD": 0, "HAS_MOD": 1},
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        fp16=args.fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        logging_dir=str(output_dir / "logs"),
        report_to=["tensorboard"],
        seed=args.seed,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    # Custom trainer with class-weighted loss
    class WeightedClassifierTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight = torch.tensor([weight_neg, weight_pos],
                                  dtype=torch.float32, device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedClassifierTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    result = trainer.train()
    print(f"\nTraining complete.")

    # Save best model
    best_dir = output_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Best model saved to {best_dir}")

    # Final evaluation
    metrics = trainer.evaluate()
    metrics["train_runtime"] = result.metrics.get("train_runtime", 0)
    metrics["train_examples"] = len(train_ds)
    metrics["val_examples"] = len(val_ds)
    with open(output_dir / "classifier_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nFinal: Acc={metrics.get('eval_accuracy', 0):.4f}, "
          f"F1={metrics.get('eval_f1', 0):.4f}, "
          f"P={metrics.get('eval_precision', 0):.4f}, "
          f"R={metrics.get('eval_recall', 0):.4f}")

    # Detailed predictions
    val_preds = trainer.predict(val_ds)
    preds = np.argmax(val_preds.predictions, axis=-1)
    labels = np.array([ex["label"] for ex in val_ds.examples])
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=["NO_MOD", "HAS_MOD"]))

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train binary modification classifier (Step 1)")
    parser.add_argument("--model", default="dicta-il/dictabert")
    parser.add_argument("--data-dir", default="data/processed_dictabert")
    parser.add_argument("--output-dir", default="models/checkpoints/classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
