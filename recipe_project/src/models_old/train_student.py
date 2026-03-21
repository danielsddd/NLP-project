#!/usr/bin/env python3
"""
Student Model Training — v5 (Enhanced)
Fine-tunes BERT models for token classification with:
  - Class-weighted loss (balanced/sqrt) to fight O-token dominance
  - Focal loss option for hard example mining
  - Negative example downsampling
  - Early stopping on entity F1

Usage:
    # Basic (same as before)
    python -m src.models.train_student --model dicta-il/dictabert --data-dir data/processed_dictabert --output-dir models/checkpoints/dictabert --fp16

    # With class weights (recommended)
    python -m src.models.train_student --model dicta-il/dictabert --data-dir data/processed_dictabert --output-dir models/checkpoints/dictabert_weighted --class-weights balanced --fp16

    # With downsampling + weights
    python -m src.models.train_student --model dicta-il/dictabert --data-dir data/processed_dictabert --output-dir models/checkpoints/dictabert_ds --class-weights balanced --neg-ratio 3.0 --fp16

    # With focal loss
    python -m src.models.train_student --model dicta-il/dictabert --data-dir data/processed_dictabert --output-dir models/checkpoints/dictabert_focal --focal-loss --focal-gamma 2.0 --fp16
"""

import json
import argparse
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
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
# FOCAL LOSS
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss: down-weights easy examples, focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, weight=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.weight,
            ignore_index=self.ignore_index, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# =============================================================================
# DATASET
# =============================================================================

class TokenClassificationDataset(Dataset):
    """Loads preprocessed JSONL with optional negative downsampling."""

    def __init__(self, path, max_examples=None, neg_ratio=None, seed=42):
        """
        Args:
            path: Path to JSONL file
            max_examples: Limit total examples (for overfit test)
            neg_ratio: Ratio of negative to positive examples.
                       None = keep all. 3.0 = keep 3x negatives per positive.
                       1.0 = balanced 50/50.
        """
        all_examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                all_examples.append(json.loads(line))
                if max_examples and len(all_examples) >= max_examples:
                    break

        if neg_ratio is not None and max_examples is None:
            positives = []
            negatives = []
            for ex in all_examples:
                has_entity = any(l not in (-100, 0) for l in ex["labels"])
                if has_entity:
                    positives.append(ex)
                else:
                    negatives.append(ex)

            max_neg = int(len(positives) * neg_ratio)
            rng = random.Random(seed)
            if len(negatives) > max_neg:
                negatives = rng.sample(negatives, max_neg)

            self.examples = positives + negatives
            rng.shuffle(self.examples)
            print(f"  Downsampled: {len(positives)} pos + {len(negatives)} neg "
                  f"= {len(self.examples)} total (ratio={neg_ratio})")
        else:
            self.examples = all_examples

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
# CLASS WEIGHT COMPUTATION
# =============================================================================

def compute_class_weights(dataset, num_labels, scheme="balanced"):
    """Compute class weights from training data.

    Schemes:
        'balanced': weight = total / (num_classes * count_per_class)
                    Rare classes get much higher weight.
        'sqrt':     weight = sqrt(total / (num_classes * count_per_class))
                    Smoother version — less extreme upweighting.
    """
    counts = Counter()
    for ex in dataset.examples:
        for l in ex["labels"]:
            if l != -100:
                counts[l] += 1

    total = sum(counts.values())
    weights = []
    for i in range(num_labels):
        count = counts.get(i, 1)
        if scheme == "balanced":
            w = total / (num_labels * count)
        elif scheme == "sqrt":
            w = (total / (num_labels * count)) ** 0.5
        else:
            w = 1.0
        weights.append(w)

    print(f"  Class weights ({scheme}):")
    for i, w in enumerate(weights):
        pct = counts.get(i, 0) / total * 100 if total > 0 else 0
        print(f"    Label {i}: count={counts.get(i, 0):>7d} ({pct:5.1f}%)  weight={w:.4f}")

    return weights

# =============================================================================
# CUSTOM TRAINER WITH WEIGHTED/FOCAL LOSS
# =============================================================================

class WeightedTrainer(Trainer):
    """Trainer with optional class-weighted or focal loss."""

    def __init__(self, class_weights=None, use_focal=False, focal_gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self._loss_fn = None

    def _get_loss_fn(self, device):
        if self._loss_fn is not None:
            return self._loss_fn

        weight = None
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=torch.float32, device=device)

        if self.use_focal:
            self._loss_fn = FocalLoss(weight=weight, gamma=self.focal_gamma, ignore_index=-100)
        else:
            self._loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)

        return self._loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = self._get_loss_fn(logits.device)
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# =============================================================================
# METRICS
# =============================================================================

def make_compute_metrics(id2label):
    """Returns a compute_metrics function for HF Trainer."""

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

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
        id2label = json.load(f)
    with open(data_dir / "label2id.json") as f:
        label2id = json.load(f)

    num_labels = len(label2id)
    print(f"{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Model:         {args.model}")
    print(f"  Labels:        {num_labels}")
    print(f"  Seed:          {args.seed}")
    print(f"  LR:            {args.lr}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Patience:      {args.patience}")
    print(f"  Class weights: {args.class_weights or 'none'}")
    print(f"  Neg ratio:     {args.neg_ratio or 'all'}")
    print(f"  Focal loss:    {args.focal_loss} (gamma={args.focal_gamma})")
    print(f"  FP16:          {args.fp16}")
    print(f"{'='*60}")

    # Load datasets
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")

    print(f"\nLoading training data...")
    train_ds = TokenClassificationDataset(
        train_path, args.max_examples, neg_ratio=args.neg_ratio, seed=args.seed
    )
    val_ds = TokenClassificationDataset(val_path)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Detect overfit test mode
    is_overfit_test = args.max_examples is not None and args.max_examples <= 200
    if is_overfit_test:
        print(f"\n{'='*60}")
        print(f"  *** OVERFIT TEST MODE (max_examples={args.max_examples}) ***")
        print(f"  Early stopping: DISABLED")
        print(f"  Val metrics will be low — this is expected.")
        print(f"  Check TRAIN loss to verify the model can memorize.")
        print(f"{'='*60}\n")

    # Compute class weights
    class_weights = None
    if args.class_weights and args.class_weights != "none":
        print(f"\nComputing class weights...")
        class_weights = compute_class_weights(train_ds, num_labels, scheme=args.class_weights)

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
        max_grad_norm=1.0,

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
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    # Use weighted trainer if class weights or focal loss requested
    use_custom = class_weights is not None or args.focal_loss
    TrainerClass = WeightedTrainer if use_custom else Trainer

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(id2label),
        callbacks=callbacks,
    )

    if use_custom:
        trainer_kwargs["class_weights"] = class_weights
        trainer_kwargs["use_focal"] = args.focal_loss
        trainer_kwargs["focal_gamma"] = args.focal_gamma

    trainer = TrainerClass(**trainer_kwargs)

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
    metrics["class_weights_scheme"] = args.class_weights or "none"
    metrics["neg_ratio"] = args.neg_ratio
    metrics["focal_loss"] = args.focal_loss
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
    parser = argparse.ArgumentParser(description="Train student model (v5 - enhanced)")
    # Model & data
    parser.add_argument("--model", default="onlplab/alephbert-base")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="models/checkpoints/student")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (default: 3)")

    # Class imbalance handling
    parser.add_argument("--class-weights", choices=["none", "balanced", "sqrt"],
                        default=None,
                        help="Class weighting scheme for loss function")
    parser.add_argument("--neg-ratio", type=float, default=None,
                        help="Ratio of negative to positive examples. "
                             "3.0 = 3x negatives per positive. None = keep all.")

    # Focal loss
    parser.add_argument("--focal-loss", action="store_true",
                        help="Use focal loss instead of cross-entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter (default: 2.0)")

    # Overfit test
    parser.add_argument("--max-examples", type=int,
                        help="Limit training examples (for overfit test)")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()