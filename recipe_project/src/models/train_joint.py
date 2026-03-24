"""
Training loop for BERT + CRF model (simplified, no intent conditioning).
=========================================================================
V7 CHANGE: Removed intent head, intent loss, scheduled sampling.
The CRF alone enforces valid BIO transitions. This simplification:
  - Eliminates cascade failure risk from wrong intent predictions
  - Makes the CRF ablation (A4) clean: CRF vs no-CRF, nothing else
  - Reduces code complexity by ~40%

Reads the same JSONL format produced by prepare_data_merged.py:
  Each line has: input_ids, attention_mask, labels, has_modification, thread_id, ...

Usage:
    python -m src.models.train_joint \
        --model dicta-il/dictabert \
        --data-dir data/processed \
        --train-file train_merged.jsonl \
        --output-dir models/checkpoints/joint_crf \
        --epochs 10 --batch-size 16 --lr 2e-5 --seed 42

    # With class weights (recommended):
    python -m src.models.train_joint \
        --model dicta-il/dictabert \
        --data-dir data/processed \
        --train-file train_merged.jsonl \
        --output-dir models/checkpoints/joint_crf \
        --epochs 10 --batch-size 16 --lr 2e-5 --seed 42 \
        --class-weights-file data/processed/stats_merged.json

    # With WandB:
    python -m src.models.train_joint \
        --model dicta-il/dictabert \
        --data-dir data/processed \
        --train-file train_merged.jsonl \
        --output-dir models/checkpoints/joint_crf \
        --epochs 10 --batch-size 16 --lr 2e-5 --seed 42 \
        --class-weights-file data/processed/stats_merged.json \
        --wandb --wandb-project recipe-mod-extraction --wandb-run joint-crf-v1
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from seqeval.metrics import (
    f1_score as seq_f1,
    precision_score as seq_precision,
    recall_score as seq_recall,
    classification_report as seq_report,
)

from .joint_model import BertCRFModel


# =============================================================================
# LABEL MAPS
# =============================================================================

BIO_LABEL2ID = {
    "O": 0,
    "B-SUBSTITUTION": 1, "I-SUBSTITUTION": 2,
    "B-QUANTITY": 3, "I-QUANTITY": 4,
    "B-TECHNIQUE": 5, "I-TECHNIQUE": 6,
    "B-ADDITION": 7, "I-ADDITION": 8,
}
BIO_ID2LABEL = {v: k for k, v in BIO_LABEL2ID.items()}


# =============================================================================
# DATASET
# =============================================================================

class CRFDataset(Dataset):
    """
    Loads preprocessed JSONL produced by prepare_data_merged.py.

    Each line contains:
        input_ids:        list of int (tokenized, padded to max_length)
        attention_mask:   list of int
        labels:           list of int (BIO label IDs, -100 for special tokens)
        has_modification: bool
        thread_id:        str
    """

    def __init__(self, path, max_len=128):
        self.examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    self.examples.append(ex)
                except json.JSONDecodeError:
                    continue
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        input_ids = ex["input_ids"][:self.max_len]
        attn_mask = ex["attention_mask"][:self.max_len]
        labels = ex["labels"][:self.max_len]

        # Pad if shorter than max_len
        pad_len = self.max_len - len(input_ids)
        input_ids = input_ids + [0] * pad_len
        attn_mask = attn_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# =============================================================================
# CLASS WEIGHTS LOADING
# =============================================================================

def load_class_weights_from_file(stats_file: str, device: torch.device) -> torch.Tensor:
    """Load inverse-frequency class weights from stats_merged.json.

    Falls back to uniform weights if file is missing or malformed.
    """
    path = Path(stats_file)
    if not path.exists():
        print(f"  WARNING: {stats_file} not found, using uniform weights")
        return None

    with open(path, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    weights_list = stats.get("class_weights", {}).get("inverse_frequency_list")
    if weights_list is None:
        print(f"  WARNING: No inverse_frequency_list in {stats_file}, using uniform weights")
        return None

    weights = torch.tensor(weights_list, dtype=torch.float32, device=device)
    print(f"  Loaded class weights from {stats_file}")
    print(f"  Weights: {[f'{w:.2f}' for w in weights.tolist()]}")
    return weights


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, loader, device):
    """Evaluate model on a data loader. Returns metrics dict."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            gold_labels = batch["labels"]

            # Inference: Viterbi decoding → list of predicted tag sequences
            best_paths = model(ids, mask)  # no labels → returns predictions

            # Convert to BIO label strings for seqeval
            for b in range(ids.size(0)):
                seq_len = int(mask[b].sum().item())
                pred_seq = []
                gold_seq = []

                for t in range(seq_len):
                    g = gold_labels[b, t].item()
                    if g == -100:
                        continue
                    gold_seq.append(BIO_ID2LABEL.get(g, "O"))
                    p = best_paths[b][t] if t < len(best_paths[b]) else 0
                    pred_seq.append(BIO_ID2LABEL.get(p, "O"))

                if gold_seq:
                    # Ensure equal length (trim pred if CRF returned extra)
                    pred_seq = pred_seq[:len(gold_seq)]
                    if len(pred_seq) < len(gold_seq):
                        pred_seq.extend(["O"] * (len(gold_seq) - len(pred_seq)))
                    all_preds.append(pred_seq)
                    all_labels.append(gold_seq)

    if not all_labels:
        return {
            "f1": 0.0, "precision": 0.0, "recall": 0.0,
            "all_preds": [], "all_labels": [],
        }

    return {
        "f1": seq_f1(all_labels, all_preds),
        "precision": seq_precision(all_labels, all_preds),
        "recall": seq_recall(all_labels, all_preds),
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


# =============================================================================
# TRAINING
# =============================================================================

def train(args):
    # ── Seeds ─────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {mem:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────
    train_path = os.path.join(args.data_dir, args.train_file)
    val_path = os.path.join(args.data_dir, "val.jsonl")

    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found")
        return
    if not os.path.exists(val_path):
        print(f"ERROR: {val_path} not found")
        return

    train_ds = CRFDataset(train_path, max_len=args.max_len)
    val_ds = CRFDataset(val_path, max_len=args.max_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    n_train_pos = sum(1 for ex in train_ds.examples if ex.get("has_modification"))
    n_val_pos = sum(1 for ex in val_ds.examples if ex.get("has_modification"))
    print(f"\nTrain: {len(train_ds)} examples ({n_train_pos} positive = "
          f"{n_train_pos/max(len(train_ds),1)*100:.1f}%)")
    print(f"Val:   {len(val_ds)} examples ({n_val_pos} positive = "
          f"{n_val_pos/max(len(val_ds),1)*100:.1f}%)")

    # ── Class weights (optional) ─────────────────────────────────────
    class_weights = None
    if args.class_weights_file:
        class_weights = load_class_weights_from_file(args.class_weights_file, device)

    # ── Model ─────────────────────────────────────────────────────────
    model = BertCRFModel(
        model_name=args.model,
        num_labels=9,
        dropout_rate=args.dropout,
        class_weights=class_weights,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model} + CRF")
    print(f"Parameters: {total_params:,} ({trainable_params:,} trainable)")

    # ── Optimizer (differential LR) ──────────────────────────────────
    # BERT layers: standard LR (2e-5)
    # CRF + classifier head: higher LR (5x) since randomly initialized
    bert_params = list(model.bert.parameters())
    head_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("bert.") and p.requires_grad
    ]

    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * 5},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── WandB (optional) ──────────────────────────────────────────────
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run,
                config={
                    "model": args.model,
                    "architecture": "BERT + CRF (no intent)",
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "head_lr": args.lr * 5,
                    "dropout": args.dropout,
                    "max_len": args.max_len,
                    "train_file": args.train_file,
                    "train_examples": len(train_ds),
                    "val_examples": len(val_ds),
                    "seed": args.seed,
                    "class_weights_file": args.class_weights_file,
                },
            )
        except Exception as e:
            print(f"WandB init failed: {e}. Continuing without WandB.")
            use_wandb = False

    # ── Output directory ──────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────
    best_f1 = 0.0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"lr={args.lr} (head: {args.lr*5})")
    print(f"Early stopping patience: {args.patience}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_start = time.time()

        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward: CRF NLL loss
            loss = model(ids, mask, labels=labels)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1

        # ── Averages ──────────────────────────────────────────────────
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t_start

        # ── Validation ────────────────────────────────────────────────
        metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"({elapsed:.0f}s)  "
              f"Loss={avg_loss:.4f}  "
              f"Val F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
              f"R={metrics['recall']:.4f}")

        # ── WandB logging ─────────────────────────────────────────────
        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_loss,
                "train/lr": scheduler.get_last_lr()[0],
                "val/f1": metrics["f1"],
                "val/precision": metrics["precision"],
                "val/recall": metrics["recall"],
            })

        # ── Early stopping ────────────────────────────────────────────
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience_counter = 0

            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "metrics": {k: v for k, v in metrics.items()
                            if k not in ("all_preds", "all_labels")},
                "args": vars(args),
            }, out_dir / "best_model.pt")
            print(f"  -> New best F1={best_f1:.4f}. Saved to {out_dir / 'best_model.pt'}")

            if use_wandb:
                import wandb
                wandb.run.summary["best_f1"] = best_f1
                wandb.run.summary["best_epoch"] = epoch + 1
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}.")
                break

    # ── Final report ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training complete. Best val F1: {best_f1:.4f}")
    print(f"{'='*60}")

    # Load best model and print detailed report
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    metrics = evaluate(model, val_loader, device)

    print(f"\nBest model (epoch {ckpt['epoch']+1}):")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")

    if metrics["all_labels"]:
        print(f"\nPer-class report:")
        print(seq_report(metrics["all_labels"], metrics["all_preds"]))

    # Save training summary
    summary = {
        "best_f1": best_f1,
        "best_epoch": ckpt["epoch"] + 1,
        "model": args.model,
        "architecture": "BERT + CRF (no intent conditioning)",
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "class_weights_file": args.class_weights_file,
        "args": vars(args),
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if use_wandb:
        import wandb
        wandb.finish()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train BERT + CRF model for BIO token classification"
    )
    # Model
    parser.add_argument("--model", default="dicta-il/dictabert",
                        help="Pretrained model name (HuggingFace)")
    parser.add_argument("--dropout", type=float, default=0.1)

    # Data
    parser.add_argument("--data-dir", default="data/processed",
                        help="Directory containing JSONL splits")
    parser.add_argument("--train-file", default="train_merged.jsonl",
                        help="Training file name within data-dir")
    parser.add_argument("--max-len", type=int, default=128)

    # Class weights
    parser.add_argument("--class-weights-file", default=None,
                        help="Path to stats_merged.json for class weights. "
                             "If not provided, no class weighting on emissions.")

    # Training
    parser.add_argument("--output-dir", default="models/checkpoints/joint_crf")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate for BERT layers (CRF head gets 5x)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--seed", type=int, default=42)

    # WandB (optional)
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="recipe-mod-extraction")
    parser.add_argument("--wandb-run", default=None,
                        help="WandB run name (auto-generated if not set)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()