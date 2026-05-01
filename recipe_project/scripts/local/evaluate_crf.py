#!/usr/bin/env python3
"""
evaluate_crf.py — Evaluate a saved BertCRFModel checkpoint on gold or silver test set.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from src.models.joint_model import BertCRFModel

BIO_ID2LABEL = {
    0: "O",
    1: "B-SUBSTITUTION", 2: "I-SUBSTITUTION",
    3: "B-QUANTITY",      4: "I-QUANTITY",
    5: "B-TECHNIQUE",     6: "I-TECHNIQUE",
    7: "B-ADDITION",      8: "I-ADDITION",
}


class EvalDataset(Dataset):
    def __init__(self, path):
        self.examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids":      torch.tensor(ex["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(ex["labels"],         dtype=torch.long),
        }


def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
    }


def evaluate_crf(ckpt_dir, test_file, output_dir, model_name, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(ckpt_dir) / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from {ckpt_path}")

    model = BertCRFModel(model_name=model_name, num_labels=9, dropout_rate=0.1)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    print(f"Model loaded: {model_name} + CRF")

    dataset = EvalDataset(test_file)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, collate_fn=collate_fn)
    print(f"Test examples: {len(dataset)}")

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            pred_sequences = model(input_ids, attention_mask)

            for i, pred_seq in enumerate(pred_sequences):
                gold_seq   = labels[i].tolist()
                pred_tags  = []
                gold_tags  = []
                for j, (p, g) in enumerate(zip(pred_seq, gold_seq)):
                    if g == -100:
                        continue
                    pred_tags.append(BIO_ID2LABEL.get(p, "O"))
                    gold_tags.append(BIO_ID2LABEL.get(g, "O"))
                all_preds.append(pred_tags)
                all_labels.append(gold_tags)

    entity_f1 = f1_score(all_labels, all_preds)
    entity_p  = precision_score(all_labels, all_preds)
    entity_r  = recall_score(all_labels, all_preds)
    report    = classification_report(all_labels, all_preds)

    # Bootstrap 95% CI
    rng      = np.random.default_rng(42)
    indices  = list(range(len(all_labels)))
    boot_f1s = []
    for _ in range(1000):
        sample   = rng.choice(indices, size=len(indices), replace=True)
        s_preds  = [all_preds[i]  for i in sample]
        s_labels = [all_labels[i] for i in sample]
        try:
            boot_f1s.append(f1_score(s_labels, s_preds))
        except Exception:
            boot_f1s.append(0.0)

    ci_low  = float(np.percentile(boot_f1s, 2.5))
    ci_high = float(np.percentile(boot_f1s, 97.5))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "entity_f1":              round(entity_f1, 4),
        "entity_precision":       round(entity_p,  4),
        "entity_recall":          round(entity_r,  4),
        "ci_95_low":              round(ci_low,    4),
        "ci_95_high":             round(ci_high,   4),
        "n_examples":             len(dataset),
        "model_path":             str(ckpt_dir),
        "test_file":              str(test_file),
        "classification_report":  report,
    }

    out_file = Path(output_dir) / "evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {Path(ckpt_dir).name}")
    print(f"{'='*60}")
    print(f"  Entity F1:        {entity_f1:.4f}")
    print(f"  Entity Precision: {entity_p:.4f}")
    print(f"  Entity Recall:    {entity_r:.4f}")
    print(f"  95% CI:           [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"\n{report}")
    print(f"  Saved to: {out_file}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir",   required=True)
    parser.add_argument("--test-file",  required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="dicta-il/dictabert")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    evaluate_crf(
        ckpt_dir   = args.ckpt_dir,
        test_file  = args.test_file,
        output_dir = args.output_dir,
        model_name = args.model_name,
        batch_size = args.batch_size,
    )


if __name__ == "__main__":
    main()
