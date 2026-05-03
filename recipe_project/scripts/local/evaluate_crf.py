#!/usr/bin/env python3
"""Evaluate a saved BertCRFModel checkpoint on gold or silver test set."""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, classification_report as seqeval_report,
)

from src.models.joint_model import BertCRFModel
from src.preprocessing.prepare_data import BIO_LABEL2ID, IO_LABEL2ID


# ── Span helpers ─────────────────────────────────────────────────────────────

def get_spans_from_bio(seq):
    """Extract (label, start, end) tuples from a BIO tag sequence."""
    spans = []
    current_label, start_idx = None, -1
    for i, tag in enumerate(seq):
        if tag.startswith("B-"):
            if current_label is not None:
                spans.append((current_label, start_idx, i - 1))
            current_label, start_idx = tag[2:], i
        elif tag.startswith("I-") and current_label == tag[2:]:
            continue
        else:
            if current_label is not None:
                spans.append((current_label, start_idx, i - 1))
                current_label = None
    if current_label is not None:
        spans.append((current_label, start_idx, len(seq) - 1))
    return spans


def compute_relaxed_metrics(true_seqs, pred_seqs):
    """
    Relaxed (partial-boundary) P/R/F1 — global + per-aspect.
    TP = predicted span overlaps >= 1 token with gold span of SAME aspect.
    Guaranteed: relaxed_f1 >= exact_f1.
    """
    from collections import defaultdict
    total_true = total_pred = matched_true = matched_pred = 0
    asp_true = defaultdict(int); asp_pred = defaultdict(int)
    asp_matched_true = defaultdict(int); asp_matched_pred = defaultdict(int)

    for true_seq, pred_seq in zip(true_seqs, pred_seqs):
        true_spans = get_spans_from_bio(true_seq)
        pred_spans = get_spans_from_bio(pred_seq)
        total_true += len(true_spans)
        total_pred += len(pred_spans)
        for t_label, t_start, t_end in true_spans:
            asp_true[t_label] += 1
        for p_label, p_start, p_end in pred_spans:
            asp_pred[p_label] += 1

        for t_label, t_start, t_end in true_spans:
            t_range = set(range(t_start, t_end + 1))
            if any(p_label == t_label and t_range & set(range(p_start, p_end + 1))
                   for p_label, p_start, p_end in pred_spans):
                matched_true += 1
                asp_matched_true[t_label] += 1

        for p_label, p_start, p_end in pred_spans:
            p_range = set(range(p_start, p_end + 1))
            if any(t_label == p_label and p_range & set(range(t_start, t_end + 1))
                   for t_label, t_start, t_end in true_spans):
                matched_pred += 1
                asp_matched_pred[p_label] += 1

    p  = matched_pred / total_pred if total_pred > 0 else 0.0
    r  = matched_true / total_true if total_true > 0 else 0.0
    f1 = 2 * p * r / (p + r)      if (p + r)    > 0 else 0.0

    per_aspect_relaxed = {}
    for asp in set(list(asp_true.keys()) + list(asp_pred.keys())):
        ap = asp_matched_pred[asp] / asp_pred[asp] if asp_pred[asp] > 0 else 0.0
        ar = asp_matched_true[asp] / asp_true[asp] if asp_true[asp] > 0 else 0.0
        af = 2*ap*ar/(ap+ar) if (ap+ar) > 0 else 0.0
        per_aspect_relaxed[asp] = {
            "relaxed_p": round(ap, 4), "relaxed_r": round(ar, 4),
            "relaxed_f1": round(af, 4),
            "n_gold": asp_true[asp], "n_pred": asp_pred[asp],
        }

    return {
        "relaxed_precision":    round(p,  4),
        "relaxed_recall":       round(r,  4),
        "relaxed_f1":           round(f1, 4),
        "per_aspect_relaxed":   per_aspect_relaxed,
    }


def compute_token_level_f1(true_seqs, pred_seqs):
    """True token-level macro F1 — flattens to individual tokens, ignores O."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    y_true = [t for seq in true_seqs for t in seq]
    y_pred = [t for seq in pred_seqs for t in seq]
    labels = sorted(set(y_true + y_pred) - {"O"})
    p  = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    r  = recall_score(   y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(       y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "token_precision": round(p,  4),
        "token_recall":    round(r,  4),
        "token_f1":        round(f1, 4),
    }


def bootstrap_relaxed_ci(true_seqs, pred_seqs, n_iter=1000, seed=42):
    """Bootstrap 95% CI for relaxed F1."""
    rng     = np.random.default_rng(seed)
    indices = list(range(len(true_seqs)))
    scores  = []
    for _ in range(n_iter):
        sample = rng.choice(indices, size=len(indices), replace=True)
        s_true = [true_seqs[i] for i in sample]
        s_pred = [pred_seqs[i] for i in sample]
        scores.append(compute_relaxed_metrics(s_true, s_pred)["relaxed_f1"])
    lo = float(np.percentile(scores, 2.5))
    hi = float(np.percentile(scores, 97.5))
    return round(lo, 4), round(hi, 4)


# Hebrew prefix set — single-char prefixes that attach to span boundaries
HEBREW_PREFIXES = {"ב", "ל", "מ", "ה", "ו", "כ", "ש", "ד", "ר"}

def analyze_span_errors(true_seqs, pred_seqs, tokens_seqs=None):
    """
    Classify boundary errors into:
      - over_extension:  pred span longer than gold (pred_len > gold_len)
      - under_extension: pred span shorter than gold (pred_len < gold_len)
      - off_by_one:      boundary differs by exactly 1 token
      - prefix_error:    off-by-one AND the boundary token is a Hebrew prefix char
      - aspect_confusion: overlap exists but wrong aspect label
      - false_negative:  gold span with no overlapping pred
      - false_positive:  pred span with no overlapping gold
    """
    from collections import defaultdict
    counts = defaultdict(int)
    prefix_examples = []

    for idx, (true_seq, pred_seq) in enumerate(zip(true_seqs, pred_seqs)):
        true_spans = get_spans_from_bio(true_seq)
        pred_spans = get_spans_from_bio(pred_seq)
        toks       = tokens_seqs[idx] if tokens_seqs else None

        matched_p = set()
        for t_label, t_start, t_end in true_spans:
            t_range   = set(range(t_start, t_end + 1))
            t_len     = t_end - t_start + 1
            overlapping = [(pl, ps, pe) for pl, ps, pe in pred_spans
                           if t_range & set(range(ps, pe + 1))]

            if not overlapping:
                counts["false_negative"] += 1
                continue

            # find best overlapping pred (same aspect preferred)
            same_asp = [(pl, ps, pe) for pl, ps, pe in overlapping if pl == t_label]
            best     = same_asp[0] if same_asp else overlapping[0]
            pl, ps, pe = best
            p_len    = pe - ps + 1

            if pl != t_label:
                counts["aspect_confusion"] += 1
                continue

            matched_p.add((pl, ps, pe))

            if ps == t_start and pe == t_end:
                counts["exact_match"] += 1
            else:
                # boundary error
                start_diff = abs(ps - t_start)
                end_diff   = abs(pe - t_end)
                total_diff = start_diff + end_diff

                if p_len > t_len:
                    counts["over_extension"] += 1
                else:
                    counts["under_extension"] += 1

                if total_diff == 1:
                    counts["off_by_one"] += 1
                    # check if boundary token is Hebrew prefix
                    if toks:
                        boundary_idx = (ps - 1) if ps < t_start else pe
                        boundary_idx = max(0, min(boundary_idx, len(toks) - 1))
                        tok = toks[boundary_idx] if boundary_idx < len(toks) else ""
                        if tok and tok[0] in HEBREW_PREFIXES:
                            counts["prefix_error"] += 1
                            if len(prefix_examples) < 10:
                                prefix_examples.append({
                                    "gold": (t_label, t_start, t_end),
                                    "pred": (pl, ps, pe),
                                    "boundary_token": tok,
                                })

        for pl, ps, pe in pred_spans:
            if (pl, ps, pe) not in matched_p:
                t_range = set(range(ps, pe + 1))
                if not any(t_range & set(range(ts, te + 1))
                           for _, ts, te in true_spans):
                    counts["false_positive"] += 1

    # boundary error rate among all boundary errors
    boundary_errors = counts["over_extension"] + counts["under_extension"]
    pct_off1   = counts["off_by_one"]   / boundary_errors if boundary_errors > 0 else 0
    pct_prefix = counts["prefix_error"] / counts["off_by_one"] if counts["off_by_one"] > 0 else 0

    return {
        "counts":           dict(counts),
        "boundary_errors":  boundary_errors,
        "pct_off_by_one":   round(pct_off1,   4),
        "pct_prefix_error": round(pct_prefix, 4),
        "prefix_examples":  prefix_examples,
    }


def compute_span_count_stats(true_seqs, pred_seqs):
    """Verbosity bias — avg spans per positive example, gold vs predicted."""
    true_counts = [len(get_spans_from_bio(seq)) for seq in true_seqs]
    pred_counts = [len(get_spans_from_bio(seq)) for seq in pred_seqs]
    pos_idx = [i for i, c in enumerate(true_counts) if c > 0]
    avg_true = sum(true_counts[i] for i in pos_idx) / len(pos_idx) if pos_idx else 0.0
    avg_pred = sum(pred_counts[i] for i in pos_idx) / len(pos_idx) if pos_idx else 0.0
    return {
        "total_gold_spans":      sum(true_counts),
        "total_pred_spans":      sum(pred_counts),
        "avg_gold_per_positive": round(avg_true, 3),
        "avg_pred_per_positive": round(avg_pred, 3),
        "n_positive_examples":   len(pos_idx),
    }


# ── IO → BIO converter ───────────────────────────────────────────────────────

def convert_io_to_bio(tags):
    bio_tags = []
    prev_type = None
    for tag in tags:
        if tag == "O":
            bio_tags.append("O")
            prev_type = None
        elif tag.startswith("I-"):
            etype = tag[2:]
            bio_tags.append(f"B-{etype}" if prev_type != etype else f"I-{etype}")
            prev_type = etype
        else:
            bio_tags.append(tag)
            prev_type = tag[2:] if tag.startswith(("B-", "I-")) else None
    return bio_tags


# ── Dataset ──────────────────────────────────────────────────────────────────

class EvalDataset(Dataset):
    def __init__(self, path):
        self.examples = []
        with open(path, encoding="utf-8") as f:
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


# ── Main evaluation logic ────────────────────────────────────────────────────

def evaluate_crf(ckpt_dir, test_file, output_dir, model_name, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(ckpt_dir) / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from {ckpt_path}")

    # Detect number of labels from checkpoint tensors
    state_dict    = ckpt["model_state_dict"]
    detected_labels = None
    for key in state_dict:
        if any(pat in key for pat in ("classifier", "emission", "hidden2tag")):
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
    print(f"Detected {num_labels} labels in checkpoint tensors.")

    ckpt_name    = str(ckpt_dir).lower()
    is_io_scheme = ("io" in ckpt_name or num_labels == 5)

    model_label2id = IO_LABEL2ID if is_io_scheme else BIO_LABEL2ID
    id2label       = {v: k for k, v in model_label2id.items()}
    print(f"Using {'IO' if is_io_scheme else 'BIO'} mapping for predictions.")

    model = BertCRFModel(model_name=model_name, num_labels=num_labels, dropout_rate=0.1)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    print(f"Model loaded: {model_name} + CRF")

    dataset = EvalDataset(test_file)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Test examples: {len(dataset)}")

    # Detect gold scheme
    max_gold_label = 0
    for ex in dataset.examples:
        for lbl in ex["labels"]:
            if lbl != -100 and lbl > max_gold_label:
                max_gold_label = lbl
    gold_is_io    = (max_gold_label <= 4)
    gold_label2id = IO_LABEL2ID if gold_is_io else BIO_LABEL2ID
    gold_id2label = {v: k for k, v in gold_label2id.items()}
    if gold_is_io != is_io_scheme:
        print(f"  NOTE: Model={'IO' if is_io_scheme else 'BIO'}  Gold={'IO' if gold_is_io else 'BIO'}")
    print(f"  Gold scheme: {'IO' if gold_is_io else 'BIO'} (max label ID={max_gold_label})")

    # Inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]
            pred_sequences = model(input_ids, attention_mask)

            for i, pred_seq in enumerate(pred_sequences):
                gold_seq   = labels[i].tolist()
                pred_tags, gold_tags = [], []
                for p, g in zip(pred_seq, gold_seq):
                    if g == -100:
                        continue
                    pred_tags.append(id2label.get(p, "O"))
                    gold_tags.append(gold_id2label.get(g, "O"))
                if is_io_scheme:
                    pred_tags = convert_io_to_bio(pred_tags)
                if gold_is_io:
                    gold_tags = convert_io_to_bio(gold_tags)
                all_preds.append(pred_tags)
                all_labels.append(gold_tags)

    # ── Exact entity metrics ─────────────────────────────────────────────────
    entity_f1 = f1_score(all_labels, all_preds)
    entity_p  = precision_score(all_labels, all_preds)
    entity_r  = recall_score(all_labels, all_preds)
    report    = classification_report(all_labels, all_preds)

    # ── Relaxed + token metrics ──────────────────────────────────────────────
    relaxed = compute_relaxed_metrics(all_labels, all_preds)
    token   = compute_token_level_f1(all_labels, all_preds)

    # Sanity check — relaxed must always be >= exact
    assert relaxed["relaxed_f1"] >= round(entity_f1, 4) - 1e-4, (
        f"BUG: relaxed_f1={relaxed['relaxed_f1']} < exact_f1={entity_f1:.4f}"
    )

    # Bootstrap CI on relaxed F1
    relaxed_ci_lo, relaxed_ci_hi = bootstrap_relaxed_ci(all_labels, all_preds)

    # Span error analysis (tokens not available from CRF output — pass None)
    span_errors  = analyze_span_errors(all_labels, all_preds, tokens_seqs=None)
    span_counts  = compute_span_count_stats(all_labels, all_preds)

    # ── Bootstrap CI (exact F1) ──────────────────────────────────────────────
    rng     = np.random.default_rng(42)
    indices = list(range(len(all_labels)))
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

    # ── Save results ─────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        # exact
        "entity_f1":              round(entity_f1, 4),
        "entity_precision":       round(entity_p,  4),
        "entity_recall":          round(entity_r,  4),
        "ci_95_low":              round(ci_low,    4),
        "ci_95_high":             round(ci_high,   4),
        # relaxed
        "relaxed_f1":             relaxed["relaxed_f1"],
        "relaxed_precision":      relaxed["relaxed_precision"],
        "relaxed_recall":         relaxed["relaxed_recall"],
        "relaxed_ci_95_low":      relaxed_ci_lo,
        "relaxed_ci_95_high":     relaxed_ci_hi,
        "per_aspect_relaxed":     relaxed["per_aspect_relaxed"],
        # token-level
        "token_f1":               token["token_f1"],
        "token_precision":        token["token_precision"],
        "token_recall":           token["token_recall"],
        # span error analysis
        "span_error_counts":      span_errors["counts"],
        "boundary_errors":        span_errors["boundary_errors"],
        "pct_off_by_one":         span_errors["pct_off_by_one"],
        "pct_prefix_error":       span_errors["pct_prefix_error"],
        "span_count_stats":       span_counts,
        "prefix_examples":        span_errors["prefix_examples"],
        # meta
        "n_examples":             len(dataset),
        "model_path":             str(ckpt_dir),
        "test_file":              str(test_file),
        "classification_report":  report,
        "gold_scheme":            "IO" if gold_is_io  else "BIO",
        "model_scheme":           "IO" if is_io_scheme else "BIO",
    }

    out_file = Path(output_dir) / "evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS: {Path(ckpt_dir).name}")
    print(f"{'='*60}")
    print(f"  Exact   F1:  {entity_f1:.4f}   "
          f"(P={entity_p:.4f}  R={entity_r:.4f})")
    print(f"  Exact CI:    [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Relaxed F1:  {relaxed['relaxed_f1']:.4f}   "
          f"(P={relaxed['relaxed_precision']:.4f}  R={relaxed['relaxed_recall']:.4f})")
    print(f"  Relaxed CI:  [{relaxed_ci_lo:.4f}, {relaxed_ci_hi:.4f}]")
    print(f"  Token   F1:  {token['token_f1']:.4f}   "
          f"(P={token['token_precision']:.4f}  R={token['token_recall']:.4f})")
    print(f"  Span counts: gold_avg={span_counts['avg_gold_per_positive']:.2f}  "
          f"pred_avg={span_counts['avg_pred_per_positive']:.2f}  "
          f"(over {span_counts['n_positive_examples']} positive examples)")
    print(f"  Span errors: boundary={span_errors['boundary_errors']}  "
          f"off-by-1={span_errors['counts'].get('off_by_one',0)}  "
          f"prefix={span_errors['counts'].get('prefix_error',0)}")
    print(f"  Per-aspect relaxed F1:")
    for asp, m in sorted(relaxed["per_aspect_relaxed"].items()):
        print(f"    {asp:<15s}  relaxed_F1={m['relaxed_f1']:.3f}  "
              f"(P={m['relaxed_p']:.3f}  R={m['relaxed_r']:.3f}  n={m['n_gold']})")
    print(f"\n{report}")
    print(f"  Saved → {out_file}")
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