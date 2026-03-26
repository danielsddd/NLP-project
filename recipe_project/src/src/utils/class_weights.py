#!/usr/bin/env python3
"""
Class Weights & Focal Loss Utilities
======================================
Provides auto-loading of class weights from stats_merged.json and a
FocalLoss implementation for train_student.py.

Resolves MASTER_PLAN_v7 clarification #1: weights are COMPUTED from
the actual training data distribution, not hardcoded.

Usage in train_student.py:
    from src.utils.class_weights import load_class_weights, FocalLoss

    # Option 1: Load from stats file (recommended)
    weights = load_class_weights("data/processed/stats_merged.json")

    # Option 2: Compute directly from training data
    weights = compute_weights_from_data("data/processed/train_merged.jsonl", num_labels=9)

    # Create loss function
    loss_fn = FocalLoss(weight=weights, gamma=2.0)
"""

import json
import math
from pathlib import Path
from collections import Counter
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# WEIGHT LOADING
# =============================================================================

def load_class_weights(
    stats_file: str,
    weight_type: str = "inverse_frequency",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Load pre-computed class weights from stats_merged.json.

    Args:
        stats_file: Path to stats_merged.json (output of prepare_data_merged.py)
        weight_type: "inverse_frequency" (auto-computed) or "uniform" (all 1.0)
        device: Target device for the tensor

    Returns:
        torch.Tensor of shape (num_labels,)

    Raises:
        FileNotFoundError: If stats_file does not exist
        KeyError: If the expected structure is missing from the JSON
    """
    path = Path(stats_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Stats file not found: {stats_file}\n"
            f"Run prepare_data_merged.py first to generate it."
        )

    with open(path, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    cw = stats.get("class_weights", {})

    if weight_type == "inverse_frequency":
        weights_list = cw.get("inverse_frequency_list")
        if weights_list is None:
            raise KeyError(
                f"'class_weights.inverse_frequency_list' not found in {stats_file}. "
                f"Re-run prepare_data_merged.py to regenerate."
            )
    elif weight_type == "uniform":
        weights_list = cw.get("uniform")
        if weights_list is None:
            # Fallback: infer num_labels from inverse_frequency_list
            inv_list = cw.get("inverse_frequency_list", [])
            weights_list = [1.0] * len(inv_list)
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}. Use 'inverse_frequency' or 'uniform'.")

    tensor = torch.tensor(weights_list, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)

    return tensor


def compute_weights_from_data(
    train_file: str,
    num_labels: int = 9,
    cap: float = 20.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute inverse-frequency weights directly from a training JSONL file.

    This is a fallback if stats_merged.json is not available. It reads the
    training data and counts label occurrences.

    Args:
        train_file: Path to train_merged.jsonl
        num_labels: Number of BIO labels (default: 9)
        cap: Maximum weight cap (default: 20.0)
        device: Target device

    Returns:
        torch.Tensor of shape (num_labels,)
    """
    label_counts = Counter()
    total_tokens = 0

    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                ex = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            for lid in ex.get("labels", []):
                if lid == -100:
                    continue
                total_tokens += 1
                label_counts[lid] += 1

    weights = []
    for i in range(num_labels):
        count = label_counts.get(i, 1)
        raw_weight = total_tokens / (num_labels * count)
        weights.append(min(raw_weight, cap))

    tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)

    return tensor


# =============================================================================
# FOCAL LOSS
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for token classification.

    Focal loss down-weights easy examples and focuses on hard ones:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma=0, this reduces to standard weighted cross-entropy.
    When gamma=2 (recommended), easy examples (p_t > 0.95, i.e. O tokens)
    contribute ~25x less to the loss than hard examples.

    Reference:
        Lin et al. (2017) "Focal Loss for Dense Object Detection"

    Args:
        weight: Optional class weights tensor of shape (num_classes,).
            Acts as alpha_t per class. Use inverse-frequency weights.
        gamma: Focusing parameter. 0 = standard CE, 2 = recommended.
        ignore_index: Label ID to ignore (-100 for HuggingFace padding).
        reduction: 'mean' or 'sum' or 'none'.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        # Register weight as a buffer so it moves with .to(device)
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Model output logits of shape (batch, seq_len, num_classes)
                    or (N, num_classes) if already flattened.
            targets: Ground truth labels of shape (batch, seq_len) or (N,).

        Returns:
            Scalar loss (if reduction='mean' or 'sum') or per-element loss.
        """
        # Flatten if needed
        if logits.dim() == 3:
            # (batch, seq_len, num_classes) → (batch*seq_len, num_classes)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        # Mask out ignore_index
        mask = targets != self.ignore_index
        logits = logits[mask]
        targets = targets[mask]

        if logits.numel() == 0:
            return logits.sum()  # empty tensor, return 0

        # Standard cross-entropy (log softmax + nll)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Gather the log-prob and prob for the target class
        # targets shape: (N,) → needs (N, 1) for gather
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - target_probs) ** self.gamma

        # Class weight (alpha_t)
        if self.weight is not None:
            alpha = self.weight.gather(0, targets)
            focal_weight = focal_weight * alpha

        # Loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        loss = -focal_weight * target_log_probs

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# =============================================================================
# CONVENIENCE: Create loss function from config
# =============================================================================

def create_loss_function(
    stats_file: Optional[str] = None,
    train_file: Optional[str] = None,
    use_class_weights: bool = True,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    weight_type: str = "inverse_frequency",
    num_labels: int = 9,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Create the appropriate loss function based on configuration.

    This is the main entry point for train_student.py to get a loss function.

    Args:
        stats_file: Path to stats_merged.json (preferred source for weights)
        train_file: Path to train_merged.jsonl (fallback for weight computation)
        use_class_weights: Whether to apply class weighting
        use_focal_loss: Whether to use focal loss (vs standard CE)
        focal_gamma: Gamma parameter for focal loss
        weight_type: "inverse_frequency" or "uniform"
        num_labels: Number of BIO labels
        device: Target device

    Returns:
        nn.Module loss function (FocalLoss or CrossEntropyLoss)

    Examples:
        # A1 ablation — no weights:
        loss_fn = create_loss_function(use_class_weights=False)

        # A1 ablation — uniform weights:
        loss_fn = create_loss_function(
            stats_file="data/processed/stats_merged.json",
            weight_type="uniform"
        )

        # A1 ablation — inverse-frequency weights:
        loss_fn = create_loss_function(
            stats_file="data/processed/stats_merged.json",
            weight_type="inverse_frequency"
        )

        # Full config — inverse-frequency + focal:
        loss_fn = create_loss_function(
            stats_file="data/processed/stats_merged.json",
            use_class_weights=True,
            use_focal_loss=True,
            focal_gamma=2.0
        )

        # A3 ablation — weighted CE (no focal):
        loss_fn = create_loss_function(
            stats_file="data/processed/stats_merged.json",
            use_class_weights=True,
            use_focal_loss=False
        )
    """
    weights = None

    if use_class_weights:
        if stats_file and Path(stats_file).exists():
            weights = load_class_weights(stats_file, weight_type, device)
            print(f"  Loaded {weight_type} class weights from {stats_file}")
        elif train_file and Path(train_file).exists():
            weights = compute_weights_from_data(train_file, num_labels, device=device)
            print(f"  Computed class weights from {train_file}")
        else:
            print(f"  ⚠️  No stats_file or train_file provided — using uniform weights")
            weights = torch.ones(num_labels, dtype=torch.float32)
            if device:
                weights = weights.to(device)

        print(f"  Weights: {[f'{w:.2f}' for w in weights.tolist()]}")

    if use_focal_loss:
        print(f"  Loss function: FocalLoss(gamma={focal_gamma})")
        return FocalLoss(weight=weights, gamma=focal_gamma)
    else:
        if weights is not None:
            print(f"  Loss function: CrossEntropyLoss(weighted)")
            return nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        else:
            print(f"  Loss function: CrossEntropyLoss(unweighted)")
            return nn.CrossEntropyLoss(ignore_index=-100)
