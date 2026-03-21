"""
Models Module
=============
Two-step pipeline for recipe modification extraction:
  Step 1: Binary classifier (has_modification?)  — train_classifier.py
  Step 2: Token classifier (BIO span extraction) — train_student.py

v5: Added WeightedTrainer, FocalLoss, class weights, negative downsampling.
"""

from .train_student import (
    TokenClassificationDataset,
    WeightedTrainer,
    FocalLoss,
    make_compute_metrics,
    compute_class_weights,
    train,
)

__all__ = [
    "TokenClassificationDataset",
    "WeightedTrainer",
    "FocalLoss",
    "make_compute_metrics",
    "compute_class_weights",
    "train",
]
