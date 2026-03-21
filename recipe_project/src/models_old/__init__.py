"""
Models Module
=============
Student model training and inference.
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