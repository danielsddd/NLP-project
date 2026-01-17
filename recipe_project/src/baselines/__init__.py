"""
Baselines Module
================
Simple baseline implementations for comparison.

Baselines implemented:
1. MajorityBaseline - Predicts "O" for all tokens
2. RandomBaseline - Random labels based on distribution
3. KeywordBaseline - Hebrew keyword matching (FIXED version)

For mBERT baseline, use train_student.py with:
    --model bert-base-multilingual-cased
"""

from .run_baselines import (
    BaselineEvaluator,
    BaselineResult,
    BaselineModel,
    MajorityBaseline,
    RandomBaseline,
    KeywordBaseline,
    ASPECT_KEYWORDS,
    BIO_LABEL2ID,
    BIO_ID2LABEL,
)

__all__ = [
    "BaselineEvaluator",
    "BaselineResult",
    "BaselineModel",
    "MajorityBaseline",
    "RandomBaseline",
    "KeywordBaseline",
    "ASPECT_KEYWORDS",
    "BIO_LABEL2ID",
    "BIO_ID2LABEL",
]