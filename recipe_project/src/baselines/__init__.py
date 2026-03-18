"""
Baselines Module
================
Simple baseline implementations for comparison.

Baselines implemented:
1. majority_predict - Predicts "O" for all tokens
2. random_predict - Random labels based on distribution
3. keyword_predict - Hebrew keyword matching

For mBERT baseline, use train_student.py with:
    --model bert-base-multilingual-cased
"""

from .run_baselines import (
    majority_predict,
    random_predict,
    keyword_predict,
    evaluate_baseline,
    get_gold_tags,
    ASPECT_KEYWORDS,
    BIO_LABEL2ID,
    BIO_ID2LABEL,
)

__all__ = [
    "majority_predict",
    "random_predict",
    "keyword_predict",
    "evaluate_baseline",
    "get_gold_tags",
    "ASPECT_KEYWORDS",
    "BIO_LABEL2ID",
    "BIO_ID2LABEL",
]