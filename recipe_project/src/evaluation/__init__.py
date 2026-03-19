"""
Evaluation Module
=================
Comprehensive evaluation and error analysis.
"""

from .evaluate import (
    evaluate,
    predict_batch,
    to_tag_sequences,
    labels_to_spans,
    analyze_errors,
    bootstrap_f1_ci,
    ASPECTS,
    BIO_ID2LABEL,
)

__all__ = [
    "evaluate",
    "predict_batch",
    "to_tag_sequences",
    "labels_to_spans",
    "analyze_errors",
    "bootstrap_f1_ci",
    "ASPECTS",
    "BIO_ID2LABEL",
]