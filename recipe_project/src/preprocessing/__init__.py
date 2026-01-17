"""
Preprocessing Module
====================
Convert silver labels to BIO-tagged training data.
"""

from .prepare_data import (
    DataPreprocessor,
    ProcessedExample,
    AlignmentStats,
    align_spans_to_bio,
    visualize_alignment,
    BIO_LABEL2ID,
    BIO_ID2LABEL,
    IO_LABEL2ID,
    IO_ID2LABEL,
)

__all__ = [
    "DataPreprocessor",
    "ProcessedExample",
    "AlignmentStats",
    "align_spans_to_bio",
    "visualize_alignment",
    "BIO_LABEL2ID",
    "BIO_ID2LABEL",
    "IO_LABEL2ID",
    "IO_ID2LABEL",
]
