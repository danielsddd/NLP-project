"""
Preprocessing Module
====================
Convert silver labels to BIO-tagged training data.

v5: Groups mods by source_comment (no more get_source_text),
    4-tier span finder, thread-level stratified split.
"""

from .prepare_data import (
    align_example,
    process_file,
    extract_thread_examples,
    find_span_in_text,
    normalize_hebrew,
    visualize_alignment,
    thread_level_split,
    BIO_LABEL2ID,
    BIO_ID2LABEL,
    IO_LABEL2ID,
    IO_ID2LABEL,
    VALID_ASPECTS,
)

__all__ = [
    "align_example",
    "process_file",
    "extract_thread_examples",
    "find_span_in_text",
    "normalize_hebrew",
    "visualize_alignment",
    "thread_level_split",
    "BIO_LABEL2ID",
    "BIO_ID2LABEL",
    "IO_LABEL2ID",
    "IO_ID2LABEL",
    "VALID_ASPECTS",
]