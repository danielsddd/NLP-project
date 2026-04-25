"""
Preprocessing Module
====================
Convert silver labels to BIO-tagged training data.

v5: Groups mods by source_comment (no more get_source_text),
    4-tier span finder, thread-level stratified split.
"""

# ── Legacy prepare_data (constants + alignment) ──────────────────────────────
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

# ── prepare_data_merged (merged-file pipeline) ────────────────────────────────
from .prepare_data_merged import (
    compute_class_weights,
    compute_uniform_weights,
    load_threads,
    downsample_negatives,
    process_merged,
)

__all__ = [
    # From prepare_data
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
    # From prepare_data_merged
    "compute_class_weights",
    "compute_uniform_weights",
    "load_threads",
    "downsample_negatives",
    "process_merged",
]
