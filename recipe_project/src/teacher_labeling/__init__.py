"""
Teacher Labeling Module
=======================
Three-pass silver label generation with majority vote.
Pass 1: Gemini (primary) + Groq (fallback)
Pass 2: Same Gemini temp=0.3 (intra-annotator)
Pass 3: Cerebras Qwen 235B (inter-annotator)
Final: Majority vote → flag disagreements for manual review
"""

from .generate_labels import (
    GeminiTeacher,
    GroqTeacher,
    CerebrasTeacher,
    TeacherOutput,
    Modification,
    LabeledComment,
    compute_pairwise_agreement,
    compute_majority_vote,
)

__all__ = [
    "GeminiTeacher",
    "GroqTeacher",
    "CerebrasTeacher",
    "TeacherOutput",
    "Modification",
    "LabeledComment",
    "compute_pairwise_agreement",
    "compute_majority_vote",
]