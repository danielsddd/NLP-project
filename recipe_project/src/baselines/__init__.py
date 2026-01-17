"""
Baselines Module
================
Simple baseline implementations for comparison.
"""

from .run_baselines import (
    BaselineEvaluator,
    BaselineResult,
    MajorityBaseline,
    RandomBaseline,
    KeywordBaseline,
    ASPECT_KEYWORDS,
)

__all__ = [
    "BaselineEvaluator",
    "BaselineResult",
    "MajorityBaseline",
    "RandomBaseline",
    "KeywordBaseline",
    "ASPECT_KEYWORDS",
]
