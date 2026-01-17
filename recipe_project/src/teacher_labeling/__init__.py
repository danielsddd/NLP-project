"""
Teacher Labeling Module
=======================
Generate silver labels using LLM teachers (Gemini/GPT-4o).
"""

from .generate_labels import (
    SilverLabelGenerator,
    GeminiTeacher,
    OpenAITeacher,
    TeacherOutput,
    Modification,
    LabeledComment,
)

__all__ = [
    "SilverLabelGenerator",
    "GeminiTeacher", 
    "OpenAITeacher",
    "TeacherOutput",
    "Modification",
    "LabeledComment",
]
