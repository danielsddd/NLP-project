"""
Teacher Labeling Module
=======================
Generate silver labels using dual LLM teachers.

Teachers:
  - GeminiTeacher:  Google Gemini 3.1 Flash Lite (free via AI Studio, 500 RPD)
  - GroqTeacher:    Llama 3.3 70B Versatile (free via Groq, fast)

Orchestrator:
  - SilverLabelGenerator: Runs one or both teachers with agreement logic

Usage:
    from src.teacher_labeling import SilverLabelGenerator

    # Groq only (primary — fast, label all 5,000)
    gen = SilverLabelGenerator(groq_key="...", mode="groq")
    record = gen.label_thread(thread_dict)

    # Gemini only (subset for agreement analysis)
    gen = SilverLabelGenerator(gemini_key="...", mode="gemini")

    # Both teachers (highest quality — rate limited by Gemini)
    gen = SilverLabelGenerator(gemini_key="...", groq_key="...", mode="both")
"""

from .generate_labels import (
    SilverLabelGenerator,
    GeminiTeacher,
    GroqTeacher,
    TeacherOutput,
    Modification,
    LabeledComment,
    compute_agreement,
    compare_files,
)

__all__ = [
    "SilverLabelGenerator",
    "GeminiTeacher",
    "GroqTeacher",
    "TeacherOutput",
    "Modification",
    "LabeledComment",
    "compute_agreement",
    "compare_files",
]