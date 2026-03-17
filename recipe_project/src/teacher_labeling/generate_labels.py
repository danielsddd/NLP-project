#!/usr/bin/env python3
"""
Dual-Teacher Silver Label Generator — v5
=========================================
Generates silver labels using TWO LLM teachers and keeps agreement-based labels.

Teachers:
  1. Gemini 3.1 Flash Lite  (Google AI Studio — free tier, 500 RPD / 15 RPM)
  2. Llama 3.3 70B          (Groq — free tier, fast)

Modes:
  --teacher groq     → Groq only  (label all 5,000 — primary)
  --teacher gemini   → Gemini only (label subset for agreement)
  --teacher both     → Both teachers, keep only agreements (highest quality)

Recommended workflow:
    # Step 1: Label ALL threads with Groq (~3 hours)
    python -m src.teacher_labeling.generate_labels \
        -i data/raw_youtube/threads.jsonl \
        --groq-key YOUR_KEY --teacher groq

    # Step 2: Label 500-thread subset with Gemini for agreement analysis
    python -m src.teacher_labeling.generate_labels \
        -i data/raw_youtube/threads.jsonl \
        --gemini-key YOUR_KEY --teacher gemini --limit 500 \
        -o data/silver_labels/gemini_subset.jsonl

    # Step 3: Compute agreement between the two
    python -m src.teacher_labeling.generate_labels \
        --compare \
        --groq-file data/silver_labels/teacher_output.jsonl \
        --gemini-file data/silver_labels/gemini_subset.jsonl \
        -o data/silver_labels/agreement_report.json
"""

from html import parser
import json
import re
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()  # reads .env file automatically

# ---------------------------------------------------------------------------
# Optional imports — fail gracefully so the user gets a clear message
# ---------------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# =============================================================================
# DATA CLASSES  (exported by __init__.py)
# =============================================================================

@dataclass
class Modification:
    """A single extracted modification."""
    span: str                           # Exact Hebrew text substring
    aspect: str                         # SUBSTITUTION | QUANTITY | TECHNIQUE | ADDITION
    source_comment: str = "top"         # "top" or "reply_1", "reply_2", …
    confidence: float = 0.0

@dataclass
class TeacherOutput:
    """Parsed output from one teacher for a single thread."""
    modifications: List[Modification] = field(default_factory=list)
    has_modification: bool = False
    thread_type: str = "statement"      # question | statement | mixed
    raw_response: Optional[str] = None
    error: Optional[str] = None

@dataclass
class LabeledComment:
    """A thread with its silver labels."""
    thread_id: str
    video_id: str
    channel_id: str
    video_title: str
    channel_title: str
    top_comment_text: str
    replies_texts: List[str] = field(default_factory=list)
    has_creator_reply: bool = False
    total_likes: int = 0
    teacher_output: Optional[Dict] = None       # final merged / single output
    gemini_output: Optional[Dict] = None        # raw Gemini result
    groq_output: Optional[Dict] = None          # raw Groq result
    agreement: Optional[str] = None             # "full" | "partial" | "none" | "single"
    teacher_model: str = ""
    labeled_at: str = ""


# =============================================================================
# VALID ASPECTS
# =============================================================================

VALID_ASPECTS = {"SUBSTITUTION", "QUANTITY", "TECHNIQUE", "ADDITION"}


# =============================================================================
# SHARED PROMPT — identical for both teachers to ensure fair comparison
# =============================================================================

SYSTEM_PROMPT = """You are an expert culinary NLP assistant specializing in Hebrew text.
Your task: analyze comment threads from cooking videos and extract recipe modification suggestions.

For each thread you receive:
- [TOP COMMENT]: The main comment
- [REPLY N, user/creator]: Replies to the comment

RULES:
1. If the top comment is a QUESTION (contains ? or words like אפשר/האם/כדאי/אפשרי), extract modifications from the REPLIES only. The question itself is NOT a modification.
2. If the top comment is a STATEMENT about what the user did differently, extract from the top comment.
3. A question with no meaningful replies (or replies like emojis, "me too") → has_modification: false.
4. "span" must be the EXACT Hebrew text substring copied from the source comment. Do NOT paraphrase.
5. Praise ("it was delicious", "תודה", "יצא מעולה") is NOT a modification.
6. aspect must be one of: SUBSTITUTION, QUANTITY, TECHNIQUE, ADDITION.
7. source_comment must be "top" or "reply_1", "reply_2", etc.
8. confidence is 0.0-1.0. Creator replies get 0.90+.

OUTPUT FORMAT (strict JSON, no markdown, no explanation):
{
  "modifications": [
    {"span": "<exact text>", "aspect": "SUBSTITUTION|QUANTITY|TECHNIQUE|ADDITION", "source_comment": "top|reply_N", "confidence": 0.0-1.0}
  ],
  "has_modification": true|false,
  "thread_type": "question|statement|mixed"
}

EXAMPLES:

Thread: [TOP COMMENT] "אפשר במקום חמאה שמן קוקוס?"
[REPLY 1, user] "כן! אותה כמות, יצא מעולה"
Output: {"modifications": [{"span": "שמן קוקוס", "aspect": "SUBSTITUTION", "source_comment": "reply_1", "confidence": 0.85}], "has_modification": true, "thread_type": "question"}

Thread: [TOP COMMENT] "שמתי כפול סוכר ויצא מתוק מדי"
Output: {"modifications": [{"span": "כפול סוכר", "aspect": "QUANTITY", "source_comment": "top", "confidence": 0.90}], "has_modification": true, "thread_type": "statement"}

Thread: [TOP COMMENT] "אפשר לעשות בלי ביצים?"
Output: {"modifications": [], "has_modification": false, "thread_type": "question"}

Thread: [TOP COMMENT] "יצא מעולה! תודה רבה!"
Output: {"modifications": [], "has_modification": false, "thread_type": "statement"}

Thread: [TOP COMMENT] "השתמשתי בקמח כוסמין במקום רגיל והוספתי קינמון"
Output: {"modifications": [{"span": "קמח כוסמין", "aspect": "SUBSTITUTION", "source_comment": "top", "confidence": 0.90}, {"span": "הוספתי קינמון", "aspect": "ADDITION", "source_comment": "top", "confidence": 0.85}], "has_modification": true, "thread_type": "statement"}"""


# =============================================================================
# THREAD FORMATTING  (shared)
# =============================================================================

def format_thread(thread: dict) -> str:
    """Format a thread dict into the prompt text both teachers receive."""
    lines = [f'[TOP COMMENT] "{thread["top_comment"]["text"]}"']
    for i, reply in enumerate(thread.get("replies", []), 1):
        role = "creator" if reply.get("is_creator") else "user"
        lines.append(f'[REPLY {i}, {role}] "{reply["text"]}"')
    return "\n".join(lines)


# =============================================================================
# JSON PARSING  (shared)
# =============================================================================

def parse_json_response(raw: str) -> Optional[dict]:
    """Robustly parse JSON from an LLM response that might contain markdown."""
    text = raw.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    if text.startswith("```"):
        # Remove opening fence (with optional "json" label)
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        # Remove closing fence
        text = re.sub(r'\n?```\s*$', '', text)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in the text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def validate_teacher_output(parsed: dict) -> Optional[dict]:
    """Validate and sanitize a parsed teacher output dict."""
    if not isinstance(parsed, dict):
        return None

    # Ensure required fields
    result = {
        "modifications": [],
        "has_modification": bool(parsed.get("has_modification", False)),
        "thread_type": parsed.get("thread_type", "statement"),
    }

    # Validate each modification
    for mod in parsed.get("modifications", []):
        if not isinstance(mod, dict):
            continue
        span = mod.get("span", "").strip()
        aspect = mod.get("aspect", "").upper().strip()
        if not span or aspect not in VALID_ASPECTS:
            continue
        result["modifications"].append({
            "span": span,
            "aspect": aspect,
            "source_comment": mod.get("source_comment", "top"),
            "confidence": min(1.0, max(0.0, float(mod.get("confidence", 0.5)))),
        })

    # Fix consistency: if mods exist, has_modification must be True
    if result["modifications"]:
        result["has_modification"] = True
    if not result["modifications"]:
        result["has_modification"] = False

    return result


# =============================================================================
# TEACHER: GEMINI 3.1 FLASH LITE  (free via AI Studio)
# =============================================================================
class GeminiTeacher:
    """
    Teacher model using Google Gemini 3.1 Flash Lite.

    Free tier limits (Google AI Studio):
        RPM:  15
        RPD:  500
        TPM:  250,000

    For 500-thread agreement subset → fits in 1 day.
    """

    NAME = "gemini-3.1-flash-lite-preview"
    # 15 RPM → 4 seconds between calls is safe
    MIN_DELAY = 4.5

    def __init__(self, api_key: str, model_name: str = None):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai not installed. Run: pip install google-genai"
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name or self.NAME

    def label(self, thread: dict) -> TeacherOutput:
        """Label a single thread. Returns TeacherOutput."""
        prompt_text = format_thread(thread)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_text,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text
            parsed = parse_json_response(raw)
            if parsed is None:
                return TeacherOutput(error=f"JSON parse failed: {raw[:200]}", raw_response=raw)

            validated = validate_teacher_output(parsed)
            if validated is None:
                return TeacherOutput(error="Validation failed", raw_response=raw)

            mods = [Modification(**m) for m in validated["modifications"]]
            return TeacherOutput(
                modifications=mods,
                has_modification=validated["has_modification"],
                thread_type=validated["thread_type"],
                raw_response=raw,
            )
        except Exception as e:
            return TeacherOutput(error=str(e))

# =============================================================================
# TEACHER: GROQ  (Llama 3.3 70B Versatile)
# =============================================================================

class GroqTeacher:
    """
    Teacher model using Llama 3.3 70B via Groq.

    Free tier — fast inference, generous limits.
    Fallback model: meta-llama/llama-4-scout-17b-16e-instruct
    """

    NAME = "llama-3.3-70b-versatile"
    # ~30 RPM is safe for free tier
    MIN_DELAY = 2.5

    def __init__(self, api_key: str, model_name: str = None):
        if not GROQ_AVAILABLE:
            raise ImportError(
                "groq not installed. Run: pip install groq"
            )
        self.client = Groq(api_key=api_key)
        self.model_name = model_name or self.NAME

    def label(self, thread: dict) -> TeacherOutput:
        """Label a single thread. Returns TeacherOutput."""
        prompt_text = format_thread(thread)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            raw = completion.choices[0].message.content
            parsed = parse_json_response(raw)
            if parsed is None:
                return TeacherOutput(error=f"JSON parse failed: {raw[:200]}", raw_response=raw)

            validated = validate_teacher_output(parsed)
            if validated is None:
                return TeacherOutput(error="Validation failed", raw_response=raw)

            mods = [Modification(**m) for m in validated["modifications"]]
            return TeacherOutput(
                modifications=mods,
                has_modification=validated["has_modification"],
                thread_type=validated["thread_type"],
                raw_response=raw,
            )
        except Exception as e:
            return TeacherOutput(error=str(e))


# =============================================================================
# AGREEMENT LOGIC
# =============================================================================

def compute_agreement(gemini_out: TeacherOutput, groq_out: TeacherOutput) -> dict:
    """
    Compare two teacher outputs and return merged result + agreement level.

    Agreement levels:
      - "full"    : both agree on has_modification AND same aspects found
      - "partial" : both agree on has_modification, but differ on some aspects
      - "none"    : disagree on has_modification entirely
    """
    # If either had an error, fall back to the one that worked
    if gemini_out.error and groq_out.error:
        return {"merged": None, "agreement": "both_error"}
    if gemini_out.error:
        return {"merged": _output_to_dict(groq_out), "agreement": "groq_only"}
    if groq_out.error:
        return {"merged": _output_to_dict(gemini_out), "agreement": "gemini_only"}

    # Both succeeded — compare
    g_has = gemini_out.has_modification
    q_has = groq_out.has_modification

    if g_has != q_has:
        # Disagree on whether modifications exist at all
        # Keep modifications if either found them (recall-oriented)
        if g_has:
            return {"merged": _output_to_dict(gemini_out), "agreement": "none"}
        else:
            return {"merged": _output_to_dict(groq_out), "agreement": "none"}

    if not g_has and not q_has:
        # Both agree: no modification
        return {
            "merged": {
                "modifications": [],
                "has_modification": False,
                "thread_type": gemini_out.thread_type,
            },
            "agreement": "full",
        }

    # Both found modifications — compare aspects
    g_aspects = {m.aspect for m in gemini_out.modifications}
    q_aspects = {m.aspect for m in groq_out.modifications}

    if g_aspects == q_aspects:
        agreement = "full"
    else:
        agreement = "partial"

    # Merge: keep modifications from BOTH teachers, deduplicate by aspect+source
    merged_mods = _merge_modifications(gemini_out.modifications, groq_out.modifications)

    return {
        "merged": {
            "modifications": [_mod_to_dict(m) for m in merged_mods],
            "has_modification": True,
            "thread_type": gemini_out.thread_type,
        },
        "agreement": agreement,
    }


def _merge_modifications(gemini_mods: List[Modification],
                         groq_mods: List[Modification]) -> List[Modification]:
    """
    Merge modifications from two teachers.
    If both found the same aspect from the same source, keep the one with
    higher confidence and boost it. If one found a mod the other didn't,
    include it but with a slight confidence penalty.
    """
    merged = {}

    for mod in gemini_mods:
        key = (mod.aspect, mod.source_comment)
        if key not in merged or mod.confidence > merged[key].confidence:
            merged[key] = Modification(
                span=mod.span,
                aspect=mod.aspect,
                source_comment=mod.source_comment,
                confidence=mod.confidence,
            )

    for mod in groq_mods:
        key = (mod.aspect, mod.source_comment)
        if key not in merged:
            # Only Groq found this — include but note lower confidence
            merged[key] = Modification(
                span=mod.span,
                aspect=mod.aspect,
                source_comment=mod.source_comment,
                confidence=mod.confidence * 0.9,  # slight penalty for single-teacher
            )
        else:
            # Both found it — boost confidence
            existing = merged[key]
            boosted_conf = min(1.0, (existing.confidence + mod.confidence) / 2 + 0.05)
            # Keep whichever span is longer (more informative)
            best_span = mod.span if len(mod.span) > len(existing.span) else existing.span
            merged[key] = Modification(
                span=best_span,
                aspect=mod.aspect,
                source_comment=mod.source_comment,
                confidence=boosted_conf,
            )

    return list(merged.values())


def _output_to_dict(output: TeacherOutput) -> dict:
    return {
        "modifications": [_mod_to_dict(m) for m in output.modifications],
        "has_modification": output.has_modification,
        "thread_type": output.thread_type,
    }


def _mod_to_dict(m: Modification) -> dict:
    return {
        "span": m.span,
        "aspect": m.aspect,
        "source_comment": m.source_comment,
        "confidence": round(m.confidence, 3),
    }


# =============================================================================
# SILVER LABEL GENERATOR  (orchestrator)
# =============================================================================

class SilverLabelGenerator:
    """
    Orchestrates one or two teachers to produce silver labels.

    Usage:
        gen = SilverLabelGenerator(groq_key="...", mode="groq")
        record = gen.label_thread(thread_dict)
    """

    def __init__(self, gemini_key: str = None, groq_key: str = None,
                 mode: str = "groq", groq_model: str = None,
                 gemini_model: str = None):
        self.mode = mode
        self.gemini = None
        self.groq = None

        if mode in ("gemini", "both"):
            if not gemini_key:
                raise ValueError("--gemini-key required for Gemini teacher")
            self.gemini = GeminiTeacher(api_key=gemini_key, model_name=gemini_model)

        if mode in ("groq", "both"):
            if not groq_key:
                raise ValueError("--groq-key required for Groq teacher")
            self.groq = GroqTeacher(api_key=groq_key, model_name=groq_model)

    def label_thread(self, thread: dict) -> dict:
        """
        Label a single thread with configured teacher(s).
        Returns a full record dict ready for JSONL output.
        """
        gemini_result = None
        groq_result = None
        teacher_output = None
        agreement = "single"

        # --- Gemini ---
        if self.gemini:
            gemini_result = self.gemini.label(thread)
            if self.mode == "gemini":
                time.sleep(self.gemini.MIN_DELAY)

        # --- Groq ---
        if self.groq:
            groq_result = self.groq.label(thread)
            if self.mode == "groq":
                time.sleep(self.groq.MIN_DELAY)

        # --- Merge / select ---
        if self.mode == "both" and gemini_result and groq_result:
            result = compute_agreement(gemini_result, groq_result)
            teacher_output = result["merged"]
            agreement = result["agreement"]
            # Rate limit: respect the slower teacher (Gemini)
            time.sleep(GeminiTeacher.MIN_DELAY)
        elif gemini_result and not gemini_result.error:
            teacher_output = _output_to_dict(gemini_result)
        elif groq_result and not groq_result.error:
            teacher_output = _output_to_dict(groq_result)

        # --- Build model name string ---
        if self.mode == "both":
            gname = self.gemini.model_name if self.gemini else "gemini"
            qname = self.groq.model_name if self.groq else "groq"
            teacher_model = f"{gname}+{qname}"
        elif self.mode == "gemini":
            teacher_model = self.gemini.model_name if self.gemini else "gemini"
        else:
            teacher_model = self.groq.model_name if self.groq else "groq"

        # --- Build output record ---
        record = {
            "thread_id": thread["thread_id"],
            "video_id": thread["video_id"],
            "channel_id": thread["channel_id"],
            "video_title": thread["video_title"],
            "channel_title": thread["channel_title"],
            "top_comment_text": thread["top_comment"]["text"],
            "replies_texts": [r["text"] for r in thread.get("replies", [])],
            "has_creator_reply": thread.get("has_creator_reply", False),
            "total_likes": thread.get("total_likes", 0),
            "teacher_output": teacher_output,
            "agreement": agreement,
            "teacher_model": teacher_model,
            "labeled_at": datetime.now(timezone.utc).isoformat(),
        }

        # Include per-teacher outputs when running both (for analysis)
        if self.mode == "both":
            record["gemini_output"] = (
                _output_to_dict(gemini_result) if gemini_result and not gemini_result.error else None
            )
            record["groq_output"] = (
                _output_to_dict(groq_result) if groq_result and not groq_result.error else None
            )

        return record


# =============================================================================
# FILE I/O HELPERS
# =============================================================================

def append_jsonl(data: dict, path: Path):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
        f.flush()


def load_existing_ids(path: Path) -> set:
    ids = set()
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["thread_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


# =============================================================================
# OFFLINE AGREEMENT COMPARISON
# =============================================================================

def compare_files(groq_file: str, gemini_file: str, output_file: str):
    """
    Compare two separately-generated label files and produce an agreement report.

    Use this after running Groq on all 5,000 and Gemini on a 500 subset.
    Matches records by thread_id and computes agreement statistics.
    """
    # Load Groq labels indexed by thread_id
    groq_labels = {}
    with open(groq_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                groq_labels[rec["thread_id"]] = rec.get("teacher_output", {})
            except (json.JSONDecodeError, KeyError):
                continue

    # Load Gemini labels indexed by thread_id
    gemini_labels = {}
    with open(gemini_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                gemini_labels[rec["thread_id"]] = rec.get("teacher_output", {})
            except (json.JSONDecodeError, KeyError):
                continue

    # Find overlapping thread IDs
    common_ids = set(groq_labels.keys()) & set(gemini_labels.keys())
    print(f"Groq labels:   {len(groq_labels)}")
    print(f"Gemini labels: {len(gemini_labels)}")
    print(f"Overlapping:   {len(common_ids)}")

    if not common_ids:
        print("No overlapping thread IDs found. Cannot compare.")
        return

    # Compare
    stats = {
        "total_compared": len(common_ids),
        "has_mod_agree": 0,
        "has_mod_disagree": 0,
        "both_no_mod": 0,
        "both_has_mod": 0,
        "aspect_full_agree": 0,
        "aspect_partial_agree": 0,
        "aspect_disagree": 0,
        "examples": {
            "full_agreement": [],
            "disagreement": [],
        },
    }

    for tid in sorted(common_ids):
        g_out = groq_labels[tid]
        m_out = gemini_labels[tid]

        g_has = g_out.get("has_modification", False) if g_out else False
        m_has = m_out.get("has_modification", False) if m_out else False

        if g_has == m_has:
            stats["has_mod_agree"] += 1
            if not g_has:
                stats["both_no_mod"] += 1
            else:
                stats["both_has_mod"] += 1
                # Compare aspects
                g_aspects = {m.get("aspect") for m in g_out.get("modifications", [])}
                m_aspects = {m.get("aspect") for m in m_out.get("modifications", [])}
                if g_aspects == m_aspects:
                    stats["aspect_full_agree"] += 1
                    if len(stats["examples"]["full_agreement"]) < 5:
                        stats["examples"]["full_agreement"].append({
                            "thread_id": tid,
                            "groq": g_out,
                            "gemini": m_out,
                        })
                elif g_aspects & m_aspects:
                    stats["aspect_partial_agree"] += 1
                else:
                    stats["aspect_disagree"] += 1
        else:
            stats["has_mod_disagree"] += 1
            if len(stats["examples"]["disagreement"]) < 5:
                stats["examples"]["disagreement"].append({
                    "thread_id": tid,
                    "groq": g_out,
                    "gemini": m_out,
                })

    # Compute rates
    total = stats["total_compared"]
    stats["has_mod_agreement_rate"] = round(stats["has_mod_agree"] / total, 3)
    if stats["both_has_mod"] > 0:
        stats["aspect_agreement_rate"] = round(
            stats["aspect_full_agree"] / stats["both_has_mod"], 3
        )
    else:
        stats["aspect_agreement_rate"] = None

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"TEACHER AGREEMENT REPORT")
    print(f"{'='*60}")
    print(f"Threads compared:          {total}")
    print(f"has_modification agree:     {stats['has_mod_agree']}/{total} "
          f"({stats['has_mod_agreement_rate']*100:.1f}%)")
    print(f"  Both no modification:    {stats['both_no_mod']}")
    print(f"  Both has modification:   {stats['both_has_mod']}")
    print(f"  Disagree:                {stats['has_mod_disagree']}")
    if stats["both_has_mod"] > 0:
        print(f"Aspect agreement (when both found mods):")
        print(f"  Full match:              {stats['aspect_full_agree']}/{stats['both_has_mod']} "
              f"({stats['aspect_agreement_rate']*100:.1f}%)")
        print(f"  Partial overlap:         {stats['aspect_partial_agree']}")
        print(f"  No overlap:              {stats['aspect_disagree']}")
    print(f"\nSaved to: {output_path}")
    print(f"{'='*60}")


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

def process_file(input_path: str, output_path: str,
                 gemini_key: str = None, groq_key: str = None,
                 mode: str = "groq", groq_model: str = None,
                 gemini_model: str = None,
                 limit: int = None, skip_existing: bool = True):
    """Main processing loop — labels threads and writes JSONL output."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    existing_ids = set()
    if skip_existing:
        existing_ids = load_existing_ids(output_path)
        if existing_ids:
            print(f"Resuming: {len(existing_ids)} already labeled, will skip")

    # Load threads
    threads = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                t = json.loads(line)
                if t["thread_id"] not in existing_ids:
                    threads.append(t)
            except (json.JSONDecodeError, KeyError):
                continue

    if limit:
        threads = threads[:limit]

    if not threads:
        print("No new threads to process.")
        return

    # Initialize generator
    generator = SilverLabelGenerator(
        gemini_key=gemini_key,
        groq_key=groq_key,
        mode=mode,
        groq_model=groq_model,
        gemini_model=gemini_model,
    )

    # Estimate time
    if mode == "both":
        delay = GeminiTeacher.MIN_DELAY   # bottleneck is Gemini
    elif mode == "gemini":
        delay = GeminiTeacher.MIN_DELAY
    else:
        delay = GroqTeacher.MIN_DELAY
    est_minutes = (len(threads) * delay) / 60
    print(f"Processing {len(threads)} threads with teacher={mode}")
    if mode == "gemini":
        teacher_name = gemini_model or GeminiTeacher.NAME
    elif mode == "groq":
        teacher_name = groq_model or GroqTeacher.NAME
    else:
        teacher_name = f"{gemini_model or GeminiTeacher.NAME} + {groq_model or GroqTeacher.NAME}"
    print(f"Model(s): {teacher_name}")
    print(f"Estimated time: ~{est_minutes:.0f} minutes ({delay}s per thread)")
    print(f"Output: {output_path}")
    print()

    # Stats
    stats = {
        "total": 0,
        "with_mods": 0,
        "no_mods": 0,
        "errors": 0,
        "agreement": {"full": 0, "partial": 0, "none": 0,
                       "single": 0, "gemini_only": 0, "groq_only": 0, "both_error": 0},
        "aspects": {"SUBSTITUTION": 0, "QUANTITY": 0, "TECHNIQUE": 0, "ADDITION": 0},
        "mode": mode,
        "teacher_model": teacher_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    for i, thread in enumerate(threads):
        record = generator.label_thread(thread)
        stats["total"] += 1

        # Track agreement
        agr = record.get("agreement", "single")
        if agr in stats["agreement"]:
            stats["agreement"][agr] += 1

        # Track content
        teacher_out = record.get("teacher_output")
        if teacher_out is None:
            stats["errors"] += 1
            print(f"  [{i+1}/{len(threads)}] ERROR: {thread['thread_id']}")
        elif teacher_out.get("has_modification"):
            stats["with_mods"] += 1
            for mod in teacher_out.get("modifications", []):
                aspect = mod.get("aspect", "")
                if aspect in stats["aspects"]:
                    stats["aspects"][aspect] += 1
        else:
            stats["no_mods"] += 1

        # Save record
        append_jsonl(record, output_path)

        # Progress
        if (i + 1) % 25 == 0 or (i + 1) == len(threads):
            pct_mod = stats["with_mods"] / stats["total"] * 100 if stats["total"] else 0
            msg = (f"  [{i+1}/{len(threads)}] "
                   f"mods={stats['with_mods']} ({pct_mod:.0f}%) "
                   f"errors={stats['errors']}")
            if mode == "both":
                msg += (f" agree: full={stats['agreement']['full']} "
                        f"partial={stats['agreement']['partial']} "
                        f"none={stats['agreement']['none']}")
            print(msg)

    # Finalize stats
    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    stats_path = output_path.parent / "generation_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Done: {stats['total']} threads processed")
    print(f"   With modifications: {stats['with_mods']} "
          f"({stats['with_mods']/max(1,stats['total'])*100:.1f}%)")
    print(f"   No modifications:   {stats['no_mods']}")
    print(f"   Errors:             {stats['errors']}")
    print(f"   Aspects: {stats['aspects']}")
    if mode == "both":
        print(f"   Agreement: full={stats['agreement']['full']}, "
              f"partial={stats['agreement']['partial']}, "
              f"none={stats['agreement']['none']}")
    print(f"   Output: {output_path}")
    print(f"   Stats:  {stats_path}")
    print(f"{'='*60}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dual-Teacher Silver Label Generator (Gemini + Groq)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Label all 5,000 threads with Groq (primary — ~3 hours)
  python -m src.teacher_labeling.generate_labels \\
      -i data/raw_youtube/threads.jsonl \\
      --groq-key KEY --teacher groq

  # Label 500-thread subset with Gemini (agreement analysis — 1 day)
  python -m src.teacher_labeling.generate_labels \\
      -i data/raw_youtube/threads.jsonl \\
      --gemini-key KEY --teacher gemini --limit 500 \\
      -o data/silver_labels/gemini_subset.jsonl

  # Compare the two runs offline
  python -m src.teacher_labeling.generate_labels \\
      --compare \\
      --groq-file data/silver_labels/teacher_output.jsonl \\
      --gemini-file data/silver_labels/gemini_subset.jsonl \\
      -o data/silver_labels/agreement_report.json

  # Both teachers on a small test batch
  python -m src.teacher_labeling.generate_labels \\
      -i data/raw_youtube/threads.jsonl \\
      --gemini-key KEY1 --groq-key KEY2 \\
      --teacher both --limit 20

  # Test with 5 threads first
  python -m src.teacher_labeling.generate_labels \\
      -i data/raw_youtube/threads.jsonl \\
      --groq-key KEY --teacher groq --limit 5
        """,
    )

    # Main mode: label threads
    parser.add_argument("--input", "-i", help="Input threads JSONL file")
    parser.add_argument("--output", "-o", default="data/silver_labels/teacher_output.jsonl",
                        help="Output JSONL path (default: data/silver_labels/teacher_output.jsonl)")
    parser.add_argument("--gemini-key", help="Google AI Studio API key (for Gemini)")
    parser.add_argument("--groq-key", help="Groq API key (for Llama 3.3 70B)")
    parser.add_argument("--teacher", choices=["gemini", "groq", "both"], default="groq",
                        help="Which teacher(s) to use (default: groq)")
    parser.add_argument("--groq-model", default="llama-3.3-70b-versatile",
                        help="Groq model name (default: llama-3.3-70b-versatile)")
    parser.add_argument("--gemini-model", default="gemini-3.1-flash-lite-preview",
                        help="Gemini model name (default: gemini-3.1-flash-lite)")
    parser.add_argument("--limit", type=int, help="Max threads to process (for testing)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Don't skip already-labeled threads (re-process all)")

    # Comparison mode: compare two label files offline
    parser.add_argument("--compare", action="store_true",
                        help="Compare two label files for agreement (offline mode)")
    parser.add_argument("--groq-file", help="Groq labels JSONL (for --compare)")
    parser.add_argument("--gemini-file", help="Gemini labels JSONL (for --compare)")

    args = parser.parse_args()

    # --- Comparison mode ---
    if args.compare:
        if not args.groq_file or not args.gemini_file:
            parser.error("--compare requires --groq-file and --gemini-file")
        compare_files(args.groq_file, args.gemini_file, args.output)
        return

    # --- Labeling mode ---
    if not args.input:
        parser.error("--input is required for labeling mode")

    if args.teacher in ("gemini", "both") and not args.gemini_key:
        parser.error("--gemini-key is required when --teacher is 'gemini' or 'both'")
    if args.teacher in ("groq", "both") and not args.groq_key:
        parser.error("--groq-key is required when --teacher is 'groq' or 'both'")

    process_file(
        input_path=args.input,
        output_path=args.output,
        gemini_key=args.gemini_key,
        groq_key=args.groq_key,
        mode=args.teacher,
        groq_model=args.groq_model,
        gemini_model=args.gemini_model,
        limit=args.limit,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()