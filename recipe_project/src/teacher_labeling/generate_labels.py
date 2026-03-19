#!/usr/bin/env python3
"""
Teacher Silver Label Generator — v9 (Three-Pass with Majority Vote)
====================================================================
Pass 1: Label all threads with Gemini (primary) + Groq fallback. Batched.
Pass 2: Re-label with same Gemini at temp=0.3 (intra-annotator agreement).
Pass 3: Re-label with Cerebras Qwen 235B (inter-annotator agreement).
Final: Majority vote (2/3 agree) → final label. All-3-disagree → manual review.

Usage:
    # Pass 1 — label everything (Gemini primary, Groq fallback):
    python -m src.teacher_labeling.generate_labels \
        -i data/raw_youtube/threads.jsonl --limit 5000 --batch-size 20

    # Pass 2 — same Gemini, different temperature (intra-annotator):
    python -m src.teacher_labeling.generate_labels --second-pass --limit 5000 --batch-size 20

    # Pass 3 — Cerebras Qwen 235B (inter-annotator):
    python -m src.teacher_labeling.generate_labels --third-pass --limit 5000 --batch-size 20

    # Compute majority vote + export disagreements:
    python -m src.teacher_labeling.generate_labels --finalize

    # Export only needs-review records:
    python -m src.teacher_labeling.generate_labels --export-review

    # View agreement stats:
    python -m src.teacher_labeling.generate_labels --agreement-stats

API keys read from .env automatically:
    GOOGLE_API_KEY=your_gemini_key
    GROQ_API_KEY=your_groq_key
    CEREBRAS_API_KEY=your_cerebras_key
"""

import json
import re
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

# ---------------------------------------------------------------------------
# Optional imports
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

try:
    from openai import OpenAI as OpenAIClient
    OPENAI_COMPAT_AVAILABLE = True
except ImportError:
    OPENAI_COMPAT_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

BATCH_SIZE = 5
DEFAULT_OUTPUT = "data/silver_labels/teacher_output.jsonl"
REVIEW_OUTPUT = "data/silver_labels/needs_review.jsonl"

GEMINI_FAMILY = "gemini"
GROQ_FAMILY = "groq"
CEREBRAS_FAMILY = "cerebras"


# =============================================================================
# DATA CLASSES (exported by __init__.py)
# =============================================================================

@dataclass
class Modification:
    """A single extracted modification."""
    span: str
    aspect: str
    source_comment: str = "top"
    confidence: float = 0.0

@dataclass
class TeacherOutput:
    """Parsed output from one teacher for a single thread."""
    modifications: List[Modification] = field(default_factory=list)
    has_modification: bool = False
    thread_type: str = "statement"
    raw_response: Optional[str] = None
    error: Optional[str] = None
    is_rate_limit: bool = False

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
    teacher_output: Optional[dict] = None
    teacher_model: str = ""
    labeled_at: str = ""


# =============================================================================
# VALID ASPECTS
# =============================================================================

VALID_ASPECTS = {"SUBSTITUTION", "QUANTITY", "TECHNIQUE", "ADDITION"}


# =============================================================================
# SYSTEM PROMPT — supports batched threads
# =============================================================================

SYSTEM_PROMPT = """You are an expert culinary NLP assistant specializing in Hebrew text.
Your task: analyze comment threads from cooking videos and extract recipe modification suggestions.

You will receive MULTIPLE threads in one request, each marked with === THREAD N (ID: xxx) ===.
You must return a JSON array with one result object per thread, in the same order.

For each thread:
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

OUTPUT FORMAT — a JSON array, one object per thread (strict JSON, no markdown, no explanation):
[
  {
    "thread_id": "<id from header>",
    "modifications": [
      {"span": "<exact text>", "aspect": "SUBSTITUTION|QUANTITY|TECHNIQUE|ADDITION", "source_comment": "top|reply_N", "confidence": 0.0-1.0}
    ],
    "has_modification": true|false,
    "thread_type": "question|statement|mixed"
  }
]

EXAMPLE INPUT:
=== THREAD 1 (ID: abc123) ===
[TOP COMMENT] "אפשר במקום חמאה שמן קוקוס?"
[REPLY 1, user] "כן! אותה כמות, יצא מעולה"

=== THREAD 2 (ID: def456) ===
[TOP COMMENT] "יצא מעולה! תודה רבה!"

EXAMPLE OUTPUT:
[
  {"thread_id": "abc123", "modifications": [{"span": "שמן קוקוס", "aspect": "SUBSTITUTION", "source_comment": "reply_1", "confidence": 0.85}], "has_modification": true, "thread_type": "question"},
  {"thread_id": "def456", "modifications": [], "has_modification": false, "thread_type": "statement"}
]"""


# =============================================================================
# THREAD FORMATTING
# =============================================================================

def format_thread(thread: dict) -> str:
    """Format a single thread into prompt text."""
    if "top_comment" in thread:
        lines = [f'[TOP COMMENT] "{thread["top_comment"]["text"]}"']
        for i, reply in enumerate(thread.get("replies", []), 1):
            role = "creator" if reply.get("is_creator") else "user"
            lines.append(f'[REPLY {i}, {role}] "{reply["text"]}"')
    else:
        lines = [f'[TOP COMMENT] "{thread["top_comment_text"]}"']
        for i, reply_text in enumerate(thread.get("replies_texts", []), 1):
            lines.append(f'[REPLY {i}, user] "{reply_text}"')
    return "\n".join(lines)


def format_batch(threads: List[dict]) -> str:
    """Format multiple threads into a single batched prompt."""
    parts = []
    for idx, thread in enumerate(threads, 1):
        tid = thread.get("thread_id", f"unknown_{idx}")
        header = f"=== THREAD {idx} (ID: {tid}) ==="
        body = format_thread(thread)
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


# =============================================================================
# JSON PARSING
# =============================================================================

def parse_json_response(raw: str) -> Any:
    """Robustly parse JSON (object or array) from an LLM response."""
    text = raw.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def validate_single_output(parsed: dict) -> Optional[dict]:
    """Validate and sanitize a parsed teacher output dict for one thread."""
    if not isinstance(parsed, dict):
        return None

    result = {
        "modifications": [],
        "has_modification": bool(parsed.get("has_modification", False)),
        "thread_type": parsed.get("thread_type", "statement"),
    }

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

    if result["modifications"]:
        result["has_modification"] = True
    if not result["modifications"]:
        result["has_modification"] = False

    return result


def parse_batch_response(raw: str, threads: List[dict]) -> Dict[str, Optional[dict]]:
    """Parse a batch response → {thread_id: validated_output or None}."""
    thread_ids = [t["thread_id"] for t in threads]
    results = {tid: None for tid in thread_ids}

    parsed = parse_json_response(raw)
    if parsed is None:
        return results

    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            tid = item.get("thread_id", "")
            if tid in results:
                validated = validate_single_output(item)
                if validated:
                    results[tid] = validated

        matched_count = sum(1 for v in results.values() if v is not None)
        if matched_count == 0 and len(parsed) == len(threads):
            for tid, item in zip(thread_ids, parsed):
                if isinstance(item, dict):
                    validated = validate_single_output(item)
                    if validated:
                        results[tid] = validated

    elif isinstance(parsed, dict) and len(threads) == 1:
        validated = validate_single_output(parsed)
        if validated:
            results[thread_ids[0]] = validated

    return results


# =============================================================================
# RATE LIMIT DETECTION
# =============================================================================

def is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a rate limit / quota error."""
    msg = str(error).lower()
    keywords = [
        "rate limit", "rate_limit", "ratelimit",
        "quota", "resource exhausted", "resourceexhausted",
        "429", "too many requests", "try again later",
        "tokens per minute", "requests per minute",
        "requests per day", "rpm", "rpd", "tpm",
    ]
    return any(kw in msg for kw in keywords)


def detect_model_family(model_name: str) -> str:
    """Detect whether a model name belongs to a known family."""
    if not model_name:
        return ""
    name = model_name.lower()
    if "gemini" in name:
        return GEMINI_FAMILY
    if "llama" in name or "mixtral" in name or "groq" in name:
        return GROQ_FAMILY
    if "qwen" in name or "cerebras" in name:
        return CEREBRAS_FAMILY
    return ""


# =============================================================================
# TEACHER: GEMINI
# =============================================================================

class GeminiTeacher:
    """Google Gemini — primary teacher (free tier)."""

    NAME = "gemini-3.1-flash-lite-preview"
    FAMILY = GEMINI_FAMILY
    MIN_DELAY = 6.0  # 15 RPM → safe margin

    def __init__(self, api_key: str, model_name: str = None, temperature: float = 0.1):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai not installed. Run: pip install google-genai")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name or self.NAME
        self.temperature = temperature

    def label_batch(self, threads: List[dict]) -> tuple:
        """Returns (results_dict, error_str, is_rate_limit)."""
        prompt_text = format_batch(threads)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_text,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=self.temperature,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text
            results = parse_batch_response(raw, threads)
            return results, None, False
        except Exception as e:
            return None, str(e), is_rate_limit_error(e)


# =============================================================================
# TEACHER: GROQ
# =============================================================================

class GroqTeacher:
    """Groq (Llama 3.3 70B) — fallback teacher."""

    NAME = "llama-3.3-70b-versatile"
    FAMILY = GROQ_FAMILY
    MIN_DELAY = 2.5

    def __init__(self, api_key: str, model_name: str = None):
        if not GROQ_AVAILABLE:
            raise ImportError("groq not installed. Run: pip install groq")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name or self.NAME

    def label_batch(self, threads: List[dict]) -> tuple:
        """Returns (results_dict, error_str, is_rate_limit)."""
        prompt_text = format_batch(threads)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            raw = completion.choices[0].message.content
            results = parse_batch_response(raw, threads)
            return results, None, False
        except Exception as e:
            return None, str(e), is_rate_limit_error(e)


# =============================================================================
# TEACHER: CEREBRAS (Qwen 3 235B via OpenAI-compatible API)
# =============================================================================

class CerebrasTeacher:
    """Cerebras Qwen 3 235B — third annotator for majority vote."""

    NAME = "qwen-3-235b-a22b-instruct-2507"
    FAMILY = CEREBRAS_FAMILY
    MIN_DELAY = 8.0  

    def __init__(self, api_key: str, model_name: str = None):
        if not OPENAI_COMPAT_AVAILABLE:
            raise ImportError("openai not installed. Run: pip install openai")
        self.client = OpenAIClient(
            api_key=api_key,
            base_url="https://api.cerebras.ai/v1",
        )
        self.model_name = model_name or self.NAME

    def label_batch(self, threads: List[dict]) -> tuple:
        """Returns (results_dict, error_str, is_rate_limit)."""
        prompt_text = format_batch(threads)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            raw = completion.choices[0].message.content
            results = parse_batch_response(raw, threads)
            return results, None, False
        except Exception as e:
            return None, str(e), is_rate_limit_error(e)


# =============================================================================
# AGREEMENT COMPUTATION
# =============================================================================

def compute_pairwise_agreement(output1: dict, output2: dict) -> dict:
    """Compare two teacher outputs → agreement info."""
    if output1 is None or output2 is None:
        return {"agreement": "error", "detail": "One teacher failed"}

    has1 = output1.get("has_modification", False)
    has2 = output2.get("has_modification", False)

    if has1 != has2:
        return {"agreement": "none", "detail": f"has_mod disagree: {has1} vs {has2}"}

    if not has1 and not has2:
        return {"agreement": "full", "detail": "Both: no modification"}

    aspects1 = {m.get("aspect") for m in output1.get("modifications", [])}
    aspects2 = {m.get("aspect") for m in output2.get("modifications", [])}

    if aspects1 == aspects2:
        return {"agreement": "full", "detail": f"Both: {aspects1}"}
    elif aspects1 & aspects2:
        return {"agreement": "partial", "detail": f"Overlap: {aspects1 & aspects2}, diff: {aspects1 ^ aspects2}"}
    else:
        return {"agreement": "none", "detail": f"No overlap: {aspects1} vs {aspects2}"}


def compute_majority_vote(out1: dict, out2: dict, out3: dict) -> dict:
    """
    Majority vote across 3 teacher outputs.
    Returns final_label, vote_method, needs_review, pairwise agreements.
    """
    outputs = [out1, out2, out3]
    valid = [o for o in outputs if o is not None]

    if len(valid) < 2:
        return {
            "final_label": valid[0] if valid else None,
            "vote_method": "insufficient",
            "needs_review": True,
            "votes": {"has_mod_true": 0, "has_mod_false": 0},
            "agreement_1v2": "error", "agreement_1v3": "error", "agreement_2v3": "error",
        }

    has_mod_votes = sum(1 for o in valid if o.get("has_modification", False))
    has_mod_false = len(valid) - has_mod_votes

    agr_1v2 = compute_pairwise_agreement(out1, out2)["agreement"] if out1 and out2 else "error"
    agr_1v3 = compute_pairwise_agreement(out1, out3)["agreement"] if out1 and out3 else "error"
    agr_2v3 = compute_pairwise_agreement(out2, out3)["agreement"] if out2 and out3 else "error"

    if has_mod_votes >= 2:
        mod_outputs = [o for o in valid if o.get("has_modification", False)]
        final = max(mod_outputs, key=lambda o: len(o.get("modifications", [])))
    elif has_mod_false >= 2:
        final = {"modifications": [], "has_modification": False, "thread_type": "statement"}
    else:
        final = valid[0]

    if has_mod_votes == 3 or has_mod_false == 3:
        if has_mod_votes == 3:
            a1 = {m.get("aspect") for m in (out1 or {}).get("modifications", [])}
            a2 = {m.get("aspect") for m in (out2 or {}).get("modifications", [])}
            a3 = {m.get("aspect") for m in (out3 or {}).get("modifications", [])}
            if a1 == a2 == a3:
                vote_method = "unanimous"
            else:
                vote_method = "majority"
        else:
            vote_method = "unanimous"
        needs_review = False
    elif has_mod_votes == 2 or has_mod_false == 2:
        vote_method = "majority"
        needs_review = False
    else:
        vote_method = "no_majority"
        needs_review = True

    return {
        "final_label": final,
        "vote_method": vote_method,
        "needs_review": needs_review,
        "votes": {"has_mod_true": has_mod_votes, "has_mod_false": has_mod_false},
        "agreement_1v2": agr_1v2, "agreement_1v3": agr_1v3, "agreement_2v3": agr_2v3,
    }


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


def load_all_records(path: Path) -> List[dict]:
    records = []
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def write_all_records(records: List[dict], path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def backup_file(path: Path, suffix: str = ".bak") -> Path:
    backup_path = path.with_suffix(path.suffix + suffix)
    shutil.copy2(path, backup_path)
    return backup_path


# =============================================================================
# GENERIC PASS RUNNER (used by pass 2 and 3)
# =============================================================================

def run_repass(todo: List[dict], teacher, output_field: str,
               model_field: str, pass_name: str,
               record_index: dict, batch_size: int = BATCH_SIZE):
    """
    Generic batch re-labeling pass for pass 2 and 3.
    Labels threads and stores results in the specified output_field.
    """
    if not todo:
        print(f"✅ No records need {pass_name}!")
        return {}

    total = len(todo)
    num_batches = (total + batch_size - 1) // batch_size
    print(f"\n[{pass_name}] Processing {total} records in ~{num_batches} batches of {batch_size}")
    print(f"  Teacher: {teacher.model_name}")
    print(f"  Storing in: {output_field}\n")

    stats = {
        "total": 0, "success": 0, "batch_errors": 0, "parse_failures": 0,
        "with_mods": 0, "no_mods": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    consecutive_errors = 0
    idx = 0
    batch_num = 0

    while idx < total:
        batch_num += 1
        batch = todo[idx:idx + batch_size]

        results, error, is_rl = teacher.label_batch(batch)

        if error or results is None:
            stats["batch_errors"] += 1
            consecutive_errors += 1
            print(f"  [Batch {batch_num}] ERROR: {(error or 'No results')[:100]}")
            if consecutive_errors >= 5:
                print(f"\n  ❌ 5 consecutive batch errors — stopping {pass_name}.")
                break
            idx += batch_size
            time.sleep(2)
            continue
        else:
            consecutive_errors = 0

        batch_ok = 0
        batch_fail = 0

        for rec in batch:
            tid = rec["thread_id"]
            output = results.get(tid)
            stats["total"] += 1

            if output is None:
                stats["parse_failures"] += 1
                batch_fail += 1
                continue

            if tid in record_index:
                record_index[tid][output_field] = output
                record_index[tid][model_field] = teacher.model_name

            if output.get("has_modification"):
                stats["with_mods"] += 1
            else:
                stats["no_mods"] += 1

            stats["success"] += 1
            batch_ok += 1

        done = min(idx + batch_size, total)
        pct = stats["with_mods"] / stats["total"] * 100 if stats["total"] else 0
        print(f"  [Batch {batch_num}: {done}/{total}] "
              f"+{batch_ok} ok, {batch_fail} fail | "
              f"mods={stats['with_mods']} ({pct:.0f}%) "
              f"errs={stats['batch_errors']}")

        idx += batch_size
        time.sleep(teacher.MIN_DELAY)

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    return stats


# =============================================================================
# PASS 1: INITIAL LABELING
# =============================================================================

def run_pass1(input_path: str, output_path: str,
              gemini_key: str = None, groq_key: str = None,
              gemini_model: str = None, groq_model: str = None,
              limit: int = None, skip_existing: bool = True,
              batch_size: int = BATCH_SIZE):
    """Pass 1: Label threads with Gemini (primary) + Groq (fallback)."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gemini = None
    groq = None

    if gemini_key and GEMINI_AVAILABLE:
        try:
            gemini = GeminiTeacher(api_key=gemini_key, model_name=gemini_model, temperature=0.1)
            print(f"✓ Gemini ready: {gemini.model_name} (temp={gemini.temperature})")
        except Exception as e:
            print(f"⚠ Gemini init failed: {e}")

    if groq_key and GROQ_AVAILABLE:
        try:
            groq = GroqTeacher(api_key=groq_key, model_name=groq_model)
            print(f"✓ Groq ready:   {groq.model_name}")
        except Exception as e:
            print(f"⚠ Groq init failed: {e}")

    if not gemini and not groq:
        print("❌ No teacher available! Set GOOGLE_API_KEY and/or GROQ_API_KEY in .env")
        return

    existing_ids = set()
    if skip_existing:
        existing_ids = load_existing_ids(output_path)
        if existing_ids:
            print(f"Resuming: {len(existing_ids)} already labeled, will skip")

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
        print("✅ No new threads to process. Pass 1 complete!")
        return

    total_threads = len(threads)
    num_batches = (total_threads + batch_size - 1) // batch_size
    print(f"\n[PASS 1] Processing {total_threads} threads in ~{num_batches} batches of {batch_size}")
    print(f"Output: {output_path}\n")

    using_gemini = gemini is not None
    gemini_switched = False
    consecutive_errors = 0

    stats = {
        "total": 0, "with_mods": 0, "no_mods": 0,
        "gemini_labeled": 0, "groq_labeled": 0,
        "batch_errors": 0, "parse_failures": 0,
        "aspects": {"SUBSTITUTION": 0, "QUANTITY": 0, "TECHNIQUE": 0, "ADDITION": 0},
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    thread_idx = 0
    batch_num = 0

    while thread_idx < total_threads:
        batch_num += 1
        batch = threads[thread_idx:thread_idx + batch_size]

        if using_gemini:
            teacher = gemini
            teacher_name = gemini.model_name
            delay = GeminiTeacher.MIN_DELAY
        else:
            teacher = groq
            teacher_name = groq.model_name
            delay = GroqTeacher.MIN_DELAY

        results, error, is_rl = teacher.label_batch(batch)

        if error and is_rl and using_gemini and groq:
            print(f"\n  ⚡ Gemini rate limit at batch {batch_num}. Switching to Groq...")
            print(f"     Gemini labeled: {stats['gemini_labeled']} threads this run")
            using_gemini = False
            gemini_switched = True
            teacher = groq
            teacher_name = groq.model_name
            delay = GroqTeacher.MIN_DELAY
            results, error, is_rl = teacher.label_batch(batch)

        if error or results is None:
            stats["batch_errors"] += 1
            consecutive_errors += 1
            print(f"  [Batch {batch_num}] ERROR: {(error or 'No results')[:100]}")
            if consecutive_errors >= 5:
                print(f"\n  ❌ 5 consecutive batch errors — stopping.")
                break
            thread_idx += batch_size
            time.sleep(2)
            continue
        else:
            consecutive_errors = 0

        batch_success = 0
        batch_fail = 0

        for thread in batch:
            tid = thread["thread_id"]
            teacher_output = results.get(tid)
            stats["total"] += 1

            if teacher_output is None:
                stats["parse_failures"] += 1
                batch_fail += 1
                continue

            if teacher_output["has_modification"]:
                stats["with_mods"] += 1
                for mod in teacher_output["modifications"]:
                    a = mod.get("aspect", "")
                    if a in stats["aspects"]:
                        stats["aspects"][a] += 1
            else:
                stats["no_mods"] += 1

            if using_gemini:
                stats["gemini_labeled"] += 1
            else:
                stats["groq_labeled"] += 1

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
                "teacher_model": teacher_name,
                "labeled_at": datetime.now(timezone.utc).isoformat(),
                "second_teacher_output": None,
                "second_teacher_model": None,
                "third_teacher_output": None,
                "third_teacher_model": None,
                "agreement_1v2": None,
                "agreement_1v3": None,
                "agreement_2v3": None,
                "final_label": None,
                "vote_method": None,
                "needs_review": None,
            }

            append_jsonl(record, output_path)
            batch_success += 1

        done = min(thread_idx + batch_size, total_threads)
        pct = stats["with_mods"] / stats["total"] * 100 if stats["total"] else 0
        tag = "Gemini" if using_gemini else "Groq"
        print(f"  [Batch {batch_num}: {done}/{total_threads}] ({tag}) "
              f"+{batch_success} ok, {batch_fail} fail | "
              f"mods={stats['with_mods']} ({pct:.0f}%) errs={stats['batch_errors']}")

        thread_idx += batch_size
        time.sleep(delay)

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    stats["total_in_output"] = len(existing_ids) + stats["total"] - stats["parse_failures"]
    stats_path = Path(output_path).parent / "pass1_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  ✅ Pass 1 Done!")
    print(f"{'='*60}")
    print(f"  Processed:         {stats['total']} threads")
    print(f"    Gemini labeled:  {stats['gemini_labeled']}")
    print(f"    Groq labeled:    {stats['groq_labeled']}")
    print(f"    With mods:       {stats['with_mods']}")
    print(f"    No mods:         {stats['no_mods']}")
    print(f"    Parse failures:  {stats['parse_failures']}")
    print(f"  Aspects: {stats['aspects']}")
    print(f"  Total in output:   ~{stats['total_in_output']}")
    print(f"  Output: {output_path}")
    if gemini_switched:
        print(f"  ⚡ Switched from Gemini to Groq mid-run (rate limit)")
    print(f"\n  Next step: run --second-pass")
    print(f"{'='*60}")


# =============================================================================
# PASS 2: INTRA-ANNOTATOR (same Gemini, temp=0.3)
# =============================================================================

def run_pass2(output_path: str,
              gemini_key: str = None, groq_key: str = None,
              gemini_model: str = None, groq_model: str = None,
              limit: int = None, batch_size: int = BATCH_SIZE):
    """Pass 2: Re-label with same Gemini at temperature=0.3."""

    output_path = Path(output_path)
    if not output_path.exists():
        print(f"❌ {output_path} not found. Run pass 1 first.")
        return

    if not gemini_key:
        print("❌ GOOGLE_API_KEY required for pass 2")
        return

    gemini = GeminiTeacher(api_key=gemini_key, model_name=gemini_model, temperature=0.3)
    print(f"✓ Gemini ready: {gemini.model_name} (temp={gemini.temperature})")

    records = load_all_records(output_path)
    print(f"Loaded {len(records)} records from {output_path}")

    todo = [r for r in records if r.get("second_teacher_output") is None]
    if limit:
        todo = todo[:limit]

    if not todo:
        print("✅ All records already have pass 2 labels!")
        return

    record_index = {r["thread_id"]: r for r in records}

    stats = run_repass(todo, gemini, "second_teacher_output", "second_teacher_model",
                       "PASS 2 — Intra-annotator", record_index, batch_size)

    # Compute pairwise agreement
    agreement_counts = {"full": 0, "partial": 0, "none": 0, "error": 0}
    for rec in records:
        if rec.get("second_teacher_output") is not None:
            agr = compute_pairwise_agreement(rec.get("teacher_output"), rec.get("second_teacher_output"))
            rec["agreement_1v2"] = agr["agreement"]
            if agr["agreement"] in agreement_counts:
                agreement_counts[agr["agreement"]] += 1

    backup_file(output_path, ".pre_pass2.bak")
    write_all_records(records, output_path)

    stats_path = output_path.parent / "pass2_stats.json"
    stats["agreement"] = agreement_counts
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  ✅ Pass 2 Done!")
    print(f"{'='*60}")
    print(f"  Re-labeled:        {stats.get('success', 0)} threads")
    print(f"    With mods:       {stats.get('with_mods', 0)}")
    print(f"    No mods:         {stats.get('no_mods', 0)}")
    print(f"    Parse failures:  {stats.get('parse_failures', 0)}")
    print(f"  Agreement (pass1 vs pass2):")
    for level, count in agreement_counts.items():
        pct = count / max(1, sum(agreement_counts.values())) * 100
        print(f"    {level:10s}  {count:5d}  ({pct:.1f}%)")
    print(f"\n  Next step: run --third-pass")
    print(f"{'='*60}")


# =============================================================================
# PASS 3: INTER-ANNOTATOR (Cerebras Qwen 235B)
# =============================================================================

def run_pass3(output_path: str,
              cerebras_key: str = None, cerebras_model: str = None,
              limit: int = None, batch_size: int = BATCH_SIZE):
    """Pass 3: Re-label with Cerebras Qwen 235B (different model family)."""

    output_path = Path(output_path)
    if not output_path.exists():
        print(f"❌ {output_path} not found. Run pass 1 first.")
        return

    if not cerebras_key:
        print("❌ CEREBRAS_API_KEY required for pass 3")
        return

    cerebras = CerebrasTeacher(api_key=cerebras_key, model_name=cerebras_model)
    print(f"✓ Cerebras ready: {cerebras.model_name}")

    records = load_all_records(output_path)
    print(f"Loaded {len(records)} records from {output_path}")

    todo = [r for r in records if r.get("third_teacher_output") is None]
    if limit:
        todo = todo[:limit]

    if not todo:
        print("✅ All records already have pass 3 labels!")
        return

    record_index = {r["thread_id"]: r for r in records}

    stats = run_repass(todo, cerebras, "third_teacher_output", "third_teacher_model",
                       "PASS 3 — Inter-annotator", record_index, batch_size)

    # Compute pairwise agreements
    agr_1v3 = {"full": 0, "partial": 0, "none": 0, "error": 0}
    agr_2v3 = {"full": 0, "partial": 0, "none": 0, "error": 0}
    for rec in records:
        if rec.get("third_teacher_output") is not None:
            a13 = compute_pairwise_agreement(rec.get("teacher_output"), rec.get("third_teacher_output"))
            rec["agreement_1v3"] = a13["agreement"]
            if a13["agreement"] in agr_1v3:
                agr_1v3[a13["agreement"]] += 1

            if rec.get("second_teacher_output") is not None:
                a23 = compute_pairwise_agreement(rec.get("second_teacher_output"), rec.get("third_teacher_output"))
                rec["agreement_2v3"] = a23["agreement"]
                if a23["agreement"] in agr_2v3:
                    agr_2v3[a23["agreement"]] += 1

    backup_file(output_path, ".pre_pass3.bak")
    write_all_records(records, output_path)

    stats_path = output_path.parent / "pass3_stats.json"
    stats["agreement_1v3"] = agr_1v3
    stats["agreement_2v3"] = agr_2v3
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  ✅ Pass 3 Done!")
    print(f"{'='*60}")
    print(f"  Labeled:           {stats.get('success', 0)} threads")
    print(f"    With mods:       {stats.get('with_mods', 0)}")
    print(f"    No mods:         {stats.get('no_mods', 0)}")
    print(f"    Parse failures:  {stats.get('parse_failures', 0)}")
    print(f"  Agreement (pass1 vs pass3 — inter-annotator):")
    for level, count in agr_1v3.items():
        pct = count / max(1, sum(agr_1v3.values())) * 100
        print(f"    {level:10s}  {count:5d}  ({pct:.1f}%)")
    print(f"  Agreement (pass2 vs pass3):")
    for level, count in agr_2v3.items():
        pct = count / max(1, sum(agr_2v3.values())) * 100
        print(f"    {level:10s}  {count:5d}  ({pct:.1f}%)")
    print(f"\n  Next step: run --finalize to compute majority vote")
    print(f"{'='*60}")


# =============================================================================
# FINALIZE: MAJORITY VOTE
# =============================================================================

def run_finalize(output_path: str):
    """Compute majority vote across all 3 passes and set final labels."""

    output_path = Path(output_path)
    records = load_all_records(output_path)
    print(f"Loaded {len(records)} records")

    vote_stats = {"unanimous": 0, "majority": 0, "no_majority": 0, "insufficient": 0}
    review_count = 0
    final_mods = 0
    final_no_mods = 0

    for rec in records:
        out1 = rec.get("teacher_output")
        out2 = rec.get("second_teacher_output")
        out3 = rec.get("third_teacher_output")

        vote = compute_majority_vote(out1, out2, out3)

        rec["final_label"] = vote["final_label"]
        rec["vote_method"] = vote["vote_method"]
        rec["needs_review"] = vote["needs_review"]
        rec["agreement_1v2"] = vote["agreement_1v2"]
        rec["agreement_1v3"] = vote["agreement_1v3"]
        rec["agreement_2v3"] = vote["agreement_2v3"]

        if vote["vote_method"] in vote_stats:
            vote_stats[vote["vote_method"]] += 1
        if vote["needs_review"]:
            review_count += 1
        if vote.get("final_label", {}).get("has_modification"):
            final_mods += 1
        else:
            final_no_mods += 1

    backup_file(output_path, ".pre_finalize.bak")
    write_all_records(records, output_path)

    stats = {"total": len(records), "vote_stats": vote_stats,
             "needs_review": review_count, "final_mods": final_mods, "final_no_mods": final_no_mods}
    stats_path = output_path.parent / "final_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    total = len(records)
    print(f"\n{'='*60}")
    print(f"  ✅ Finalized — Majority Vote Complete!")
    print(f"{'='*60}")
    print(f"  Total records:     {total}")
    print(f"  Vote results:")
    for method, count in vote_stats.items():
        pct = count / max(1, total) * 100
        print(f"    {method:15s}  {count:5d}  ({pct:.1f}%)")
    print(f"  Final labels:")
    print(f"    With modification: {final_mods} ({final_mods/max(1,total)*100:.1f}%)")
    print(f"    No modification:   {final_no_mods} ({final_no_mods/max(1,total)*100:.1f}%)")
    print(f"  Needs manual review: {review_count}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")


# =============================================================================
# EXPORT REVIEW + AGREEMENT STATS
# =============================================================================

def export_review(output_path: str, review_path: str = REVIEW_OUTPUT):
    """Export records that need manual review."""
    records = load_all_records(Path(output_path))
    review_records = [r for r in records if r.get("needs_review")]

    if not review_records:
        print("✅ No records need review! All annotators agree.")
        return

    review_path = Path(review_path)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    write_all_records(review_records, review_path)

    print(f"\n{'='*60}")
    print(f"  REVIEW EXPORT")
    print(f"{'='*60}")
    print(f"  Total needing review: {len(review_records)}")
    print(f"  Exported to: {review_path}")
    print(f"{'='*60}")


def show_agreement_stats(output_path: str):
    """Show comprehensive agreement statistics."""
    records = load_all_records(Path(output_path))
    total = len(records)

    has_p2 = sum(1 for r in records if r.get("second_teacher_output") is not None)
    has_p3 = sum(1 for r in records if r.get("third_teacher_output") is not None)
    has_final = sum(1 for r in records if r.get("final_label") is not None)

    print(f"\n{'='*60}")
    print(f"  AGREEMENT STATISTICS")
    print(f"{'='*60}")
    print(f"  Total records:     {total}")
    print(f"  Pass 2 done:       {has_p2}")
    print(f"  Pass 3 done:       {has_p3}")
    print(f"  Finalized:         {has_final}")

    for pair, field_name in [("Pass1 vs Pass2", "agreement_1v2"),
                             ("Pass1 vs Pass3", "agreement_1v3"),
                             ("Pass2 vs Pass3", "agreement_2v3")]:
        counts = {"full": 0, "partial": 0, "none": 0, "error": 0}
        counted = 0
        for r in records:
            val = r.get(field_name)
            if val and val in counts:
                counts[val] += 1
                counted += 1
        if counted > 0:
            print(f"\n  {pair} ({counted} records):")
            for level, count in counts.items():
                pct = count / max(1, counted) * 100
                print(f"    {level:10s}  {count:5d}  ({pct:.1f}%)")

    if has_final > 0:
        vote_counts = {}
        review_count = 0
        for r in records:
            vm = r.get("vote_method")
            if vm:
                vote_counts[vm] = vote_counts.get(vm, 0) + 1
            if r.get("needs_review"):
                review_count += 1
        print(f"\n  Majority vote ({has_final} records):")
        for method, count in sorted(vote_counts.items()):
            pct = count / max(1, has_final) * 100
            print(f"    {method:15s}  {count:5d}  ({pct:.1f}%)")
        print(f"  Needs manual review: {review_count}")

    model_counts = {}
    for r in records:
        m = r.get("teacher_model", "unknown")
        model_counts[m] = model_counts.get(m, 0) + 1
    print(f"\n  Pass 1 model distribution:")
    for model, count in sorted(model_counts.items()):
        print(f"    {model}: {count}")

    print(f"{'='*60}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Three-Pass Silver Label Generator with Majority Vote",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Pass 1 — Gemini labels all threads:
     python -m src.teacher_labeling.generate_labels \\
         -i data/raw_youtube/threads.jsonl --limit 5000 --batch-size 20

  2. Pass 2 — Same Gemini, temp=0.3 (intra-annotator):
     python -m src.teacher_labeling.generate_labels --second-pass --limit 5000 --batch-size 20

  3. Pass 3 — Cerebras Qwen 235B (inter-annotator):
     python -m src.teacher_labeling.generate_labels --third-pass --limit 5000 --batch-size 20

  4. Finalize — Majority vote, flag disagreements:
     python -m src.teacher_labeling.generate_labels --finalize

  5. Export disagreements for manual review:
     python -m src.teacher_labeling.generate_labels --export-review

  6. View agreement stats:
     python -m src.teacher_labeling.generate_labels --agreement-stats
        """,
    )

    parser.add_argument("--second-pass", action="store_true",
                        help="Pass 2: same Gemini, temp=0.3 (intra-annotator)")
    parser.add_argument("--third-pass", action="store_true",
                        help="Pass 3: Cerebras Qwen 235B (inter-annotator)")
    parser.add_argument("--finalize", action="store_true",
                        help="Compute majority vote and set final labels")
    parser.add_argument("--export-review", action="store_true",
                        help="Export disagreements to needs_review.jsonl")
    parser.add_argument("--agreement-stats", action="store_true",
                        help="Show agreement statistics")

    parser.add_argument("--input", "-i", help="Input threads JSONL (pass 1)")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--gemini-key", default=os.environ.get("GOOGLE_API_KEY"),
                        help="Gemini API key (default: GOOGLE_API_KEY env var)")
    parser.add_argument("--groq-key", default=os.environ.get("GROQ_API_KEY"),
                        help="Groq API key (default: GROQ_API_KEY env var)")
    parser.add_argument("--cerebras-key", default=os.environ.get("CEREBRAS_API_KEY"),
                        help="Cerebras API key (default: CEREBRAS_API_KEY env var)")
    parser.add_argument("--gemini-model", default=None,
                        help=f"Gemini model (default: {GeminiTeacher.NAME})")
    parser.add_argument("--groq-model", default=None,
                        help=f"Groq model (default: {GroqTeacher.NAME})")
    parser.add_argument("--cerebras-model", default=None,
                        help=f"Cerebras model (default: {CerebrasTeacher.NAME})")
    parser.add_argument("--limit", type=int, help="Max threads to process per run")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Threads per API call (default: {BATCH_SIZE})")
    parser.add_argument("--no-skip", action="store_true",
                        help="Pass 1: don't skip already-labeled threads")

    args = parser.parse_args()

    if args.export_review:
        export_review(args.output)
        return

    if args.agreement_stats:
        show_agreement_stats(args.output)
        return

    if args.finalize:
        run_finalize(args.output)
        return

    if args.third_pass:
        run_pass3(
            output_path=args.output,
            cerebras_key=args.cerebras_key,
            cerebras_model=args.cerebras_model,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        return

    if args.second_pass:
        run_pass2(
            output_path=args.output,
            gemini_key=args.gemini_key,
            groq_key=args.groq_key,
            gemini_model=args.gemini_model,
            groq_model=args.groq_model,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        return

    if not args.input:
        parser.error("--input / -i is required for pass 1 labeling")

    if not args.gemini_key and not args.groq_key:
        parser.error(
            "No API keys found!\n"
            "Set GOOGLE_API_KEY and/or GROQ_API_KEY in .env,\n"
            "or pass --gemini-key / --groq-key."
        )

    run_pass1(
        input_path=args.input,
        output_path=args.output,
        gemini_key=args.gemini_key,
        groq_key=args.groq_key,
        gemini_model=args.gemini_model,
        groq_model=args.groq_model,
        limit=args.limit,
        skip_existing=not args.no_skip,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()