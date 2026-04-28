#!/usr/bin/env python3
"""
Teacher Silver Label Generator — v10 (Tiebreaker Pass 3 + Span Validation)
==========================================================================
Pass 1: Label all threads with Gemini (primary). Batched. temp=0.0.
Pass 2: Re-label with same Gemini at temp=0.3 (intra-annotator agreement).
Pass 3: TIEBREAKER ONLY — Cerebras Qwen 235B labels records where
        Pass 1 and Pass 2 disagree (full agreement → skipped, marked
        as "gemini_unanimous"). Cuts Qwen quota usage by ~30-70%.
Final: Majority vote (2/3) OR gemini_unanimous (Pass 3 skipped).
       Final spans are validated against the actual thread text;
       hallucinated spans are dropped before training data is built.

CHANGELOG vs v9.1:
  - Pass 3 runs in TIEBREAKER MODE by default (selective routing).
    Records where Pass 1 ≡ Pass 2 (full agreement on aspects) are
    flagged pass3_skipped=True and never sent to Qwen, saving quota.
    Use --full-coverage to opt back into labeling every record.
  - New vote_method "gemini_unanimous" for skipped-Pass-3 records.
  - Span validation hook in run_finalize (drops hallucinations,
    flips threads to no_mod when every span fails substring check).
  - thread_type voting in compute_majority_vote (no more hardcoded
    "statement" in the no-mod branch).
  - SYSTEM_PROMPT patches for the three Qwen-specific bug classes:
    * RULE 2 bullet for "describing what the creator did"
    * RULE 4 TECHNIQUE definition includes omissions ("בלי X")
    * Aspect decision order has a dedicated omission step
    * Three new few-shot examples (I, J, K) covering Bugs 3, 4, 5

Usage:
    # Pass 1 — label everything (Gemini):
    python -m src.teacher_labeling.generate_labels \
        -i data/raw_youtube/threads.jsonl --limit 5000 --batch-size 20

    # Pass 2 — same Gemini, different temperature (intra-annotator):
    python -m src.teacher_labeling.generate_labels --second-pass --limit 5000 --batch-size 20

    # Pass 3 — TIEBREAKER MODE (default): only records where pass1 ≢ pass2:
    python -m src.teacher_labeling.generate_labels --third-pass --batch-size 10

    # Pass 3 — full coverage (legacy behavior, ALL records to Qwen):
    python -m src.teacher_labeling.generate_labels --third-pass --full-coverage --batch-size 10

    # Compute majority vote + validate spans + flag disagreements:
    python -m src.teacher_labeling.generate_labels --finalize

    # Export only needs-review records:
    python -m src.teacher_labeling.generate_labels --export-review

    # View agreement stats:
    python -m src.teacher_labeling.generate_labels --agreement-stats

API keys read from .env automatically:
    GOOGLE_API_KEY=your_gemini_key
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
from collections import Counter as _Counter

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
# SYSTEM PROMPT — v2 (master plan §4.3 compliant, 5-part structure)
#                   + Qwen-bug patches (RULE 2 bullet, RULE 4 omission,
#                     decision-order omission step, examples I/J/K)
# =============================================================================

SYSTEM_PROMPT = """You are an expert culinary NLP assistant specializing in Hebrew text.
Your task: analyze comment threads from cooking videos and extract recipe modification suggestions.

⚠️ GUIDING PRINCIPLE — READ FIRST
Precision matters more than recall. When in doubt, return has_modification: false
rather than guessing. A missed modification is acceptable; a fabricated modification
poisons the training data. If you cannot confidently identify the exact span,
the exact aspect, AND verify it is not praise/question/complaint — emit nothing.

You will receive MULTIPLE threads in one request, each marked with === THREAD N (ID: xxx) ===.
You must return a JSON array with one result object per thread, in the same order.

For each thread:
- [TOP COMMENT]: The main comment
- [REPLY N, user/creator]: Replies to the comment

════════════════════════════════════════════════════════
RULES — read all of them before labeling anything
════════════════════════════════════════════════════════

RULE 1 — QUESTION vs STATEMENT
  If the top comment is a QUESTION (contains ? or words like אפשר/האם/כדאי/אפשרי/מה אם),
  extract modifications from the REPLIES only.
  The question itself is NOT a modification.
  A question with no meaningful replies → has_modification: false.

RULE 2 — THE USER MUST HAVE ACTUALLY COOKED IT DIFFERENTLY
  Only label a span if the commenter is describing something they ACTUALLY DID
  (or a clear, actionable suggestion they are making about the recipe).
  The following are NOT modifications — set has_modification: false:
    • Pointing out an error or discrepancy in the video ("you said 300g but wrote 500g")
    • Complaining about a result ("it gives a bad aftertaste")
    • Expressing an opinion or praise ("it was amazing", "it was terrible")
    • Asking a question even if it contains an ingredient name
    • Simply mentioning an ingredient exists without saying what to do with it
    • Describing what the CREATOR/host did in the video itself ("she added X",
      "הוא שם Y", "היא הוסיפה Z") — this is the viewer NOTING the recipe,
      not modifying it. Only count it if the viewer states THEY changed something
      themselves, or proposes adding/changing something.

RULE 3 — SPAN MUST BE EXACT HEBREW TEXT
  "span" must be copied VERBATIM from the source_comment field you specify.
  If source_comment is "reply_1", the span MUST be a literal substring of reply_1's text.
  Do NOT paraphrase, translate, or summarize.
  Do NOT include leading filler, hedge words, or question particles in the span.

  Strip these from the START of spans:
    • Hedges:    עדיף, אפשר, אולי, כדאי, ניתן, אני ממליץ, אפשרי, רצוי, מומלץ
    • Questions: האם, אפשר, כדאי
    • Verbs (when not essential): הוספתי, שמתי, עשיתי, השתמשתי

  Span should fully include the modification AND its essential comparison context.
  Typical length 2-6 tokens. Longer is OK when context demands it.
  Critical: if the comment uses "X במקום Y" (X instead of Y), the span MUST
  include BOTH X and Y. If it uses "יותר X" or "פחות X", include both words.
  Do NOT chop the comparison and leave only the new value.

  Example 1: "עדיף להוסיף כמה כפות מהרוטב"
             CORRECT span: "להוסיף כמה כפות מהרוטב"
             WRONG span:   "עדיף להוסיף כמה כפות מהרוטב"  (hedge included)

  Example 2: "300 גרם קמח במקום 700 גרם"
             CORRECT span: "300 גרם קמח במקום 700 גרם"
             WRONG span:   "300 גרם קמח"  (comparison context lost)

RULE 4 — ASPECT DEFINITIONS (pick exactly one)
  SUBSTITUTION : replacing one ingredient/tool with another
                 Example: "במקום סוכר שמתי סטיביה" → span: "סטיביה במקום סוכר"
  QUANTITY     : a different amount of something already in the recipe,
                 OR a specific cooking time/temperature modification,
                 OR "more/less of X" where X is already in the recipe
                 Example: "כפול כמות השוקולד" → span: "כפול כמות השוקולד"
                 Example: "30 דקות במקום 20" → span: "30 דקות במקום 20"
                 Example: "יותר סילאן" → span: "יותר סילאן"  (QUANTITY, not ADDITION)
  TECHNIQUE    : a different method, tool, order of steps, preparation approach,
                 OR an OMISSION — the user removed an ingredient that was in
                 the original recipe (phrases like "בלי X", "ללא X", "לא שמתי X"
                 / "without X"). Omissions are TECHNIQUE, never SUBSTITUTION,
                 because there is no replacement ingredient.
                 Example: "במיקסר עם וו גיטרה" → span: "במיקסר עם וו גיטרה"
                 Example: "להשרות לילה שלם" → span: "להשרות לילה שלם"
                 Example: "בלי אבקת אפיה" → span: "בלי אבקת אפיה"  (omission)
                 Example: "ללא סוכר" → span: "ללא סוכר"  (omission)
  ADDITION     : adding a NEW ingredient that is NOT in the original recipe.
                 REQUIRES an explicit add-verb in Hebrew:
                   הוספתי, שמתי גם, נתתי, פיזרתי, בנוסף, יחד עם, גם, ועוד
                 If NO add-verb appears, do NOT label ADDITION even if a
                 new ingredient is mentioned — likely SUBSTITUTION/TECHNIQUE.
                 Example POSITIVE: "הוספתי כוסברה טרייה בסוף" → span: "כוסברה טרייה"
                 Example NEGATIVE: "במקום סוכר שמתי דבש" → SUBSTITUTION not ADDITION
                                    ("במקום" present, no add-verb)

  ⚠️  ASPECT DECISION ORDER (apply in this exact sequence):
      1. If "במקום" (instead of) is present:
         - Comparing amounts/times/temperatures → QUANTITY
         - Comparing ingredients/tools         → SUBSTITUTION
         (NEVER ADDITION when "במקום" appears)
      2. If a comparative quantity word is present (יותר/פחות/כפול/חצי) → QUANTITY
      3. If an explicit add-verb is present (הוספתי/שמתי גם/בנוסף/...) → ADDITION
      4. If the comment OMITS an existing ingredient ("בלי X" / "ללא X" /
         "לא שמתי X" / "without X") → TECHNIQUE.
         Never SUBSTITUTION — omission has no replacement ingredient.
      5. If a method/tool/order/time/temperature change is described → TECHNIQUE
      6. If NONE of the above clearly applies → set has_modification: false.
         DO NOT default to ADDITION. DO NOT guess. Precision over recall.

RULE 5 — source_comment
  "top"     → the modification span is in the top comment
  "reply_1" → first reply, "reply_2" → second reply, etc.
  Whichever value you pick, the span MUST appear verbatim in THAT comment.

RULE 6 — confidence
  0.85–1.00 : Clear, unambiguous modification the commenter clearly did/recommends
  0.70–0.84 : Somewhat clear but phrased tentatively or partially ambiguous
  Below 0.70: Uncertain — set has_modification: false. Do NOT include low-confidence
              modifications. Precision matters more than recall.
  Creator replies always get 0.90+ if a modification is present.

════════════════════════════════════════════════════════
FEW-SHOT EXAMPLES
════════════════════════════════════════════════════════

──── EXAMPLE A — Complaint that looks like a modification (→ false) ────
Comment: "100 אחוז נותן טעם לוואי😢"
English: "100% gives off-taste 😢"
Why false: The commenter is complaining about the result, not describing a change they made.
Output: {"has_modification": false, "modifications": []}

──── EXAMPLE B — Error-pointing, not a modification (→ false) ────
Comment: "אתה אומר 300גרם והמתכון רשמתה 500גרם ויש הבדל‼️"
English: "You say 300g but the recipe says 500g and there's a difference!!"
Why false: The commenter is pointing out an inconsistency in the video, not describing
           something they did differently. No modification was made.
Output: {"has_modification": false, "modifications": []}

──── EXAMPLE C — Observation that sounds like an ingredient (→ false) ────
Comment: "חסר שמנת חמוצה בצד"
English: "Missing sour cream on the side"
Why false: The commenter is noting something they feel is missing from the dish as presented,
           not describing a change they made to the recipe.
Output: {"has_modification": false, "modifications": []}

──── EXAMPLE D — Hedge word at span boundary (→ strip it) ────
Comment: "עדיף להוסיף כמה כפות מהרוטב לתוך כוס גדולה עם רסק העגבניות עד להמסה ולהוסיף לקציצות"
English: "Better to add a few spoons of sauce into a large cup with tomato paste until dissolved
          then add to the meatballs"
Correct output:
{
  "has_modification": true,
  "thread_type": "statement",
  "modifications": [
    {
      "span": "להוסיף כמה כפות מהרוטב",
      "aspect": "TECHNIQUE",
      "source_comment": "top",
      "confidence": 0.85
    }
  ]
}
Note: "עדיף" stripped from start (hedge). Span kept short (2-4 tokens). The rest of the
      sentence is procedural detail, not a separate modification.

──── EXAMPLE E — QUANTITY vs ADDITION (amount of existing ingredient) ────
Comment: "רק לשים הרבה נוזלים שלא יהיה כבד בגלל הקטניות ואני אוהב יותר סילאן"
English: "Just add a lot of liquid so it's not heavy from the legumes, and I like more silan"
Correct output:
{
  "has_modification": true,
  "thread_type": "statement",
  "modifications": [
    {
      "span": "הרבה נוזלים",
      "aspect": "QUANTITY",
      "source_comment": "top",
      "confidence": 0.82
    },
    {
      "span": "יותר סילאן",
      "aspect": "QUANTITY",
      "source_comment": "top",
      "confidence": 0.85
    }
  ]
}
Note: Both are QUANTITY. "יותר סילאן" is NOT ADDITION — silan is already in the recipe,
      the commenter uses more of it. "לשים" stripped (verb, not essential).

──── EXAMPLE F — Question thread, modification confirmed in reply ────
[TOP COMMENT]: "אפשרי להחליף במירין?"
[REPLY 1, user]: "כן, פשוט החלף במירין באותה כמות"
Correct output:
{
  "has_modification": true,
  "thread_type": "question",
  "modifications": [
    {
      "span": "החלף במירין",
      "aspect": "SUBSTITUTION",
      "source_comment": "reply_1",
      "confidence": 0.87
    }
  ]
}
Note: The question itself is NOT a modification. The reply confirms the substitution
      and the span is a verbatim substring of reply_1's text. The span MUST come from
      whichever comment you set as source_comment.

──── EXAMPLE G — Clear technique modification in top comment ────
Comment: "תשים שמן בדף אפייה ככה לא ידבק לך"
English: "Put oil on the parchment paper so it doesn't stick"
Correct output:
{
  "has_modification": true,
  "thread_type": "statement",
  "modifications": [
    {
      "span": "שמן בדף אפייה",
      "aspect": "TECHNIQUE",
      "source_comment": "top",
      "confidence": 0.90
    }
  ]
}
Note: "תשים" stripped (verb). The "ככה לא ידבק לך" tail is the rationale, not
      a separate modification.

──── EXAMPLE H — Substitution buried in a long comment ────
Comment: "כוסמין לבן יעבוד טוב כי הוא מתנהג זהה. המלא מטבעו קל אז לא בטוחה איך יצא לך"
English: "White spelt will work well because it behaves the same. Whole [spelt] is
          naturally light so not sure how it'll turn out for you"
Correct output:
{
  "has_modification": true,
  "thread_type": "statement",
  "modifications": [
    {
      "span": "כוסמין לבן",
      "aspect": "SUBSTITUTION",
      "source_comment": "top",
      "confidence": 0.86
    }
  ]
}
Note: Only the substituted ingredient is the span — not the entire explanation.

──── EXAMPLE I — Substitution question with no confirming reply (→ false) ────
[TOP COMMENT]: "אפשר קוטג' במקום גבינה לבנה?"
English: "Can I use cottage cheese instead of white cheese?"
[no replies, or replies are emojis / "me too" / thanks only]
Why false: The viewer is ASKING about a substitution, not REPORTING one.
           Without a reply that confirms the substitution works, there is no
           modification to extract. The ingredients named in the question
           (קוטג', גבינה לבנה) are NOT a SUBSTITUTION span.
Output: {"has_modification": false, "thread_type": "question", "modifications": []}

──── EXAMPLE J — Reply describing what the creator did (→ false) ────
[TOP COMMENT]: "צריך להוסיף חומוס מבושל למרק, למקרה ששכחת"
[REPLY 1, user]: "היא הוסיפה לסיר גרגירי חומוס 🎉"
English: TOP: "You need to add cooked chickpeas to the soup, in case you forgot"
         REPLY 1: "She added chickpeas to the pot 🎉"
Why false: Reply 1 is the viewer NOTING that the creator already added chickpeas
           in the video — descriptive, not a viewer-suggested modification.
           The top comment is a misunderstanding (the viewer thought the creator
           forgot, but the creator didn't), so it isn't a real modification
           suggestion either.
Output: {"has_modification": false, "thread_type": "statement", "modifications": []}

──── EXAMPLE K — Omission is TECHNIQUE, not SUBSTITUTION ────
Comment: "עשיתי בלי אבקת אפיה ויצא מצוין"
English: "I made it without baking powder and it came out great"
Why TECHNIQUE: The user OMITTED an ingredient that was in the original recipe.
               There is NO replacement ingredient → not SUBSTITUTION.
Correct output:
{
  "has_modification": true,
  "thread_type": "statement",
  "modifications": [
    {
      "span": "בלי אבקת אפיה",
      "aspect": "TECHNIQUE",
      "source_comment": "top",
      "confidence": 0.90
    }
  ]
}
Note: "עשיתי" stripped (verb, not essential). Span keeps the negation "בלי"
      because that IS the modification — without "בלי" the span "אבקת אפיה"
      would name an ingredient with no indication of what was changed.

════════════════════════════════════════════════════════
SELF-CHECK BEFORE OUTPUTTING
════════════════════════════════════════════════════════

For EACH modification you are about to emit, verify all of the following.
If ANY check fails, remove that modification from your output:

  ☐ (a) The span string appears VERBATIM as a substring of the comment
        named in source_comment (top / reply_N). If it doesn't, the span
        is wrong — fix or drop it.
  ☐ (b) The aspect is exactly one of SUBSTITUTION/QUANTITY/TECHNIQUE/ADDITION
        and matches the definition in Rule 4.
  ☐ (c) The span is not praise, a complaint, an error-pointing observation,
        or a bare question (Rule 2).
  ☐ (d) The span does NOT start with a hedge word, question particle, or
        unnecessary leading verb (Rule 3).
  ☐ (e) The span is short (2-4 tokens typical, never more than 8).
  ☐ (f) Confidence ≥ 0.70. If lower, drop the modification.

After all checks, if no modifications survive, set has_modification: false
and modifications: [].

════════════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════════════

Return a JSON array with exactly one object per thread, in input order.
Each object must have this exact shape:

{
  "thread_id": "<the ID from the thread header>",
  "has_modification": true/false,
  "thread_type": "statement" | "question" | "mixed",
  "modifications": [
    {
      "span": "<exact Hebrew substring of source comment>",
      "aspect": "SUBSTITUTION" | "QUANTITY" | "TECHNIQUE" | "ADDITION",
      "source_comment": "top" | "reply_1" | "reply_2" | ...,
      "confidence": 0.0–1.0
    }
  ]
}

If has_modification is false, modifications must be an empty array [].
Do not add any text, explanation, or markdown outside the JSON array.
"""


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
        # PRIMARY STRATEGY: positional alignment when lengths match.
        # Reason: Qwen-3-235B's tokenizer occasionally rewrites YouTube thread_ids
        # mid-stream (drops chars from long alphanumeric IDs). String matching on
        # `thread_id` then fails for those records. Order is preserved by the model,
        # so position-based matching is more robust than string matching.
        if len(parsed) == len(threads):
            for tid, item in zip(thread_ids, parsed):
                if isinstance(item, dict):
                    validated = validate_single_output(item)
                    if validated:
                        results[tid] = validated
        else:
            # FALLBACK: lengths differ (model dropped or duplicated items).
            # Match by thread_id string. Items whose IDs don't match input
            # are dropped (no recovery possible without lengths agreeing).
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                tid = item.get("thread_id", "")
                if tid in results:
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

def classify_api_error(error: Exception) -> str:
    """Classify an API exception into one of:
        'rate_limit'  — 429 / quota exceeded / TPM-RPM-RPD exhausted (your account)
        'overloaded'  — 503 service unavailable / model busy (their server)
        'auth'        — 401 / 403 / invalid API key (NEVER retry — fail fast)
        'transient'   — 500 / connection / timeout (retry with short backoff)
        'other'       — anything else (count toward kill switch)
    """
    msg = str(error).lower()

    # Auth errors — never retry, no point waiting
    if any(kw in msg for kw in ["401", "403", "unauthorized", "invalid api key",
                                 "permission denied", "api key not valid"]):
        return "auth"

    # Server overload — their problem, back off hard
    if any(kw in msg for kw in [
        "503", "unavailable", "experiencing high demand",
        "model is overloaded", "model_overloaded", "overloaded",
        "service unavailable", "try again later",
    ]):
        return "overloaded"

    # Rate limit — your quota
    if any(kw in msg for kw in [
        "rate limit", "rate_limit", "ratelimit",
        "quota", "resource exhausted", "resourceexhausted",
        "429", "too many requests",
        "tokens per minute", "requests per minute",
        "requests per day", "rpm", "rpd", "tpm",
    ]):
        return "rate_limit"

    # Transient network/server faults — retry with short backoff
    if any(kw in msg for kw in [
        "500", "502", "504",
        "connection", "timeout", "timed out",
        "deadline exceeded", "remote end closed",
        "ssl", "broken pipe",
    ]):
        return "transient"

    return "other"


def is_rate_limit_error(error: Exception) -> bool:
    """Backward-compat wrapper. Returns True for rate_limit OR overloaded
    (both should trigger backoff). Auth/other errors return False so the
    kill switch can fire on them.
    """
    kind = classify_api_error(error)
    return kind in ("rate_limit", "overloaded")


def detect_model_family(model_name: str) -> str:
    """Detect whether a model name belongs to a known family."""
    if not model_name:
        return ""
    name = model_name.lower()
    if "gemini" in name:
        return GEMINI_FAMILY
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

    def __init__(self, api_key: str, model_name: str = None, temperature: float = 0.0):
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
# TEACHER: CEREBRAS (Qwen 3 235B via OpenAI-compatible API)
# =============================================================================

class CerebrasTeacher:
    """Cerebras Qwen 3 235B — third annotator for majority vote."""

    NAME = "qwen-3-235b-a22b-instruct-2507"
    FAMILY = CEREBRAS_FAMILY
    # HARD LIMITS (from Cerebras dashboard, qwen-3-235b-a22b-instruct-2507 Preview):
    #   RPM=5, RPH=900, RPD=14,400, TPM=30,000, context=65,536
    # MIN_DELAY: 60s/5RPM = 12s, +1s safety margin = 13s.
    # MAX_BATCH_SIZE: with v2 prompt (~3.5K tokens) + N threads × ~250 tokens each
    #   + output ~250 tokens/thread, batch=10 stays under 30K TPM at 5 RPM.
    #   batch=20 would be ~6-9K tokens/call × 5 RPM = 30-45K TPM → OVER limit.
    MIN_DELAY = 13.0
    MAX_BATCH_SIZE = 30

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
                max_tokens=8192,  # was 4096 — bumped for v2 longer prompt + JSON safety margin
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


def should_skip_pass3(rec: dict) -> bool:
    """Tiebreaker filter: should this record skip Pass 3?

    Skip ONLY when both Pass 1 and Pass 2 succeeded AND have full agreement
    (same has_modification and same aspect set). In every other case
    (Pass 2 failed, partial/none agreement, etc.) Pass 3 must run as
    tiebreaker / inter-annotator signal.
    """
    out1 = rec.get("teacher_output")
    out2 = rec.get("second_teacher_output")
    if out1 is None or out2 is None:
        return False
    agr = compute_pairwise_agreement(out1, out2)
    return agr["agreement"] == "full"


def compute_majority_vote(out1: dict, out2: dict, out3: dict,
                          pass3_skipped: bool = False) -> dict:
    """
    Majority vote across 3 teacher outputs.
    Returns final_label, vote_method, needs_review, pairwise agreements.

    FIX 2: thread_type is voted on among the contributing outputs
    (no_mod branch no longer hardcodes "statement"). When 2/3 agree on
    no_modification, the thread_type is the most common type among
    those 2-3 no-mod outputs (ties broken by pass 1 > pass 2 > pass 3).

    NEW (v10): selective routing. If pass3_skipped=True (Pass 1 ≡ Pass 2
    full agreement caused us to skip Qwen), we treat the 2-pass agreement
    as a unanimous "gemini_unanimous" decision rather than calling it
    "majority" (which would be misleading — only 2 voters were polled).
    """
    # ─── Selective routing: Pass 3 was deliberately skipped ─────────────
    if pass3_skipped and out3 is None and out1 is not None and out2 is not None:
        agr_1v2 = compute_pairwise_agreement(out1, out2)["agreement"]
        has_mod = bool(out1.get("has_modification", False))

        if has_mod:
            # Both Gemini passes agree there's a modification with full
            # aspect agreement — pick out1 (primary pass).
            final = out1
        else:
            # Both agree no_modification — vote on thread_type.
            types = [
                out1.get("thread_type", "statement"),
                out2.get("thread_type", "statement"),
            ]
            most_common_type = _Counter(types).most_common(1)[0][0]
            final = {
                "modifications": [],
                "has_modification": False,
                "thread_type": most_common_type,
            }

        return {
            "final_label": final,
            "vote_method": "gemini_unanimous",
            "needs_review": False,
            "votes": {
                "has_mod_true": 2 if has_mod else 0,
                "has_mod_false": 0 if has_mod else 2,
            },
            "agreement_1v2": agr_1v2,
            "agreement_1v3": "skipped",
            "agreement_2v3": "skipped",
        }

    # ─── Normal 3-pass (or partial) majority vote ───────────────────────
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

    # ─── Pick final label ───────────────────────────────────────────────
    if has_mod_votes >= 2:
        # Majority says has_mod=True. Pick the mod-output with most modifications.
        mod_outputs = [o for o in valid if o.get("has_modification", False)]
        final = max(mod_outputs, key=lambda o: len(o.get("modifications", [])))
    elif has_mod_false >= 2:
        # Majority says has_mod=False. Vote on thread_type instead of hardcoding.
        no_mod_outputs = [o for o in valid if not o.get("has_modification", False)]
        types = [o.get("thread_type", "statement") for o in no_mod_outputs]
        # Tie-break by frequency, then by pass order (pass 1 wins ties)
        # Counter.most_common preserves insertion order for ties in Python 3.7+
        type_counts = _Counter(types)
        most_common_type = type_counts.most_common(1)[0][0]
        final = {
            "modifications": [],
            "has_modification": False,
            "thread_type": most_common_type,
        }
    else:
        # Edge case: len(valid)==2 and they disagree on has_mod.
        # Fall back to pass 1 if available.
        final = valid[0]

    # ─── Decide vote_method and needs_review ────────────────────────────
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
               record_index: dict, batch_size: int = BATCH_SIZE,
               all_records: List[dict] = None, output_path: Path = None,
               save_every_n_batches: int = 10):
    """
    Generic batch re-labeling pass for pass 2 and 3.

    v10.1 — now handles 503 overload separately from 429 rate-limit:
      - 503 / model overloaded:  exponential backoff (60s → 120s → 240s, capped 300s)
                                 indefinitely (does NOT count toward kill switch)
      - 429 / rate limit:        15s sleep + retry (does NOT count toward kill switch)
      - Auth failure:             abort immediately, no retry
      - Transient (500/conn):     short sleep + retry, counts toward kill switch
      - Other:                    advance past batch, counts toward kill switch
    """
    if not todo:
        print(f"✅ No records need {pass_name}!")
        return {}

    total = len(todo)
    num_batches = (total + batch_size - 1) // batch_size
    print(f"\n[{pass_name}] Processing {total} records in ~{num_batches} batches of {batch_size}")
    print(f"  Teacher: {teacher.model_name}")
    print(f"  Storing in: {output_field}")
    if output_path and all_records:
        print(f"  💾 Incremental save: every {save_every_n_batches} batches ({save_every_n_batches * batch_size} threads)")
    print()

    stats = {
        "total": 0, "success": 0, "batch_errors": 0, "parse_failures": 0,
        "with_mods": 0, "no_mods": 0,
        "rate_limit_hits": 0, "overload_hits": 0, "transient_hits": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    consecutive_hard_errors = 0   # only counts 'transient'/'other', not overload/rate_limit
    overload_streak = 0           # for exponential backoff
    idx = 0
    batch_num = 0
    batches_since_save = 0

    while idx < total:
        batch_num += 1
        batch = todo[idx:idx + batch_size]

        results, error, _is_rl_legacy = teacher.label_batch(batch)

        if error or results is None:
            err_kind = classify_api_error(Exception(error)) if error else "other"
            stats["batch_errors"] += 1
            short_err = (error or 'No results')[:120]
            print(f"  [Batch {batch_num}] ERROR ({err_kind}): {short_err}")

            if err_kind == "auth":
                print(f"\n  ❌ Authentication failure — check your API key. Aborting.")
                break

            if err_kind == "overloaded":
                # Exponential backoff: 60, 120, 240, 300, 300, ...
                overload_streak += 1
                stats["overload_hits"] += 1
                wait_time = min(300, 60 * (2 ** (overload_streak - 1)))
                print(f"  🔥 Server overloaded (streak={overload_streak}). "
                      f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue  # retry same batch

            # Any other recoverable error resets the overload streak
            overload_streak = 0

            if err_kind == "rate_limit":
                stats["rate_limit_hits"] += 1
                wait_time = 30
                print(f"  ⏳ Rate limited (your quota). Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue  # retry same batch

            if err_kind == "transient":
                stats["transient_hits"] += 1
                consecutive_hard_errors += 1
                wait_time = 10
                print(f"  ⚠ Transient error (consec={consecutive_hard_errors}). "
                      f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                if consecutive_hard_errors >= 5:
                    print(f"\n  ❌ 5 consecutive transient errors — stopping {pass_name}.")
                    break
                continue  # retry same batch

            # err_kind == "other" — unknown error, advance past batch
            consecutive_hard_errors += 1
            if consecutive_hard_errors >= 5:
                print(f"\n  ❌ 5 consecutive unknown errors — stopping {pass_name}.")
                break
            idx += batch_size
            time.sleep(2)
            continue

        # Success path
        consecutive_hard_errors = 0
        overload_streak = 0

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
              f"errs={stats['batch_errors']} (rl={stats['rate_limit_hits']} "
              f"503={stats['overload_hits']})")

        idx += batch_size
        batches_since_save += 1

        # Incremental save every N batches
        if output_path and all_records and batches_since_save >= save_every_n_batches:
            print(f"  💾 Saving progress ({stats['success']} records labeled so far)...")
            backup_file(output_path, f".incremental_{batch_num}.bak")
            write_all_records(all_records, output_path)
            batches_since_save = 0
            print(f"  ✓ Saved to {output_path}")

        time.sleep(teacher.MIN_DELAY)

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    return stats

# =============================================================================
# PASS 1: INITIAL LABELING (Gemini only)
# =============================================================================

def run_pass1(input_path: str, output_path: str,
              gemini_key: str = None,
              gemini_model: str = None,
              limit: int = None, skip_existing: bool = True,
              batch_size: int = BATCH_SIZE):
    """Pass 1: Label threads with Gemini. temp=0.0 (greedy, deterministic)."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gemini = None

    if gemini_key and GEMINI_AVAILABLE:
        try:
            # MASTER PLAN §4.3 COMMITTED CHANGE: temp 0.1 → 0.0 for boundary stability
            gemini = GeminiTeacher(api_key=gemini_key, model_name=gemini_model, temperature=0.0)
            print(f"✓ Gemini ready: {gemini.model_name} (temp={gemini.temperature})")
        except Exception as e:
            print(f"⚠ Gemini init failed: {e}")

    if not gemini:
        print("❌ No teacher available! Set GOOGLE_API_KEY in .env")
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

    # GEMINI BUDGET PRE-CHECK
    # Gemini 3.1 Flash Lite limits: RPM=15, TPM=250K, RPD=500
    GEMINI_RPD = 500
    if num_batches > GEMINI_RPD:
        days_needed = (num_batches + GEMINI_RPD - 1) // GEMINI_RPD
        print(f"⚠️  WARNING: {num_batches} calls needed, Gemini free-tier RPD = {GEMINI_RPD}")
        print(f"   This run will need ~{days_needed} days at the current quota.")
        print(f"   Options: (a) use --limit to split across days,")
        print(f"            (b) upgrade to paid tier (no daily cap),")
        print(f"            (c) accept the multi-day timeline.")
        print(f"   Resume is supported (existing thread_ids will be skipped).\n")

    teacher = gemini
    teacher_name = gemini.model_name
    delay = GeminiTeacher.MIN_DELAY
    consecutive_errors = 0
    overload_streak = 0

    stats = {
        "total": 0, "with_mods": 0, "no_mods": 0,
        "gemini_labeled": 0,
        "batch_errors": 0, "parse_failures": 0,
        "aspects": {"SUBSTITUTION": 0, "QUANTITY": 0, "TECHNIQUE": 0, "ADDITION": 0},
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    thread_idx = 0
    batch_num = 0
    
    while thread_idx < total_threads:
        batch_num += 1
        batch = threads[thread_idx:thread_idx + batch_size]

        results, error, is_rl = teacher.label_batch(batch)

        if error or results is None:
            err_kind = classify_api_error(Exception(error)) if error else "other"
            stats["batch_errors"] += 1
            short_err = (error or 'No results')[:120]
            print(f"  [Batch {batch_num}] ERROR ({err_kind}): {short_err}")

            if err_kind == "auth":
                print(f"\n  ❌ Authentication failure — check your API key. Aborting.")
                break

            if err_kind == "overloaded":
                overload_streak += 1
                wait_time = min(300, 60 * (2 ** (overload_streak - 1)))
                print(f"  🔥 Server overloaded (streak={overload_streak}). "
                      f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue  # retry same batch — overload doesn't count toward kill switch

            overload_streak = 0  # reset on any non-overload error

            if err_kind == "rate_limit":
                wait_time = 30
                print(f"  ⏳ Rate limited (your quota). Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue  # retry same batch

            if err_kind == "transient":
                consecutive_errors += 1
                wait_time = 10
                print(f"  ⚠ Transient error (consec={consecutive_errors}). "
                      f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                if consecutive_errors >= 5:
                    print(f"\n  ❌ 5 consecutive transient errors — stopping.")
                    break
                continue  # retry same batch

            # 'other' — unknown, advance past
            consecutive_errors += 1
            if consecutive_errors >= 5:
                print(f"\n  ❌ 5 consecutive unknown errors — stopping.")
                break
            thread_idx += batch_size
            time.sleep(2)
            continue
        else:
            consecutive_errors = 0
            overload_streak = 0

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

            stats["gemini_labeled"] += 1

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
                "pass3_skipped": None,            # NEW (v10): set by Pass 3 tiebreaker filter
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
        print(f"  [Batch {batch_num}: {done}/{total_threads}] (Gemini) "
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
    print(f"    With mods:       {stats['with_mods']}")
    print(f"    No mods:         {stats['no_mods']}")
    print(f"    Parse failures:  {stats['parse_failures']}")
    print(f"  Aspects: {stats['aspects']}")
    print(f"  Total in output:   ~{stats['total_in_output']}")
    print(f"  Output: {output_path}")
    print(f"\n  Next step: run --second-pass")
    print(f"{'='*60}")


# =============================================================================
# PASS 2: INTRA-ANNOTATOR (same Gemini, temp=0.3)
# =============================================================================

def run_pass2(output_path: str,
              gemini_key: str = None,
              gemini_model: str = None,
              limit: int = None, batch_size: int = BATCH_SIZE):
    """Pass 2: Re-label with same Gemini at temperature=0.3 (LOCKED per master plan)."""

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
                       "PASS 2 — Intra-annotator", record_index, batch_size,
                       all_records=records, output_path=output_path)

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
    # Preview Pass 3 quota savings if --third-pass runs in tiebreaker mode
    full_pct = 100.0 * agreement_counts.get("full", 0) / max(1, sum(agreement_counts.values()))
    print(f"\n  Tiebreaker preview: ~{full_pct:.0f}% of records have FULL agreement")
    print(f"     and would be SKIPPED by Pass 3 in tiebreaker mode.")
    print(f"\n  Next step: run --third-pass (tiebreaker mode is the default)")
    print(f"{'='*60}")


# =============================================================================
# PASS 3: INTER-ANNOTATOR (Cerebras Qwen 235B) — TIEBREAKER MODE (default)
# =============================================================================

def run_pass3(output_path: str,
              cerebras_key: str = None, cerebras_model: str = None,
              limit: int = None, batch_size: int = BATCH_SIZE,
              full_coverage: bool = False):
    """Pass 3: Re-label with Cerebras Qwen 235B (different model family).

    DEFAULT (tiebreaker mode): only records where Pass 1 ≢ Pass 2 are sent
    to Qwen. Records with full Pass 1 ≡ Pass 2 agreement are flagged
    pass3_skipped=True and treated as "gemini_unanimous" at finalize time.
    Cuts Qwen API usage substantially (typically 30-70%).

    Set full_coverage=True to send EVERY record to Qwen (legacy behavior,
    research/comparison only).
    """

    output_path = Path(output_path)
    if not output_path.exists():
        print(f"❌ {output_path} not found. Run pass 1 first.")
        return

    if not cerebras_key:
        print("❌ CEREBRAS_API_KEY required for pass 3")
        return

    # HARD CLAMP: Cerebras TPM=30K limit means batch_size > 10 will exceed TPM
    # at 5 RPM with the v2 prompt (~3.5K tokens). Master plan §4.6 already
    # specifies --batch-size 10 for Pass 3, but enforce it defensively here.
    if batch_size > CerebrasTeacher.MAX_BATCH_SIZE:
        print(f"⚠️  WARNING: Cerebras batch_size {batch_size} exceeds safe max "
              f"({CerebrasTeacher.MAX_BATCH_SIZE}).")
        print(f"   Clamping to {CerebrasTeacher.MAX_BATCH_SIZE} to stay within "
              f"30K TPM limit at 5 RPM.")
        batch_size = CerebrasTeacher.MAX_BATCH_SIZE

    cerebras = CerebrasTeacher(api_key=cerebras_key, model_name=cerebras_model)
    print(f"✓ Cerebras ready: {cerebras.model_name}")
    print(f"  Rate limits: 5 RPM, 14.4K RPD, 30K TPM, 65K context")
    print(f"  MIN_DELAY: {cerebras.MIN_DELAY}s between calls")
    print(f"  Batch size: {batch_size}")
    print(f"  Mode: {'FULL COVERAGE (every record)' if full_coverage else 'TIEBREAKER (skip full-agreement records)'}")

    records = load_all_records(output_path)
    print(f"Loaded {len(records)} records from {output_path}")

    # ─── Filter pending records (haven't been labeled or skipped yet) ───
    all_pending = [
        r for r in records
        if r.get("third_teacher_output") is None
        and not r.get("pass3_skipped", False)
    ]

    if not all_pending:
        print("✅ All records already have pass 3 labels (or were skipped)!")
        return

    # ─── Tiebreaker filter (NEW v10) ────────────────────────────────────
    skip_records = []
    run_records = []
    no_pass2_count = 0

    if full_coverage:
        run_records = all_pending
    else:
        for r in all_pending:
            out2 = r.get("second_teacher_output")
            if out2 is None:
                # Pass 2 hasn't run for this record — can't decide, default to running Pass 3.
                no_pass2_count += 1
                run_records.append(r)
            elif should_skip_pass3(r):
                skip_records.append(r)
            else:
                run_records.append(r)

    # Persist skip flags BEFORE making any API calls (in case of interruption)
    if skip_records:
        for r in skip_records:
            r["pass3_skipped"] = True
            r["agreement_1v3"] = "skipped"
            r["agreement_2v3"] = "skipped"
        backup_file(output_path, ".pre_pass3_skip.bak")
        write_all_records(records, output_path)

    print(f"\n{'─'*60}")
    print(f"  TIEBREAKER FILTER")
    print(f"{'─'*60}")
    print(f"  Pending records:          {len(all_pending)}")
    if not full_coverage:
        print(f"  Skipped (Pass 1 ≡ Pass 2): {len(skip_records)}  → marked 'gemini_unanimous'")
        if no_pass2_count:
            print(f"  No Pass 2 yet (run anyway): {no_pass2_count}")
    print(f"  Sending to Qwen:          {len(run_records)}")
    if all_pending:
        saved_pct = 100.0 * len(skip_records) / len(all_pending)
        print(f"  Quota saved by tiebreaker: ~{saved_pct:.1f}%")
    print(f"{'─'*60}\n")

    todo = run_records
    if limit:
        todo = todo[:limit]

    if not todo:
        print("✅ Tiebreaker filter left nothing to send to Qwen — Pass 3 done.")
        # Re-write final agreement state for skipped records and exit cleanly.
        write_all_records(records, output_path)
        return

    record_index = {r["thread_id"]: r for r in records}

    stats = run_repass(todo, cerebras, "third_teacher_output", "third_teacher_model",
                       "PASS 3 — Inter-annotator (tiebreaker)", record_index, batch_size,
                       all_records=records, output_path=output_path)

    # ─── Compute pairwise agreements (ONLY for records that got Pass 3) ─
    agr_1v3 = {"full": 0, "partial": 0, "none": 0, "error": 0}
    agr_2v3 = {"full": 0, "partial": 0, "none": 0, "error": 0}
    for rec in records:
        # Skipped records keep their "skipped" markers — don't overwrite.
        if rec.get("pass3_skipped"):
            continue
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
    stats["mode"] = "full_coverage" if full_coverage else "tiebreaker"
    stats["pass3_skipped_count"] = len(skip_records)
    stats["pass3_run_count"] = len(run_records)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  ✅ Pass 3 Done! (mode: {stats['mode']})")
    print(f"{'='*60}")
    print(f"  Skipped (gemini_unanimous): {len(skip_records)}")
    print(f"  Sent to Qwen:               {stats.get('total', 0)}")
    print(f"    Labeled successfully:     {stats.get('success', 0)}")
    print(f"    With mods:                {stats.get('with_mods', 0)}")
    print(f"    No mods:                  {stats.get('no_mods', 0)}")
    print(f"    Parse failures:           {stats.get('parse_failures', 0)}")
    print(f"\n  Agreement (pass1 vs pass3 — Qwen-labeled subset only):")
    for level, count in agr_1v3.items():
        pct = count / max(1, sum(agr_1v3.values())) * 100
        print(f"    {level:10s}  {count:5d}  ({pct:.1f}%)")
    print(f"  Agreement (pass2 vs pass3 — Qwen-labeled subset only):")
    for level, count in agr_2v3.items():
        pct = count / max(1, sum(agr_2v3.values())) * 100
        print(f"    {level:10s}  {count:5d}  ({pct:.1f}%)")
    print(f"\n  Next step: run --finalize to compute majority vote + validate spans")
    print(f"{'='*60}")


# =============================================================================
# SPAN VALIDATION (catches teacher hallucinations — Bug #2 from triage)
# =============================================================================

def _normalize_for_span_check(s: str) -> str:
    """Light normalization for span containment check.

    We do NOT lowercase Hebrew (no case), but we collapse whitespace and
    strip leading/trailing punctuation so that 'X.' still matches 'X' in
    the source text. We keep the comparison strict otherwise — if a span
    isn't a substring of the thread text after this, it's a hallucination.
    """
    if not s:
        return ""
    # Collapse runs of whitespace
    s = " ".join(s.split())
    # Strip a small set of trailing/leading punct that LLMs tend to add
    s = s.strip(" \t\n\r.,!?;:\"'`׳״()[]{}")
    return s


def _span_in_thread(span: str, thread_text: str) -> bool:
    """Return True iff span (after light normalization) appears in thread_text."""
    span_n = _normalize_for_span_check(span)
    if not span_n:
        # Empty span after normalization — treat as hallucinated (LLM gave us nothing usable)
        return False
    text_n = _normalize_for_span_check(thread_text)
    return span_n in text_n


def validate_and_clean_spans(rec: dict) -> dict:
    """Validate every span in rec['final_label'] against the thread's actual text.

    Drops modifications whose span does not appear in
    top_comment_text + replies_texts. If all modifications are dropped,
    flips has_modification to False and clears the modifications list.

    Adds a 'span_validation' diagnostic block to the record:
        {
          "checked": int,        # total mods inspected
          "kept": int,
          "dropped": int,
          "dropped_examples": [{"span": str, "aspect": str, "source_comment": str}, ...],
          "flipped_to_no_mod": bool,
        }

    Returns the (mutated) record.
    """
    final_label = rec.get("final_label") or {}
    mods = final_label.get("modifications", []) or []

    if not mods:
        rec["span_validation"] = {
            "checked": 0, "kept": 0, "dropped": 0,
            "dropped_examples": [], "flipped_to_no_mod": False,
        }
        return rec

    top = rec.get("top_comment_text", "") or ""
    replies = rec.get("replies_texts", []) or []
    full_text = top + "\n" + "\n".join(replies)

    kept, dropped, dropped_examples = [], 0, []
    for mod in mods:
        span = mod.get("span", "") or ""
        if _span_in_thread(span, full_text):
            kept.append(mod)
        else:
            dropped += 1
            if len(dropped_examples) < 5:  # cap to keep records small
                dropped_examples.append({
                    "span": span,
                    "aspect": mod.get("aspect", ""),
                    "source_comment": mod.get("source_comment", ""),
                })

    flipped = False
    if dropped > 0:
        final_label["modifications"] = kept
        if len(kept) == 0:
            # All mods were hallucinated — flip the thread to no_mod.
            final_label["has_modification"] = False
            # Preserve thread_type if the majority vote already set one,
            # otherwise default to "statement".
            if "thread_type" not in final_label:
                final_label["thread_type"] = "statement"
            flipped = True
        rec["final_label"] = final_label

    rec["span_validation"] = {
        "checked": len(mods),
        "kept": len(kept),
        "dropped": dropped,
        "dropped_examples": dropped_examples,
        "flipped_to_no_mod": flipped,
    }
    return rec


# =============================================================================
# FINALIZE: MAJORITY VOTE  (with span validation hook + selective routing)
# =============================================================================

def run_finalize(output_path: str):
    """Compute majority vote across all 3 passes (or 2-pass gemini_unanimous
    for tiebreaker-skipped records), set final labels, AND validate every
    final span against the actual thread text.

    Hallucinated spans (i.e. spans not found in top_comment_text +
    replies_texts) are dropped from final_label.modifications. If all
    mods for a thread are dropped, the thread is flipped to
    has_modification=False so the training pipeline never sees a
    phantom annotation.
    """

    output_path = Path(output_path)
    records = load_all_records(output_path)
    print(f"Loaded {len(records)} records")

    vote_stats = {
        "unanimous": 0,
        "majority": 0,
        "no_majority": 0,
        "insufficient": 0,
        "gemini_unanimous": 0,   # NEW (v10): Pass 3 skipped via tiebreaker filter
    }
    review_count = 0
    final_mods = 0
    final_no_mods = 0

    # Span validation accumulators (Fix 1)
    val_total_checked = 0
    val_total_kept = 0
    val_total_dropped = 0
    val_threads_with_drops = 0
    val_threads_flipped = 0
    val_examples = []  # capped sample of dropped spans for the stats file

    for rec in records:
        out1 = rec.get("teacher_output")
        out2 = rec.get("second_teacher_output")
        out3 = rec.get("third_teacher_output")
        pass3_skipped = bool(rec.get("pass3_skipped", False))

        vote = compute_majority_vote(out1, out2, out3, pass3_skipped=pass3_skipped)

        rec["final_label"] = vote["final_label"]
        rec["vote_method"] = vote["vote_method"]
        rec["needs_review"] = vote["needs_review"]
        rec["agreement_1v2"] = vote["agreement_1v2"]
        rec["agreement_1v3"] = vote["agreement_1v3"]
        rec["agreement_2v3"] = vote["agreement_2v3"]

        # ─── Span validation (Fix 1) ────────────────────────────────────
        rec = validate_and_clean_spans(rec)
        sv = rec.get("span_validation", {})
        val_total_checked += sv.get("checked", 0)
        val_total_kept += sv.get("kept", 0)
        val_total_dropped += sv.get("dropped", 0)
        if sv.get("dropped", 0) > 0:
            val_threads_with_drops += 1
            for ex in sv.get("dropped_examples", []):
                if len(val_examples) < 50:
                    val_examples.append({
                        "thread_id": rec.get("thread_id", ""),
                        **ex,
                    })
        if sv.get("flipped_to_no_mod", False):
            val_threads_flipped += 1
        # ────────────────────────────────────────────────────────────────

        if vote["vote_method"] in vote_stats:
            vote_stats[vote["vote_method"]] += 1
        if vote["needs_review"]:
            review_count += 1
        # NB: count AFTER validation so the numbers reflect the cleaned data.
        if rec.get("final_label", {}).get("has_modification"):
            final_mods += 1
        else:
            final_no_mods += 1

    backup_file(output_path, ".pre_finalize.bak")
    write_all_records(records, output_path)

    stats = {
        "total": len(records),
        "vote_stats": vote_stats,
        "needs_review": review_count,
        "final_mods": final_mods,
        "final_no_mods": final_no_mods,
        "span_validation": {
            "total_modifications_checked": val_total_checked,
            "kept": val_total_kept,
            "dropped_hallucinations": val_total_dropped,
            "threads_with_at_least_one_drop": val_threads_with_drops,
            "threads_flipped_to_no_mod": val_threads_flipped,
            "drop_rate_pct": (
                100.0 * val_total_dropped / max(1, val_total_checked)
            ),
            "sample_dropped_spans": val_examples,
        },
    }
    stats_path = output_path.parent / "final_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    total = len(records)
    print(f"\n{'='*60}")
    print(f"  ✅ Finalized — Majority Vote Complete!")
    print(f"{'='*60}")
    print(f"  Total records:       {total}")
    print(f"  Final has_mod=True:  {final_mods}")
    print(f"  Final has_mod=False: {final_no_mods}")
    print(f"  Needs manual review: {review_count}")
    print(f"  Vote breakdown:")
    for k, v in vote_stats.items():
        pct = 100.0 * v / max(1, total)
        print(f"    {k:18s} {v:5d}  ({pct:.1f}%)")
    print(f"\n  --- Span Validation (Fix 1) ---")
    print(f"  Modifications checked:   {val_total_checked}")
    print(f"  Kept (span found):       {val_total_kept}")
    print(f"  Dropped (hallucinated):  {val_total_dropped} "
          f"({100.0 * val_total_dropped / max(1, val_total_checked):.2f}%)")
    print(f"  Threads with ≥1 drop:    {val_threads_with_drops}")
    print(f"  Threads flipped to no_mod after drops: {val_threads_flipped}")
    if val_examples:
        print(f"\n  Sample of dropped (hallucinated) spans:")
        for ex in val_examples[:5]:
            print(f"    [{ex.get('aspect','?'):14s}] '{ex.get('span','')[:40]}'  "
                  f"(thread {ex.get('thread_id','?')[:24]})")
    print(f"\n  Stats saved to: {stats_path}")
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
    p3_skipped = sum(1 for r in records if r.get("pass3_skipped"))
    has_final = sum(1 for r in records if r.get("final_label") is not None)

    print(f"\n{'='*60}")
    print(f"  AGREEMENT STATISTICS")
    print(f"{'='*60}")
    print(f"  Total records:              {total}")
    print(f"  Pass 2 done:                {has_p2}")
    print(f"  Pass 3 done (Qwen-labeled): {has_p3}")
    print(f"  Pass 3 skipped (tiebreaker): {p3_skipped}  → 'gemini_unanimous'")
    print(f"  Finalized:                  {has_final}")

    for pair, field_name in [("Pass1 vs Pass2", "agreement_1v2"),
                             ("Pass1 vs Pass3", "agreement_1v3"),
                             ("Pass2 vs Pass3", "agreement_2v3")]:
        counts = {"full": 0, "partial": 0, "none": 0, "error": 0, "skipped": 0}
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
            print(f"    {method:18s}  {count:5d}  ({pct:.1f}%)")
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
        description="Three-Pass Silver Label Generator with Majority Vote (v10 — Tiebreaker Pass 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Pass 1 — Gemini labels all threads:
     python -m src.teacher_labeling.generate_labels \\
         -i data/raw_youtube/threads.jsonl --limit 5000 --batch-size 20

  2. Pass 2 — Same Gemini, temp=0.3 (intra-annotator):
     python -m src.teacher_labeling.generate_labels --second-pass --limit 5000 --batch-size 20

  3. Pass 3 — TIEBREAKER MODE (default): Qwen labels only records where
     Pass 1 ≢ Pass 2 (records with full agreement are skipped):
     python -m src.teacher_labeling.generate_labels --third-pass --batch-size 10

     Or, full coverage (legacy — every record sent to Qwen):
     python -m src.teacher_labeling.generate_labels --third-pass --full-coverage --batch-size 10

  4. Finalize — Majority vote + span validation:
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
                        help="Pass 3: Cerebras Qwen 235B, TIEBREAKER mode by default")
    parser.add_argument("--full-coverage", action="store_true",
                        help="Pass 3: send EVERY record to Qwen (disable tiebreaker filter)")
    parser.add_argument("--finalize", action="store_true",
                        help="Compute majority vote + validate spans + set final labels")
    parser.add_argument("--export-review", action="store_true",
                        help="Export disagreements to needs_review.jsonl")
    parser.add_argument("--agreement-stats", action="store_true",
                        help="Show agreement statistics")

    parser.add_argument("--input", "-i", help="Input threads JSONL (pass 1)")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--gemini-key", default=os.environ.get("GOOGLE_API_KEY"),
                        help="Gemini API key (default: GOOGLE_API_KEY env var)")
    parser.add_argument("--cerebras-key", default=os.environ.get("CEREBRAS_API_KEY"),
                        help="Cerebras API key (default: CEREBRAS_API_KEY env var)")
    parser.add_argument("--gemini-model", default=None,
                        help=f"Gemini model (default: {GeminiTeacher.NAME})")
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
            full_coverage=args.full_coverage,
        )
        return

    if args.second_pass:
        run_pass2(
            output_path=args.output,
            gemini_key=args.gemini_key,
            gemini_model=args.gemini_model,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        return

    if not args.input:
        parser.error("--input / -i is required for pass 1 labeling")

    if not args.gemini_key:
        parser.error(
            "No API key found!\n"
            "Set GOOGLE_API_KEY in .env, or pass --gemini-key."
        )

    run_pass1(
        input_path=args.input,
        output_path=args.output,
        gemini_key=args.gemini_key,
        gemini_model=args.gemini_model,
        limit=args.limit,
        skip_existing=not args.no_skip,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()