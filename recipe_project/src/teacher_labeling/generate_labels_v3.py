"""
Teacher Labeling Module v3 — Thread-Level
==========================================
Group 11: Daniel Simanovsky & Roei Ben Artzi
NLP 2025a - Recipe Modification Extraction

MAJOR CHANGE IN v3:
  Labels are generated at the THREAD level, not the comment level.
  A thread = top comment + all replies.

  Core rule enforced in the prompt:
    "A question comment is NEVER the signal.
     The REPLY to that question is the signal.
     The question is only context."

  New output fields:
    - target         : the ingredient/technique in the recipe being modified
    - alternative    : the suggested replacement or modification value
    - signal_type    : confirmed / creator_validated / negative / mixed / suggestion
    - confidence     : 0.0–1.0
    - quantity_note  : "אותה כמות" etc. if mentioned
    - warning        : negative aspect if mixed
    - output_note    : READY-TO-DISPLAY Hebrew string for inject.py

BUG FIX (v3):
  - channel_id now passed through to output (was missing in v1/v2)
  - Thread-level context prevents questions being labeled as modifications

Usage:
    # Using Gemini (recommended)
    python -m src.teacher_labeling.generate_labels \\
        --input data/raw_youtube/threads.jsonl \\
        --provider gemini \\
        --api-key YOUR_GOOGLE_API_KEY

    # Resume interrupted run (skips already-labeled threads)
    python -m src.teacher_labeling.generate_labels \\
        --input data/raw_youtube/threads.jsonl

    # Test with 50 threads
    python -m src.teacher_labeling.generate_labels \\
        --input data/raw_youtube/threads.jsonl --limit 50
"""

import json
import os
import time
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed. Run: pip install google-generativeai")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Modification:
    """A single extracted recipe modification from a thread."""
    # ── Span (for BIO alignment) ─────────────────────────────────────────────
    span:       str           # EXACT text from comment (Hebrew preserved)
    aspect:     str           # SUBSTITUTION | QUANTITY | TECHNIQUE | ADDITION
    sentiment:  str = "constructive"
    start_char: Optional[int] = None
    end_char:   Optional[int] = None

    # ── Semantic fields (new in v3) ───────────────────────────────────────────
    target:       str = ""    # ingredient/technique in the recipe (e.g. "קמח")
    alternative:  str = ""    # the modification value (e.g. "קמח כוסמין")
    signal_type:  str = "confirmed"
    #   confirmed         → user stated they did it + positive result
    #   creator_validated → channel owner confirmed in reply
    #   multi_user        → multiple users confirmed same thing
    #   negative          → tried it and it failed / not recommended
    #   mixed             → conflicting replies (some good, some bad)
    #   suggestion        → suggested but unconfirmed
    confidence:   float = 0.0
    quantity_note: str = ""   # "אותה כמות", "כפול", etc.
    warning:      str = ""    # negative note for mixed signal_type

    # ── Ready-to-display (new in v3) ─────────────────────────────────────────
    output_note:  str = ""
    # Full Hebrew string ready to inject into recipe, e.g.:
    # "אפשר להחליף בקמח כוסמין באותה כמות (עלול לצאת קצת דחוס)"
    # "לא מומלץ קמח ללא גלוטן — הלחם לא יתפח"


@dataclass
class TeacherOutput:
    modifications:     List[Modification] = field(default_factory=list)
    has_modification:  bool = False
    overall_sentiment: str = "neutral"
    source:            str = ""   # e.g. "question_answered_positively"
    raw_response:      Optional[str] = None
    error:             Optional[str] = None


# =============================================================================
# SYSTEM PROMPT  (v3 — thread-aware with question rules)
# =============================================================================

SYSTEM_PROMPT = """You are an expert culinary NLP assistant specializing in Hebrew text analysis.
Your task is to analyze YouTube comment THREADS from cooking videos and extract recipe modifications.

A thread contains:
  - [TOP COMMENT]: the comment posted by a viewer
  - [REPLY N]: replies to that comment (may include the video creator)

=== CRITICAL RULE: QUESTIONS vs MODIFICATIONS ===

If the top comment is a QUESTION (ends with ? or contains האם/אפשר/כדאי):

  RULE 1: The question itself is NEVER a modification. Extract from REPLIES only.
          Use the question as context to understand what the replies refer to.

  RULE 2: If there are NO replies, OR all replies are unhelpful
          ("גם אני רוצה לדעת", emojis only, "לא יודע") →
          return has_modification: false immediately. Do not invent a label.

  RULE 3: NEGATIVE reply to a question:
          "לא, יצא נוראי" / "ניסיתי ולא יצא" / "לא כדאי" →
          signal_type: "negative"
          output_note starts with: "לא מומלץ..."

  RULE 4: POSITIVE reply to a question:
          "כן! יצא מעולה" / "בטח, אותה כמות" →
          signal_type: "confirmed"
          output_note starts with: "אפשר..."

  RULE 5: CREATOR reply (marked [CREATOR REPLY]):
          Any creator reply confirming → signal_type: "creator_validated", confidence: 0.95
          Any creator reply rejecting  → signal_type: "negative", confidence: 0.95

  RULE 6: CONFLICTING replies (one good, one bad):
          signal_type: "mixed"
          output_note includes both: "אפשר... (עלול לצאת...)"
          Also populate the 'warning' field with the negative note.

=== FOR NON-QUESTION TOP COMMENTS ===

  RULE 7: "החלפתי X ב-Y" / "השתמשתי ב-Y במקום X" → confirmed substitution
  RULE 8: "הוספתי עוד X" / "הכפלתי את X" → confirmed quantity/addition
  RULE 9: "אפיתי 35 דקות במקום 25" → confirmed technique
  RULE 10: "ניסיתי עם X ולא יצא" → negative signal

=== OUTPUT FORMAT ===

Respond ONLY with valid JSON (no markdown, no preamble):

{
  "modifications": [
    {
      "span":          "<exact text from comment — Hebrew preserved>",
      "aspect":        "SUBSTITUTION|QUANTITY|TECHNIQUE|ADDITION",
      "sentiment":     "constructive|positive",
      "target":        "<ingredient or step in the recipe being modified>",
      "alternative":   "<the modification: new ingredient, amount, method>",
      "signal_type":   "confirmed|creator_validated|negative|mixed|suggestion",
      "confidence":    0.0-1.0,
      "quantity_note": "<amount info if mentioned, else empty string>",
      "warning":       "<negative note if mixed, else empty string>",
      "output_note":   "<complete ready-to-display Hebrew sentence>"
    }
  ],
  "has_modification": true|false,
  "overall_sentiment": "constructive|positive|neutral",
  "source": "<question_answered_positively|question_answered_negatively|confirmed_statement|etc>"
}

=== output_note FORMATTING RULES ===

  Positive/confirmed: "אפשר להחליף [target] ב[alternative][quantity if any]"
  Confirmed+warning:  "אפשר להחליף [target] ב[alternative] (⚠️ [warning])"
  Negative:           "לא מומלץ [alternative] — [reason from comment]"
  Creator validated:  "אפשר [alternative] (אושר על ידי יוצר הערוץ)"
  Suggestion only:    "ניתן לנסות [alternative] (לא אושר)"
"""


FEW_SHOT_EXAMPLES = """
=== EXAMPLES ===

EXAMPLE 1 — Question with positive answer:
[TOP COMMENT] "אפשר במקום קמח רגיל להשתמש בקמח כוסמין?"
[REPLY 1, user] "כן! ניסיתי ויצא מעולה, אותה כמות"
[REPLY 2, user] "אני גם עושה את זה תמיד, יוצא יותר בריא"

Output:
{
  "modifications": [{
    "span": "קמח כוסמין",
    "aspect": "SUBSTITUTION",
    "sentiment": "constructive",
    "target": "קמח",
    "alternative": "קמח כוסמין",
    "signal_type": "confirmed",
    "confidence": 0.90,
    "quantity_note": "אותה כמות",
    "warning": "",
    "output_note": "אפשר להחליף קמח בקמח כוסמין באותה כמות"
  }],
  "has_modification": true,
  "overall_sentiment": "constructive",
  "source": "question_answered_positively"
}

EXAMPLE 2 — Question with NEGATIVE answer:
[TOP COMMENT] "האם אפשר להשתמש בקמח ללא גלוטן?"
[REPLY 1, user] "ניסיתי ולא יצא, הלחם לא תפח בכלל"

Output:
{
  "modifications": [{
    "span": "קמח ללא גלוטן",
    "aspect": "SUBSTITUTION",
    "sentiment": "constructive",
    "target": "קמח",
    "alternative": "קמח ללא גלוטן",
    "signal_type": "negative",
    "confidence": 0.85,
    "quantity_note": "",
    "warning": "הלחם לא יתפח בכלל",
    "output_note": "לא מומלץ קמח ללא גלוטן — הלחם לא יתפח"
  }],
  "has_modification": true,
  "overall_sentiment": "constructive",
  "source": "question_answered_negatively"
}

EXAMPLE 3 — Question with NO meaningful answer:
[TOP COMMENT] "אפשר לעשות עם חלב שקדים?"
[REPLY 1, user] "גם אני רוצה לדעת!"
[REPLY 2, user] "😋😋"

Output:
{
  "modifications": [],
  "has_modification": false,
  "overall_sentiment": "neutral",
  "source": "question_unanswered"
}

EXAMPLE 4 — Confirmed statement (not a question):
[TOP COMMENT] "עשיתי עם חמאה תמרים במקום חמאה רגילה ויצא מדהים! יותר עשיר"

Output:
{
  "modifications": [{
    "span": "חמאה תמרים במקום חמאה רגילה",
    "aspect": "SUBSTITUTION",
    "sentiment": "constructive",
    "target": "חמאה",
    "alternative": "חמאה תמרים",
    "signal_type": "confirmed",
    "confidence": 0.85,
    "quantity_note": "",
    "warning": "",
    "output_note": "אפשר להחליף חמאה בחמאה תמרים — יצא יותר עשיר"
  }],
  "has_modification": true,
  "overall_sentiment": "constructive",
  "source": "confirmed_statement"
}

EXAMPLE 5 — Creator reply:
[TOP COMMENT] "אפשר לאפות ב-160 במקום 180?"
[CREATOR REPLY] "כן, אבל תוסיפו 10 דקות לזמן האפייה"

Output:
{
  "modifications": [{
    "span": "לאפות ב-160 במקום 180",
    "aspect": "TECHNIQUE",
    "sentiment": "constructive",
    "target": "טמפרטורת אפייה",
    "alternative": "160 מעלות",
    "signal_type": "creator_validated",
    "confidence": 0.95,
    "quantity_note": "הוסיפו 10 דקות לזמן האפייה",
    "warning": "",
    "output_note": "אפשר לאפות ב-160 מעלות (הוסיפו 10 דקות — אושר על ידי יוצר הערוץ)"
  }],
  "has_modification": true,
  "overall_sentiment": "constructive",
  "source": "question_answered_by_creator"
}

EXAMPLE 6 — Mixed replies:
[TOP COMMENT] "אפשר להחליף בקמח כוסמין?"
[REPLY 1, user] "כן יצא טוב"
[REPLY 2, user] "ניסיתי, יצא קצת דחוס"

Output:
{
  "modifications": [{
    "span": "קמח כוסמין",
    "aspect": "SUBSTITUTION",
    "sentiment": "constructive",
    "target": "קמח",
    "alternative": "קמח כוסמין",
    "signal_type": "mixed",
    "confidence": 0.60,
    "quantity_note": "",
    "warning": "עלול לצאת קצת דחוס",
    "output_note": "אפשר להחליף בקמח כוסמין (⚠️ עלול לצאת קצת דחוס)"
  }],
  "has_modification": true,
  "overall_sentiment": "constructive",
  "source": "question_mixed_answers"
}
"""


# =============================================================================
# HELPER — FORMAT THREAD FOR PROMPT
# =============================================================================

def format_thread_for_prompt(thread: Dict) -> str:
    """Convert a thread dict to the prompt input format."""
    top  = thread.get('top_comment', {})
    reps = thread.get('replies', [])

    lines = [f'[TOP COMMENT] "{top.get("text", "")}"']
    for i, r in enumerate(reps, 1):
        prefix = "[CREATOR REPLY]" if r.get('is_creator') else f"[REPLY {i}, user]"
        lines.append(f'{prefix} "{r.get("text", "")}"')

    return "\n".join(lines)


# =============================================================================
# TEACHER MODELS
# =============================================================================

class GeminiTeacher:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed.")
        genai.configure(api_key=api_key)
        self.model      = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.gen_config = genai.types.GenerationConfig(temperature=0.0, max_output_tokens=1024)

    def generate(self, thread_text: str) -> TeacherOutput:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{FEW_SHOT_EXAMPLES}\n\n"
            f"=== NOW ANALYZE THIS THREAD ===\n\n"
            f"{thread_text}\n\n"
            f"Output (JSON only, no markdown):"
        )
        try:
            resp = self.model.generate_content(prompt, generation_config=self.gen_config)
            return self._parse(resp.text, thread_text)
        except Exception as e:
            return TeacherOutput(error=str(e))

    def _parse(self, response: str, thread_text: str) -> TeacherOutput:
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```json?\n?', '', cleaned)
                cleaned = re.sub(r'\n?```$', '', cleaned)
            data  = json.loads(cleaned)
            mods  = self._parse_mods(data.get("modifications", []), thread_text)
            return TeacherOutput(
                modifications=mods,
                has_modification=data.get("has_modification", len(mods) > 0),
                overall_sentiment=data.get("overall_sentiment", "neutral"),
                source=data.get("source", ""),
                raw_response=response,
            )
        except json.JSONDecodeError as e:
            return TeacherOutput(error=f"JSON parse error: {e}", raw_response=response)

    def _parse_mods(self, raw: List[Dict], thread_text: str) -> List[Modification]:
        mods = []
        # Find all text in thread for span search
        all_text = re.sub(r'\[.*?\]\s*"?', ' ', thread_text)  # strip [TAG] prefixes
        for m in raw:
            span = m.get("span", "")
            sc   = all_text.find(span) if span else -1
            mods.append(Modification(
                span=span,
                aspect=m.get("aspect", "UNKNOWN"),
                sentiment=m.get("sentiment", "constructive"),
                start_char=sc if sc >= 0 else None,
                end_char=(sc + len(span)) if sc >= 0 else None,
                target=m.get("target", ""),
                alternative=m.get("alternative", ""),
                signal_type=m.get("signal_type", "confirmed"),
                confidence=float(m.get("confidence", 0.0)),
                quantity_note=m.get("quantity_note", ""),
                warning=m.get("warning", ""),
                output_note=m.get("output_note", ""),
            ))
        return mods


class OpenAITeacher:
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed.")
        self.client     = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, thread_text: str) -> TeacherOutput:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLES},
                    {"role": "user",   "content": f"Analyze this thread:\n\n{thread_text}\n\nOutput (JSON only):"},
                ],
                temperature=0.0, max_tokens=1024,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            return GeminiTeacher._parse(self, raw, thread_text)
        except Exception as e:
            return TeacherOutput(error=str(e))


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class SilverLabelGenerator:
    """
    Reads threads.jsonl → calls Teacher (Gemini/GPT-4o) → writes teacher_output.jsonl.

    Key behaviors:
      - Labels at THREAD level (top_comment + replies together)
      - Questions without meaningful answers → has_modification: false
      - Creator replies flagged and weighted
      - Safe resume: skips already-labeled thread_ids
      - Incremental writes (flush after each thread)
    """

    def __init__(
        self,
        api_key:             str,
        provider:            str = "gemini",
        model_name:          Optional[str] = None,
        output_dir:          str = "data/silver_labels",
        delay_between_calls: float = 0.5,
        max_retries:         int = 3,
    ):
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay       = delay_between_calls
        self.max_retries = max_retries

        if provider == "gemini":
            self.teacher = GeminiTeacher(api_key, model_name or "gemini-1.5-pro")
        elif provider == "openai":
            self.teacher = OpenAITeacher(api_key, model_name or "gpt-4o")
        else:
            raise ValueError(f"Unknown provider '{provider}'. Use 'gemini' or 'openai'.")

        self.model_name = self.teacher.model_name
        logging.basicConfig(level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s')
        self.logger = logging.getLogger(__name__)

    # ─────────────────────────────────────────────────────────────────────────

    def process_file(
        self,
        input_file:    str,
        output_file:   Optional[str] = None,
        limit:         Optional[int] = None,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Process threads.jsonl and generate silver labels.

        Args:
            input_file:    Path to threads.jsonl (output of collect.py v3)
            output_file:   Output path (default: silver_labels/teacher_output.jsonl)
            limit:         Max threads to process (None = all)
            skip_existing: Skip thread_ids already in output (safe resume)
        """
        input_path  = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_file}")

        output_path = Path(output_file) if output_file \
            else self.output_dir / "teacher_output.jsonl"

        # Resume: load already-labeled IDs
        existing_ids: set = set()
        if skip_existing and output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        existing_ids.add(json.loads(line).get("thread_id"))
                    except Exception:
                        pass
            self.logger.info(f"Resume mode: {len(existing_ids)} threads already labeled, skipping.")

        # Load threads
        threads: List[Dict] = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    t = json.loads(line)
                    if t.get("thread_id") not in existing_ids:
                        threads.append(t)
                except json.JSONDecodeError:
                    continue

        if limit:
            threads = threads[:limit]

        self.logger.info(
            f"Processing {len(threads)} threads with {self.model_name} "
            f"({len(existing_ids)} skipped)"
        )

        stats = {
            "total_processed": 0, "successful": 0, "errors": 0,
            "with_modifications": 0, "total_modifications": 0,
            "discarded_unanswered_questions": 0,
            "aspect_counts": {"SUBSTITUTION": 0, "QUANTITY": 0, "TECHNIQUE": 0, "ADDITION": 0},
            "signal_type_counts": {"confirmed": 0, "creator_validated": 0, "negative": 0,
                                   "mixed": 0, "suggestion": 0},
            "started_at": datetime.now().isoformat(),
        }

        with open(output_path, 'a', encoding='utf-8') as f:
            for thread in tqdm(threads, desc="Labeling threads"):
                thread_text = format_thread_for_prompt(thread)
                result      = self._process_with_retry(thread_text)

                # BUG FIX (v3): channel_id now included in output
                output_data = {
                    "thread_id":     thread.get("thread_id"),
                    "video_id":      thread.get("video_id"),
                    "channel_id":    thread.get("channel_id", ""),   # ← BUG FIX
                    "video_title":   thread.get("video_title", ""),
                    "channel_title": thread.get("channel_title", ""),
                    "source":        thread.get("source", "youtube"),
                    # Keep original text fields for BIO alignment
                    "top_comment_text": thread.get("top_comment", {}).get("text", ""),
                    "replies_texts":    [r.get("text","") for r in thread.get("replies", [])],
                    "has_creator_reply": thread.get("has_creator_reply", False),
                    "total_likes":      thread.get("total_likes", 0),
                    "appearance_count": thread.get("appearance_count", 1),
                    # Teacher output
                    "teacher_output":   self._serialize(result),
                    "teacher_model":    self.model_name,
                    "labeled_at":       datetime.now().isoformat(),
                }

                f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                f.flush()

                # Stats
                stats["total_processed"] += 1
                if result.error:
                    stats["errors"] += 1
                else:
                    stats["successful"] += 1
                    if result.has_modification:
                        stats["with_modifications"] += 1
                        stats["total_modifications"] += len(result.modifications)
                        for mod in result.modifications:
                            if mod.aspect in stats["aspect_counts"]:
                                stats["aspect_counts"][mod.aspect] += 1
                            if mod.signal_type in stats["signal_type_counts"]:
                                stats["signal_type_counts"][mod.signal_type] += 1
                    elif result.source == "question_unanswered":
                        stats["discarded_unanswered_questions"] += 1

                time.sleep(self.delay)

        stats["ended_at"] = datetime.now().isoformat()

        stats_path = self.output_dir / "generation_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        self._print_summary(stats)
        return stats

    def _process_with_retry(self, thread_text: str) -> TeacherOutput:
        last = TeacherOutput(error="No attempts")
        for attempt in range(self.max_retries):
            result = self.teacher.generate(thread_text)
            if not result.error:
                return result
            last = result
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        return last

    def _serialize(self, output: TeacherOutput) -> Dict:
        return {
            "modifications": [
                {
                    "span":          m.span,
                    "aspect":        m.aspect,
                    "sentiment":     m.sentiment,
                    "start_char":    m.start_char,
                    "end_char":      m.end_char,
                    "target":        m.target,
                    "alternative":   m.alternative,
                    "signal_type":   m.signal_type,
                    "confidence":    m.confidence,
                    "quantity_note": m.quantity_note,
                    "warning":       m.warning,
                    "output_note":   m.output_note,
                }
                for m in output.modifications
            ],
            "has_modification":   output.has_modification,
            "overall_sentiment":  output.overall_sentiment,
            "source":             output.source,
            "error":              output.error,
        }

    def _print_summary(self, stats: Dict) -> None:
        print(f"\n{'='*60}\nSILVER LABEL GENERATION SUMMARY\n{'='*60}")
        print(f"  Total processed     : {stats['total_processed']:,}")
        print(f"  Successful          : {stats['successful']:,}")
        print(f"  Errors              : {stats['errors']:,}")
        print(f"  With modifications  : {stats['with_modifications']:,}")
        print(f"  Unanswered questions: {stats['discarded_unanswered_questions']:,}")
        print(f"  Total modifications : {stats['total_modifications']:,}")
        print("\n  Aspect breakdown:")
        for asp, cnt in stats['aspect_counts'].items():
            pct = 100 * cnt / stats['total_modifications'] if stats['total_modifications'] else 0
            print(f"    {asp:<16}: {cnt:>5}  ({pct:.1f}%)")
        print("\n  Signal type breakdown:")
        for st, cnt in stats['signal_type_counts'].items():
            print(f"    {st:<20}: {cnt:>5}")
        if stats['total_processed']:
            rate = 100 * stats['with_modifications'] / stats['total_processed']
            print(f"\n  Modification rate : {rate:.1f}%")
        print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    import argparse
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Generate silver labels (thread-level) using Gemini or GPT-4o",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.teacher_labeling.generate_labels --input data/raw_youtube/threads.jsonl
  python -m src.teacher_labeling.generate_labels --input threads.jsonl --limit 50
  python -m src.teacher_labeling.generate_labels --input threads.jsonl --no-skip  # reprocess all
        """
    )
    parser.add_argument("--input",      "-i", required=True,  help="Input threads.jsonl")
    parser.add_argument("--output",     "-o",                 help="Output file")
    parser.add_argument("--provider",   choices=["gemini","openai"], default="gemini")
    parser.add_argument("--api-key",                          help="API key")
    parser.add_argument("--model",                            help="Model name override")
    parser.add_argument("--limit",      type=int,             help="Max threads to process")
    parser.add_argument("--delay",      type=float, default=0.5)
    parser.add_argument("--retries",    type=int,   default=3)
    parser.add_argument("--no-skip",    action="store_true",  help="Reprocess all")
    parser.add_argument("--output-dir", default="data/silver_labels")
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        env = "GOOGLE_API_KEY" if args.provider == "gemini" else "OPENAI_API_KEY"
        api_key = os.environ.get(env)
    if not api_key:
        env = "GOOGLE_API_KEY" if args.provider == "gemini" else "OPENAI_API_KEY"
        print(f"❌ No API key. Set {env} or use --api-key")
        return 1

    gen = SilverLabelGenerator(
        api_key=api_key, provider=args.provider, model_name=args.model,
        output_dir=args.output_dir, delay_between_calls=args.delay, max_retries=args.retries,
    )
    gen.process_file(
        input_file=args.input, output_file=args.output,
        limit=args.limit, skip_existing=not args.no_skip,
    )
    return 0


if __name__ == "__main__":
    exit(main())