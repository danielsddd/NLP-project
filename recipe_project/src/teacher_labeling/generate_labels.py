"""
Teacher Labeling — Gemini + Groq Ensemble
==========================================
Group 11: Daniel Simanovsky & Roei Ben Artzi
NLP 2025a - Recipe Modification Extraction

Sends each comment to TWO models simultaneously:
  - Gemini 3.1 Flash Lite  (50 comments per batch, 500 RPD)
  - Groq Llama 3.3 70B     (25 comments per batch, 1000 RPD)

Agreement    → HIGH CONFIDENCE label → teacher_output.jsonl
Disagreement → flagged for review   → needs_review.jsonl

Resume-safe: re-running skips already-processed comments.

Usage:
    python generate_labels.py \
        --input  youtube_collector/data/raw_youtube/comments.jsonl \
        --output data/silver_labels/teacher_output.jsonl \
        --review data/silver_labels/needs_review.jsonl \
        --limit  50

Dependencies:
    pipenv install google-genai groq tqdm python-dotenv

.env file:
    GOOGLE_API_KEY=...
    GROQ_API_KEY=...
"""

import os
import sys
import json
import time
import logging
import argparse
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
except ImportError:
    print("❌ Missing: pipenv install google-genai")
    sys.exit(1)

try:
    from groq import Groq
except ImportError:
    print("❌ Missing: pipenv install groq")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("❌ Missing: pipenv install tqdm")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================

GEMINI_MODEL      = "gemini-2.5-flash-lite"    # FIX 1: correct model string
GROQ_MODEL        = "llama-3.3-70b-versatile"  # 1,000 RPD free tier

GEMINI_BATCH_SIZE = 50   # comments per Gemini request
GROQ_BATCH_SIZE   = 25   # comments per Groq request

GEMINI_DELAY      = 4.0  # seconds between Gemini calls (15 RPM limit)
GROQ_DELAY        = 2.5  # seconds between Groq calls   (30 RPM limit)

VALID_ASPECTS = {"SUBSTITUTION", "QUANTITY", "TECHNIQUE", "ADDITION"}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Modification:
    span:       str
    aspect:     str
    start_char: Optional[int] = None
    end_char:   Optional[int] = None


@dataclass
class ModelLabel:
    has_modification: bool  = False
    modifications:    list  = field(default_factory=list)
    confidence:       float = 0.0
    error:            Optional[str] = None


@dataclass
class LabeledComment:
    comment_id:    str
    video_id:      str
    text:          str
    like_count:    int
    video_title:   str
    channel_title: str

    gemini_label:  Optional[ModelLabel] = None
    groq_label:    Optional[ModelLabel] = None

    agreement:     Optional[bool]       = None
    final_label:   Optional[ModelLabel] = None
    review_needed: bool                 = False
    labeled_at:    str                  = ""


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert NLP data annotator specializing in Hebrew culinary text.
Your ONLY task is STRICT EXTRACTIVE ANNOTATION. You will analyze YouTube comments and extract recipe modifications.

A "recipe modification" is when a commenter:
- Changes an ingredient quantity (QUANTITY)
- Substitutes one ingredient for another (SUBSTITUTION)
- Changes cooking method, time, temperature, or equipment (TECHNIQUE)
- Adds an ingredient not in the original recipe (ADDITION)

CRITICAL RULES FOR SPAN EXTRACTION:
1. COPY-PASTE ONLY: The `span` MUST be a perfect, exact substring copied character-by-character from the original comment.
2. DO NOT FIX TYPOS: If the user wrote with spelling mistakes (e.g., "בתאם" instead of "בטעם"), you MUST extract the mistake exactly as written.
3. NO PARAPHRASING: Do not summarize the modification. Do not translate slang to formal Hebrew.
4. MULTI-SPAN: If a modification is interrupted by other words, extract EACH part as a separate object. Do not extract long noisy sentences.
5. CONFIDENCE: Rate your certainty 0.0-1.0 that you identified modifications correctly.

If the span you return cannot be found using a simple text.find(span) string match in Python, YOU HAVE FAILED.

Return ONLY valid JSON. No markdown, no explanations, no backticks."""


def build_batch_prompt(comments: list) -> str:
    lines = [
        "Analyze each comment below and return a JSON array.",
        "",
        "Return exactly this structure — one object per comment:",
        "",
        "[",
        "  {",
        '    "index": 0,',
        '    "has_modification": true,',
        '    "confidence": 0.92,',
        '    "modifications": [',
        "      {",
        '        "span": "exact Hebrew text copied from comment",',
        '        "aspect": "QUANTITY"',
        "      }",
        "    ]",
        "  }",
        "]",
        "",
        "RULES:",
        "- Return a JSON array with EXACTLY the same number of objects as input comments",
        "- aspect must be one of: SUBSTITUTION, QUANTITY, TECHNIQUE, ADDITION",
        "- span must be copied EXACTLY from the comment — character-by-character, including typos",
        "- If no modification: has_modification=false, modifications=[]",
        "- Return ONLY the JSON array. No markdown. No explanation. No backticks.",
        "",
        "Comments to analyze:",
    ]
    for i, c in enumerate(comments):
        lines.append(f"[{i}] {c['text']}")
    return "\n".join(lines)


# =============================================================================
# JSON PARSING + HALLUCINATION SAFETY NET
# =============================================================================

def extract_json(raw: str) -> list:
    """Robustly extract a JSON array from model output."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    raw = raw.strip("`").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []


def parse_batch_response(raw: str, comments: list, model_name: str) -> list:
    """Parse model batch response into ModelLabel list with hallucination safety net."""
    results = [ModelLabel(error="not processed") for _ in comments]
    parsed  = extract_json(raw)

    if not parsed:
        logging.warning(f"[{model_name}] Failed to parse JSON response")
        return results

    if len(parsed) != len(comments):
        logging.warning(
            f"[{model_name}] Expected {len(comments)} results, got {len(parsed)}"
        )

    for item in parsed:
        idx = item.get("index", -1)
        if not isinstance(idx, int) or idx < 0 or idx >= len(comments):
            continue

        mods = []
        for m in item.get("modifications", []):
            span   = m.get("span", "").strip()
            aspect = m.get("aspect", "").upper()

            if not span or aspect not in VALID_ASPECTS:
                continue

            original = comments[idx]["text"]
            start    = original.find(span)

            # HALLUCINATION SAFETY NET: reject spans not in original text
            if start == -1:
                logging.warning(
                    f"[{model_name}] HALLUCINATION: span '{span[:50]}' "
                    f"not found in '{original[:60]}'. Dropping."
                )
                continue

            mods.append(Modification(
                span=span,
                aspect=aspect,
                start_char=start,
                end_char=start + len(span),
            ))

        results[idx] = ModelLabel(
            has_modification=bool(item.get("has_modification", len(mods) > 0)),
            modifications=mods,
            confidence=float(item.get("confidence", 0.5)),
        )

    return results


# =============================================================================
# GEMINI CLIENT — FIX 3: proper SDK usage with config object
# =============================================================================

class GeminiLabeler:
    def __init__(self, api_key: str):
        self.client    = google_genai.Client(api_key=api_key)
        self.last_call = 0.0

    def label_batch(self, comments: list) -> list:
        elapsed = time.time() - self.last_call
        if elapsed < GEMINI_DELAY:
            time.sleep(GEMINI_DELAY - elapsed)

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=build_batch_prompt(comments),
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,   # FIX 3: low temp for JSON stability
                ),
            )
            self.last_call = time.time()
            return parse_batch_response(response.text, comments, "Gemini")
        except Exception as e:
            logging.error(f"[Gemini] API error: {e}")
            self.last_call = time.time()
            return [ModelLabel(error=str(e)) for _ in comments]


# =============================================================================
# GROQ CLIENT — FIX 4: bumped max_tokens to 8192
# =============================================================================

class GroqLabeler:
    def __init__(self, api_key: str):
        self.client    = Groq(api_key=api_key)
        self.last_call = 0.0

    def label_batch(self, comments: list) -> list:
        elapsed = time.time() - self.last_call
        if elapsed < GROQ_DELAY:
            time.sleep(GROQ_DELAY - elapsed)

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_batch_prompt(comments)},
                ],
                temperature=0.1,
                max_tokens=8192,   # FIX 4: bumped up to avoid cut-off JSON
            )
            self.last_call = time.time()
            raw = response.choices[0].message.content
            return parse_batch_response(raw, comments, "Groq")
        except Exception as e:
            logging.error(f"[Groq] API error: {e}")
            self.last_call = time.time()
            return [ModelLabel(error=str(e)) for _ in comments]


# =============================================================================
# ENSEMBLE LOGIC — FIX 2: exact span overlap check
# =============================================================================

def labels_agree(gemini: ModelLabel, groq: ModelLabel) -> bool:
    """
    Two labels agree if:
    1. Both agree on has_modification
    2. If both say true → at least one span pair shares the same aspect
       AND the spans actually overlap (one contains the other)
    """
    if gemini.error or groq.error:
        return False

    # Both say no modification
    if not gemini.has_modification and not groq.has_modification:
        return True

    # One says yes, one says no
    if gemini.has_modification != groq.has_modification:
        return False

    # FIX 2: EXACT SPAN OVERLAP CHECK — aspect match + span containment
    for g_mod in gemini.modifications:
        for gr_mod in groq.modifications:
            if g_mod.aspect == gr_mod.aspect:
                # Check if spans overlap (one contains the other)
                if g_mod.span in gr_mod.span or gr_mod.span in g_mod.span:
                    return True

    return False


def merge_labels(gemini: ModelLabel, groq: ModelLabel) -> ModelLabel:
    """
    Merge two agreeing labels:
    - Use Gemini spans (better Hebrew handling)
    - Keep only modifications confirmed by span overlap with Groq
    - Average confidence scores
    """
    if not gemini.has_modification:
        return ModelLabel(has_modification=False, confidence=1.0)

    # Keep only Gemini mods that have a matching overlapping span in Groq
    groq_confirmed = set()
    for gr_mod in groq.modifications:
        groq_confirmed.add(gr_mod.aspect)

    kept_mods = [
        m for m in gemini.modifications
        if m.aspect in groq_confirmed
    ]

    if not kept_mods:
        kept_mods = gemini.modifications  # fallback: keep all Gemini mods

    return ModelLabel(
        has_modification=True,
        modifications=kept_mods,
        confidence=(gemini.confidence + groq.confidence) / 2,
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class EnsembleLabeler:
    def __init__(
        self,
        gemini_key:  str,
        groq_key:    str,
        output_file: str = "data/silver_labels/teacher_output.jsonl",
        review_file: str = "data/silver_labels/needs_review.jsonl",
    ):
        self.gemini = GeminiLabeler(gemini_key)
        self.groq   = GroqLabeler(groq_key)
        self.output = Path(output_file)
        self.review = Path(review_file)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.done_ids = self._load_done_ids()
        logging.info(f"Already processed: {len(self.done_ids)} comments (resume-safe)")

    def _load_done_ids(self) -> set:
        done = set()
        for fpath in [self.output, self.review]:
            if fpath.exists():
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        try:
                            done.add(json.loads(line)["comment_id"])
                        except Exception:
                            pass
        return done

    def _save(self, labeled: LabeledComment):
        fpath = self.review if labeled.review_needed else self.output
        with open(fpath, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(labeled), ensure_ascii=False) + "\n")

    def process_file(self, input_file: str, limit: int = None):
        comments = []
        with open(input_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    comments.append(json.loads(line))

        if limit:
            comments = comments[:limit]
            logging.info(f"TEST MODE: processing first {limit} comments only")

        todo = [c for c in comments if c["comment_id"] not in self.done_ids]
        logging.info(
            f"Total: {len(comments)} | Done: {len(self.done_ids)} | Remaining: {len(todo)}"
        )

        if not todo:
            print("✅ All comments already processed!")
            return

        stats = {
            "total":    len(todo),
            "agreed":   0,
            "disagreed": 0,
            "errors":   0,
            "with_mod": 0,
        }

        gemini_batches = [todo[i:i+GEMINI_BATCH_SIZE] for i in range(0, len(todo), GEMINI_BATCH_SIZE)]
        groq_batches   = [todo[i:i+GROQ_BATCH_SIZE]   for i in range(0, len(todo), GROQ_BATCH_SIZE)]

        # Step 1: Groq labels everything first
        groq_labels = {}
        print(f"\n🤖 Step 1/2: Groq labeling ({len(groq_batches)} batches × {GROQ_BATCH_SIZE} comments)...")
        for batch in tqdm(groq_batches, unit="batch"):
            results = self.groq.label_batch(batch)
            for comment, label in zip(batch, results):
                groq_labels[comment["comment_id"]] = label

        # Step 2: Gemini labels + ensemble decision
        print(f"\n🤖 Step 2/2: Gemini labeling + ensemble ({len(gemini_batches)} batches × {GEMINI_BATCH_SIZE} comments)...")
        for batch in tqdm(gemini_batches, unit="batch"):
            gemini_results = self.gemini.label_batch(batch)

            for comment, g_label in zip(batch, gemini_results):
                gr_label = groq_labels.get(
                    comment["comment_id"], ModelLabel(error="not processed")
                )

                agree  = labels_agree(g_label, gr_label)
                final  = merge_labels(g_label, gr_label) if agree else g_label
                review = not agree

                if g_label.error or gr_label.error:
                    stats["errors"] += 1
                    review = True
                elif agree:
                    stats["agreed"] += 1
                else:
                    stats["disagreed"] += 1

                if final.has_modification:
                    stats["with_mod"] += 1

                self._save(LabeledComment(
                    comment_id=comment["comment_id"],
                    video_id=comment["video_id"],
                    text=comment["text"],
                    like_count=comment.get("like_count", 0),
                    video_title=comment.get("video_title", ""),
                    channel_title=comment.get("channel_title", ""),
                    gemini_label=g_label,
                    groq_label=gr_label,
                    agreement=agree,
                    final_label=final,
                    review_needed=review,
                    labeled_at=datetime.now().isoformat(),
                ))

        print("\n" + "=" * 55)
        print("  ENSEMBLE LABELING SUMMARY")
        print("=" * 55)
        print(f"  Total processed:    {stats['total']:,}")
        print(f"  Models agreed:      {stats['agreed']:,}  ({100*stats['agreed']/max(stats['total'],1):.1f}%)")
        print(f"  Models disagreed:   {stats['disagreed']:,}  ({100*stats['disagreed']/max(stats['total'],1):.1f}%)  → review file")
        print(f"  Errors:             {stats['errors']:,}")
        print(f"  With modifications: {stats['with_mod']:,}  ({100*stats['with_mod']/max(stats['total'],1):.1f}%)")
        print(f"\n  ✅ High-confidence → {self.output}")
        print(f"  ⚠️  Needs review    → {self.review}")
        print("=" * 55)


# =============================================================================
# CLI
# =============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Ensemble Teacher Labeling (Gemini + Groq)"
    )
    parser.add_argument("--input",      required=True,  help="Path to comments.jsonl")
    parser.add_argument("--output",     default="data/silver_labels/teacher_output.jsonl")
    parser.add_argument("--review",     default="data/silver_labels/needs_review.jsonl")
    parser.add_argument("--gemini-key", help="Gemini API key (or GOOGLE_API_KEY in .env)")
    parser.add_argument("--groq-key",   help="Groq API key   (or GROQ_API_KEY in .env)")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Process only first N comments (for testing)")
    args = parser.parse_args()

    gemini_key = args.gemini_key or os.environ.get("GOOGLE_API_KEY")
    groq_key   = args.groq_key   or os.environ.get("GROQ_API_KEY")

    if not gemini_key:
        print("❌ No Gemini key! Set GOOGLE_API_KEY in .env or use --gemini-key")
        sys.exit(1)
    if not groq_key:
        print("❌ No Groq key! Set GROQ_API_KEY in .env or use --groq-key")
        sys.exit(1)

    EnsembleLabeler(
        gemini_key=gemini_key,
        groq_key=groq_key,
        output_file=args.output,
        review_file=args.review,
    ).process_file(args.input, limit=args.limit)


if __name__ == "__main__":
    main()