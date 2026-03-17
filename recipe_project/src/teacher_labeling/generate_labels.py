#!/usr/bin/env python3
"""
Teacher Labeling — v4
Generates silver labels using Gemini 1.5 Pro on comment threads.

Usage:
    python -m src.teacher_labeling.generate_labels \
        --input data/raw_youtube/threads_cooking_only.jsonl \
        --api-key YOUR_KEY
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

import google.generativeai as genai

# =============================================================================
# PROMPT
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
5. Praise ("it was delicious") is NOT a modification.
6. aspect must be one of: SUBSTITUTION, QUANTITY, TECHNIQUE, ADDITION.
7. source_comment must be "top" or "reply_1", "reply_2", etc.
8. confidence is 0.0-1.0. Creator replies get 0.90+.

OUTPUT FORMAT (strict JSON, no markdown):
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


def format_thread(thread):
    """Format a thread dict into prompt text."""
    lines = [f'[TOP COMMENT] "{thread["top_comment"]["text"]}"']
    for i, reply in enumerate(thread.get("replies", []), 1):
        role = "creator" if reply.get("is_creator") else "user"
        lines.append(f'[REPLY {i}, {role}] "{reply["text"]}"')
    return "\n".join(lines)

# =============================================================================
# LABELING
# =============================================================================

def label_thread(model, thread):
    """Call Gemini on a single thread. Returns parsed dict or None on error."""
    prompt_text = format_thread(thread)
    try:
        response = model.generate_content(
            prompt_text,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            try:
                return json.loads(text.strip())
            except json.JSONDecodeError:
                return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

def process_file(input_path, output_path, api_key, limit=None, skip_existing=True):
    """Process all threads and generate silver labels."""
    # Setup Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        system_instruction=SYSTEM_PROMPT,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing IDs for resume
    existing_ids = set()
    if skip_existing and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["thread_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
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

    print(f"Processing {len(threads)} threads...")

    # Stats
    stats = {"total": 0, "with_mods": 0, "errors": 0,
             "aspects": {"SUBSTITUTION": 0, "QUANTITY": 0, "TECHNIQUE": 0, "ADDITION": 0}}

    for i, thread in enumerate(threads):
        teacher_output = label_thread(model, thread)
        stats["total"] += 1

        if teacher_output is None:
            stats["errors"] += 1
            print(f"  [{i+1}/{len(threads)}] ERROR: {thread['thread_id']}")
            time.sleep(1)
            continue

        if teacher_output.get("has_modification"):
            stats["with_mods"] += 1
            for mod in teacher_output.get("modifications", []):
                aspect = mod.get("aspect", "")
                if aspect in stats["aspects"]:
                    stats["aspects"][aspect] += 1

        # Build output record
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
            "teacher_model": "gemini-1.5-pro",
            "labeled_at": datetime.now(timezone.utc).isoformat(),
        }

        append_jsonl(record, output_path)

        if (i + 1) % 50 == 0:
            pct = stats["with_mods"] / stats["total"] * 100 if stats["total"] else 0
            print(f"  [{i+1}/{len(threads)}] {stats['with_mods']} with mods ({pct:.0f}%), {stats['errors']} errors")

        time.sleep(0.5)  # Rate limiting

    # Save stats
    stats_path = output_path.parent / "generation_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅ Done: {stats['total']} processed, {stats['with_mods']} with modifications, {stats['errors']} errors")
    print(f"   Aspects: {stats['aspects']}")
    print(f"   Output: {output_path}")
    print(f"   Stats:  {stats_path}")

def append_jsonl(data, path):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
        f.flush()

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate silver labels with Gemini")
    parser.add_argument("--input", "-i", required=True, help="Input threads JSONL")
    parser.add_argument("--output", "-o", default="data/silver_labels/teacher_output.jsonl")
    parser.add_argument("--api-key", required=True, help="Google AI Studio API key")
    parser.add_argument("--limit", type=int, help="Max threads to process")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip existing")
    args = parser.parse_args()

    process_file(args.input, args.output, args.api_key,
                 limit=args.limit, skip_existing=not args.no_skip)

if __name__ == "__main__":
    main()