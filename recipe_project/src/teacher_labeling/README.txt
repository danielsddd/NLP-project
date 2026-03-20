# Teacher Labeling — Silver Label Generator
## Three-Pass Majority Vote System (v9)

This module generates "silver labels" for training data by having multiple LLM
teachers independently label Hebrew YouTube comment threads, then combining
their outputs via majority vote.

---

## Overview

The labeling runs in 3 passes + a finalization step:

  Pass 1: Gemini (primary) + Groq fallback → teacher_output
  Pass 2: Same Gemini at temp=0.3         → second_teacher_output  (intra-annotator)
  Pass 3: Cerebras Qwen 3 235B            → third_teacher_output   (inter-annotator)
  Finalize: Majority vote (2/3 agree)     → final_label

Records where all 3 disagree are flagged needs_review=true for manual inspection.

---

## Prerequisites

1. API keys in your .env file (project root):

    GOOGLE_API_KEY=your_gemini_key
    GROQ_API_KEY=your_groq_key
    CEREBRAS_API_KEY=your_cerebras_key

2. Install optional dependencies:

    pip install google-genai groq openai

   Only the providers you use need to be installed.
   Gemini alone is enough for Pass 1 + Pass 2.

3. Input file: data/raw_youtube/threads.jsonl
   (produced by the YouTube collector step)

---

## Usage — Full Workflow

Run all commands from the project root (recipe_project/).

### Pass 1 — Initial labeling (Gemini primary, Groq fallback)

    python -m src.teacher_labeling.generate_labels \
        -i data/raw_youtube/threads.jsonl \
        --limit 5000 \
        --batch-size 20

  - Reads threads from the input file
  - Sends batches of threads to Gemini (falls back to Groq on rate limit)
  - Writes results to data/silver_labels/teacher_output.jsonl
  - Saves stats to data/silver_labels/pass1_stats.json
  - Safe to interrupt and resume: already-labeled threads are skipped

### Pass 2 — Intra-annotator agreement (same Gemini, different temperature)

    python -m src.teacher_labeling.generate_labels \
        --second-pass \
        --limit 5000 \
        --batch-size 20

  - Re-labels all threads with Gemini at temperature=0.3
  - Stores in second_teacher_output field
  - Computes pairwise agreement (pass1 vs pass2)
  - Creates backup: teacher_output.jsonl.pre_pass2.bak

### Pass 3 — Inter-annotator agreement (different model family)

    python -m src.teacher_labeling.generate_labels \
        --third-pass \
        --limit 5000 \
        --batch-size 20

  - Labels with Cerebras Qwen 3 235B (different architecture than Gemini)
  - Stores in third_teacher_output field
  - Computes pairwise agreement (pass1 vs pass3, pass2 vs pass3)
  - Creates backup: teacher_output.jsonl.pre_pass3.bak

### Finalize — Majority vote

    python -m src.teacher_labeling.generate_labels --finalize

  - Computes majority vote across all 3 passes
  - Sets final_label, vote_method, needs_review for each record
  - Vote methods: unanimous (all 3 agree), majority (2/3 agree), no_majority (all disagree)
  - Creates backup: teacher_output.jsonl.pre_finalize.bak
  - Saves stats to data/silver_labels/final_stats.json

### Export disagreements for manual review

    python -m src.teacher_labeling.generate_labels --export-review

  - Exports records with needs_review=true to data/silver_labels/needs_review.jsonl

### View agreement statistics

    python -m src.teacher_labeling.generate_labels --agreement-stats

  - Prints pairwise agreement rates, vote method distribution, model usage

---

## CLI Reference

    python -m src.teacher_labeling.generate_labels [OPTIONS]

  Mode flags (pick one):
    -i, --input FILE        Pass 1: input threads JSONL (required for pass 1)
    --second-pass           Pass 2: re-label with Gemini temp=0.3
    --third-pass            Pass 3: re-label with Cerebras Qwen 235B
    --finalize              Compute majority vote, set final labels
    --export-review         Export disagreements to needs_review.jsonl
    --agreement-stats       Print agreement statistics

  Options:
    -o, --output FILE       Output JSONL (default: data/silver_labels/teacher_output.jsonl)
    --gemini-key KEY        Gemini API key (default: GOOGLE_API_KEY from .env)
    --groq-key KEY          Groq API key (default: GROQ_API_KEY from .env)
    --cerebras-key KEY      Cerebras API key (default: CEREBRAS_API_KEY from .env)
    --gemini-model NAME     Override Gemini model (default: gemini-3.1-flash-lite-preview)
    --groq-model NAME       Override Groq model (default: llama-3.3-70b-versatile)
    --cerebras-model NAME   Override Cerebras model (default: qwen-3-235b-a22b-instruct-2507)
    --limit N               Max threads to process per run
    --batch-size N          Threads per API call (default: 5)
    --no-skip               Pass 1: re-label already-labeled threads

---

## Output Format

Each record in teacher_output.jsonl contains:

    {
        "thread_id": "Ugz...",
        "video_id": "abc123",
        "channel_id": "UC...",
        "video_title": "...",
        "channel_title": "...",
        "top_comment_text": "Hebrew comment text...",
        "replies_texts": ["reply 1", "reply 2"],
        "has_creator_reply": false,
        "total_likes": 42,

        "teacher_output": { ... },          // Pass 1 result
        "teacher_model": "gemini-...",
        "second_teacher_output": { ... },   // Pass 2 result
        "second_teacher_model": "gemini-...",
        "third_teacher_output": { ... },    // Pass 3 result
        "third_teacher_model": "qwen-...",

        "agreement_1v2": "full",            // full / partial / none / error
        "agreement_1v3": "partial",
        "agreement_2v3": "full",

        "final_label": {                    // Majority vote winner
            "has_modification": true,
            "modifications": [
                {
                    "span": "exact Hebrew text",
                    "aspect": "SUBSTITUTION",
                    "source_comment": "top",
                    "confidence": 0.85
                }
            ],
            "thread_type": "statement"
        },
        "vote_method": "majority",          // unanimous / majority / no_majority
        "needs_review": false
    }

---

## Aspect Types

    SUBSTITUTION  — Replacing an ingredient (e.g. "used butter instead of oil")
    QUANTITY      — Changing amounts (e.g. "doubled the sugar")
    TECHNIQUE     — Changing method/time/temp (e.g. "baked 10 min longer")
    ADDITION      — Adding something not in the original (e.g. "added cinnamon")

---

## Rate Limits and Resumability

  - Gemini free tier: ~15 RPM. The script enforces 6-second delays.
  - Groq free tier: faster, used as fallback when Gemini rate-limits.
  - Cerebras: 8-second delay between batches.
  - All passes are resumable: the script skips already-labeled threads.
  - Backups are created before each pass overwrites the output file.
  - 5 consecutive batch errors will stop the run (likely an API issue).

---

## What Goes Into Preprocessing Next

After finalization, the preprocessing step (prepare_data.py) reads the
final_label field from teacher_output.jsonl and converts the character-level
spans into token-level BIO tags aligned with AlephBERT's tokenizer.

    teacher_output.jsonl → prepare_data.py → train.jsonl / val.jsonl / test.jsonl