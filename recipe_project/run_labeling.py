#!/usr/bin/env python3
"""
Multi-Model Auto-Rotating Labeler + Full Consensus Validation
==============================================================

MODE 1 — LABEL (default):
  Each model labels threads until its quota is exhausted, then rotates
  to the next model. Continues until all threads are done or all models
  are exhausted. Resume-safe: rerun same command to continue.

  python run_labeling.py --gemini-key KEY --groq-key KEY

MODE 2 — VALIDATE (--validate):
  Samples N threads from the labeled output. For each thread, ALL models
  EXCEPT the original labeler re-label it. Produces a full consensus
  report: per-thread vote counts, pairwise agreement matrix, per-model
  reliability scores.

  python run_labeling.py --validate --gemini-key KEY --groq-key KEY --validate-n 100

  With 8 models and 100 threads, each thread gets 7 validation labels.
  That's ~700 API calls spread across models — fits in free-tier quotas.
"""

import json
import time
import re
import argparse
import random
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# =============================================================================
# MODEL REGISTRY
# =============================================================================

def build_model_pool(gemini_key=None, groq_key=None):
    pool = []
    if gemini_key:
        for m in ["gemini-2.0-flash", "gemini-2.0-flash-lite",
                   "gemini-1.5-flash", "gemini-1.5-flash-8b"]:
            pool.append(_model_entry("gemini", m, gemini_key, 4.5))
    if groq_key:
        for m in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                   "mixtral-8x7b-32768", "gemma2-9b-it"]:
            pool.append(_model_entry("groq", m, groq_key, 2.5))
    return pool


def _model_entry(provider, model, api_key, min_delay):
    return {
        "provider": provider, "model": model, "api_key": api_key,
        "min_delay": min_delay, "consecutive_errors": 0,
        "total_labeled": 0, "total_errors": 0, "exhausted": False,
    }


def model_id(m):
    return f"{m['provider']}/{m['model']}"


# =============================================================================
# TEACHER WRAPPERS
# =============================================================================

def call_teacher(provider, model_name, api_key, thread_text, system_prompt):
    if provider == "gemini":
        return _call_gemini(thread_text, model_name, api_key, system_prompt)
    else:
        return _call_groq(thread_text, model_name, api_key, system_prompt)


def _call_gemini(thread_text, model_name, api_key, system_prompt):
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        return None, None, "google-genai not installed"
    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model_name, contents=thread_text,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt, temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        return _process_raw(resp.text)
    except Exception as e:
        return _handle_error(e)


def _call_groq(thread_text, model_name, api_key, system_prompt):
    try:
        from groq import Groq
    except ImportError:
        return None, None, "groq not installed"
    try:
        client = Groq(api_key=api_key)
        comp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": thread_text}],
            temperature=0.1, max_tokens=1024,
        )
        return _process_raw(comp.choices[0].message.content)
    except Exception as e:
        return _handle_error(e)


def _process_raw(raw):
    parsed = _parse_json(raw)
    if parsed is None:
        return None, raw, "JSON parse failed"
    validated = _validate(parsed)
    if validated is None:
        return None, raw, "Validation failed"
    return validated, raw, None


def _handle_error(e):
    err = str(e)
    is_rl = any(k in err.lower() for k in ["429", "quota", "rate", "resource", "exhausted"])
    return None, None, f"RATE_LIMIT:{err}" if is_rl else err


# =============================================================================
# SHARED HELPERS
# =============================================================================

VALID_ASPECTS = {"SUBSTITUTION", "QUANTITY", "TECHNIQUE", "ADDITION"}

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


def format_thread(thread):
    lines = [f'[TOP COMMENT] "{thread["top_comment"]["text"]}"']
    for i, reply in enumerate(thread.get("replies", []), 1):
        role = "creator" if reply.get("is_creator") else "user"
        lines.append(f'[REPLY {i}, {role}] "{reply["text"]}"')
    return "\n".join(lines)


def _parse_json(raw):
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
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _validate(parsed):
    if not isinstance(parsed, dict):
        return None
    result = {"modifications": [], "has_modification": False,
              "thread_type": parsed.get("thread_type", "statement")}
    for mod in parsed.get("modifications", []):
        if not isinstance(mod, dict):
            continue
        span = mod.get("span", "").strip()
        aspect = mod.get("aspect", "").upper().strip()
        if not span or aspect not in VALID_ASPECTS:
            continue
        result["modifications"].append({
            "span": span, "aspect": aspect,
            "source_comment": mod.get("source_comment", "top"),
            "confidence": min(1.0, max(0.0, float(mod.get("confidence", 0.5)))),
        })
    result["has_modification"] = len(result["modifications"]) > 0
    return result


def append_jsonl(data, path):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
        f.flush()


def load_existing_ids(path):
    ids = set()
    p = Path(path)
    if p.exists():
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["thread_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


# =============================================================================
# LABEL MODE
# =============================================================================

MAX_CONSECUTIVE_ERRORS = 3


def run_label(input_path, output_path, pool, limit=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = load_existing_ids(output_path)
    if existing_ids:
        print(f"Resuming: {len(existing_ids)} already labeled")

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

    stats = {
        "total": 0, "with_mods": 0, "errors": 0,
        "aspects": {"SUBSTITUTION": 0, "QUANTITY": 0, "TECHNIQUE": 0, "ADDITION": 0},
        "per_model": {},
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    cur = _next_active(pool)
    if not cur:
        print("No models available!")
        return

    print(f"Processing {len(threads)} threads")
    print(f"Models: {[model_id(m) for m in pool if not m['exhausted']]}")
    print(f"Starting: {model_id(cur)}\n")

    for i, thread in enumerate(threads):
        cur = _next_active(pool, cur)
        if not cur:
            print(f"\n⛔ ALL MODELS EXHAUSTED at {i+1}/{len(threads)}. Rerun tomorrow.")
            break

        text = format_thread(thread)
        out, raw, err = call_teacher(cur["provider"], cur["model"], cur["api_key"], text, SYSTEM_PROMPT)

        # Rate limit handling
        if err and err.startswith("RATE_LIMIT:"):
            cur["consecutive_errors"] += 1
            cur["total_errors"] += 1
            if cur["consecutive_errors"] >= MAX_CONSECUTIVE_ERRORS:
                cur["exhausted"] = True
                print(f"\n⚠️  {model_id(cur)} — exhausted after {cur['total_labeled']} threads")
                cur = _next_active(pool)
                if not cur:
                    print(f"⛔ ALL MODELS EXHAUSTED at {i+1}/{len(threads)}")
                    break
                print(f"🔄 Switched to: {model_id(cur)}")
                time.sleep(2)
                out, raw, err = call_teacher(cur["provider"], cur["model"], cur["api_key"], text, SYSTEM_PROMPT)
                if err:
                    if err.startswith("RATE_LIMIT:"):
                        cur["consecutive_errors"] += 1
                    stats["errors"] += 1
                    continue
            else:
                time.sleep(5)
                stats["errors"] += 1
                continue
        elif err:
            cur["total_errors"] += 1
            stats["errors"] += 1
            time.sleep(1)
            continue
        else:
            cur["consecutive_errors"] = 0

        mid = model_id(cur)
        if mid not in stats["per_model"]:
            stats["per_model"][mid] = {"labeled": 0, "with_mods": 0}
        stats["per_model"][mid]["labeled"] += 1
        cur["total_labeled"] += 1
        stats["total"] += 1

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
            "teacher_output": out,
            "teacher_model": mid,
            "labeled_at": datetime.now(timezone.utc).isoformat(),
        }

        if out and out.get("has_modification"):
            stats["with_mods"] += 1
            stats["per_model"][mid]["with_mods"] += 1
            for mod in out.get("modifications", []):
                a = mod.get("aspect", "")
                if a in stats["aspects"]:
                    stats["aspects"][a] += 1

        append_jsonl(record, output_path)

        if (i + 1) % 25 == 0 or (i + 1) == len(threads):
            pct = stats["with_mods"] / stats["total"] * 100 if stats["total"] else 0
            print(f"  [{i+1}/{len(threads)}] mods={stats['with_mods']} ({pct:.0f}%) "
                  f"errors={stats['errors']} model={model_id(cur)} "
                  f"(done {cur['total_labeled']})")

        time.sleep(cur["min_delay"])

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    stats_path = output_path.parent / "generation_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅ LABELING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total: {stats['total']}  |  Mods: {stats['with_mods']}  |  Errors: {stats['errors']}")
    print(f"  Aspects: {stats['aspects']}")
    for mid, ms in stats["per_model"].items():
        print(f"    {mid}: {ms['labeled']} labeled, {ms['with_mods']} with mods")
    exh = [m for m in pool if m["exhausted"]]
    if exh:
        print(f"  Exhausted: {[model_id(m) for m in exh]}")
    remaining = len(threads) - stats["total"]
    if remaining > 0:
        print(f"  ⚠️ {remaining} remaining — rerun to continue")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")


# =============================================================================
# VALIDATE MODE — ALL other models re-label each sampled thread
# =============================================================================

def run_validate(labels_path, output_dir, pool, validate_n=100, seed=42):
    """
    For each sampled thread:
      1. Look up which model originally labeled it
      2. Send it to ALL OTHER available models
      3. Record every model's answer
      4. Compute consensus: how many models agree on has_modification? On aspects?

    Output files:
      validation/validation_labels.jsonl  — one record per (thread, model) pair
      validation/agreement_report.json    — full stats for paper
    """
    labels_path = Path(labels_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_path = output_dir / "validation_labels.jsonl"

    print(f"{'='*60}")
    print(f"FULL CONSENSUS VALIDATION")
    print(f"{'='*60}")

    # Load original labels
    originals = {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                originals[rec["thread_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"Original labels: {len(originals)}")

    # Sample threads
    all_tids = list(originals.keys())
    random.seed(seed)
    random.shuffle(all_tids)
    sample_tids = all_tids[:validate_n]
    print(f"Sampled {len(sample_tids)} threads for validation")

    # Load already-validated (thread_id, model) pairs for resume
    done_pairs = set()
    if val_path.exists():
        with open(val_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_pairs.add((r["thread_id"], r["validation_model"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    if done_pairs:
        print(f"Already validated: {len(done_pairs)} (thread, model) pairs")

    # Build work queue: (thread_id, model_config) for each thread × each non-original model
    work = []
    all_model_ids = [model_id(m) for m in pool]
    print(f"Available models: {all_model_ids}")

    for tid in sample_tids:
        orig_model = originals[tid].get("teacher_model", "")
        for m in pool:
            mid = model_id(m)
            if mid == orig_model:
                continue  # skip the original labeler
            if (tid, mid) in done_pairs:
                continue  # already done
            work.append((tid, m))

    if not work:
        print("All validation work already done. Computing report...")
        _compute_report(val_path, originals, sample_tids, pool, output_dir)
        return

    print(f"Validation calls needed: {len(work)}")
    est = sum(m["min_delay"] for _, m in work) / 60
    print(f"Estimated time: ~{est:.0f} minutes\n")

    # Process
    val_stats = {"total": 0, "errors": 0, "per_model": defaultdict(int)}

    for i, (tid, m) in enumerate(work):
        if m["exhausted"]:
            continue

        rec = originals[tid]
        thread = {
            "thread_id": tid,
            "video_id": rec["video_id"],
            "channel_id": rec["channel_id"],
            "video_title": rec["video_title"],
            "channel_title": rec["channel_title"],
            "top_comment": {"text": rec["top_comment_text"]},
            "replies": [{"text": t, "is_creator": False} for t in rec.get("replies_texts", [])],
        }
        text = format_thread(thread)

        out, raw, err = call_teacher(m["provider"], m["model"], m["api_key"], text, SYSTEM_PROMPT)

        if err and err.startswith("RATE_LIMIT:"):
            m["consecutive_errors"] += 1
            m["total_errors"] += 1
            if m["consecutive_errors"] >= MAX_CONSECUTIVE_ERRORS:
                m["exhausted"] = True
                print(f"\n⚠️  {model_id(m)} — exhausted")
            else:
                time.sleep(5)
            val_stats["errors"] += 1
            continue
        elif err:
            m["total_errors"] += 1
            val_stats["errors"] += 1
            time.sleep(1)
            continue
        else:
            m["consecutive_errors"] = 0

        mid = model_id(m)
        m["total_labeled"] += 1
        val_stats["total"] += 1
        val_stats["per_model"][mid] += 1

        val_record = {
            "thread_id": tid,
            "original_model": rec.get("teacher_model", "unknown"),
            "original_output": rec.get("teacher_output"),
            "validation_model": mid,
            "validation_output": out,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(val_record, val_path)

        if (i + 1) % 50 == 0 or (i + 1) == len(work):
            print(f"  [{i+1}/{len(work)}] done={val_stats['total']} "
                  f"errors={val_stats['errors']} model={mid}")

        time.sleep(m["min_delay"])

    # Compute report
    _compute_report(val_path, originals, sample_tids, pool, output_dir)


def _compute_report(val_path, originals, sample_tids, pool, output_dir):
    """
    Build full consensus report from validation labels.

    For each thread:
      - original label (model X says has_mod=T/F, aspects=[...])
      - N validation labels from other models
      - consensus = majority vote on has_modification
      - aspect consensus = aspects that majority of models agree on

    Report includes:
      - Overall has_modification agreement rate
      - Aspect agreement rate
      - Pairwise agreement matrix
      - Per-model reliability (how often does model X agree with majority?)
      - Per-thread consensus strength (unanimous, majority, split)
      - Examples for paper
    """
    output_dir = Path(output_dir)

    # Load all validation records grouped by thread_id
    val_by_thread = defaultdict(list)
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                val_by_thread[r["thread_id"]].append(r)
            except (json.JSONDecodeError, KeyError):
                continue

    if not val_by_thread:
        print("No validation data found.")
        return

    # Build per-thread consensus
    thread_results = []
    all_models = set()

    for tid in sample_tids:
        if tid not in val_by_thread and tid not in originals:
            continue

        orig = originals.get(tid, {})
        orig_out = orig.get("teacher_output") or {}
        orig_model = orig.get("teacher_model", "unknown")
        orig_has_mod = orig_out.get("has_modification", False)
        orig_aspects = {m.get("aspect") for m in orig_out.get("modifications", [])}

        all_models.add(orig_model)

        # All votes: original + validations
        votes = [{
            "model": orig_model,
            "has_mod": orig_has_mod,
            "aspects": orig_aspects,
            "is_original": True,
        }]

        for vr in val_by_thread.get(tid, []):
            v_out = vr.get("validation_output") or {}
            v_model = vr.get("validation_model", "unknown")
            all_models.add(v_model)
            votes.append({
                "model": v_model,
                "has_mod": v_out.get("has_modification", False),
                "aspects": {m.get("aspect") for m in v_out.get("modifications", [])},
                "is_original": False,
            })

        # Consensus on has_modification
        has_mod_votes = [v["has_mod"] for v in votes]
        yes_count = sum(has_mod_votes)
        no_count = len(has_mod_votes) - yes_count
        consensus_has_mod = yes_count > no_count
        total_voters = len(votes)

        # Consensus strength
        majority_pct = max(yes_count, no_count) / total_voters
        if majority_pct == 1.0:
            strength = "unanimous"
        elif majority_pct >= 0.75:
            strength = "strong"
        elif majority_pct > 0.5:
            strength = "majority"
        else:
            strength = "split"

        # Aspect consensus (among those who found mods)
        aspect_counter = defaultdict(int)
        mod_voters = [v for v in votes if v["has_mod"]]
        for v in mod_voters:
            for a in v["aspects"]:
                aspect_counter[a] += 1
        # Aspects found by majority of mod-voters
        consensus_aspects = set()
        if mod_voters:
            threshold = len(mod_voters) / 2
            consensus_aspects = {a for a, c in aspect_counter.items() if c > threshold}

        # Did original agree with consensus?
        orig_agrees_has_mod = (orig_has_mod == consensus_has_mod)
        orig_agrees_aspects = (orig_aspects == consensus_aspects) if consensus_has_mod else True

        thread_results.append({
            "thread_id": tid,
            "total_voters": total_voters,
            "has_mod_yes": yes_count,
            "has_mod_no": no_count,
            "consensus_has_mod": consensus_has_mod,
            "strength": strength,
            "orig_model": orig_model,
            "orig_has_mod": orig_has_mod,
            "orig_agrees_has_mod": orig_agrees_has_mod,
            "orig_aspects": sorted(orig_aspects),
            "consensus_aspects": sorted(consensus_aspects),
            "orig_agrees_aspects": orig_agrees_aspects,
            "all_votes": [{"model": v["model"], "has_mod": v["has_mod"],
                           "aspects": sorted(v["aspects"])} for v in votes],
        })

    # Aggregate stats
    total = len(thread_results)
    report = {
        "total_threads_validated": total,
        "models_used": sorted(all_models),
        "num_models": len(all_models),

        # Consensus strength distribution
        "consensus_strength": {
            "unanimous": sum(1 for t in thread_results if t["strength"] == "unanimous"),
            "strong": sum(1 for t in thread_results if t["strength"] == "strong"),
            "majority": sum(1 for t in thread_results if t["strength"] == "majority"),
            "split": sum(1 for t in thread_results if t["strength"] == "split"),
        },

        # Original label vs consensus
        "original_vs_consensus": {
            "has_mod_agree": sum(1 for t in thread_results if t["orig_agrees_has_mod"]),
            "has_mod_agree_rate": round(
                sum(1 for t in thread_results if t["orig_agrees_has_mod"]) / total, 4
            ) if total else 0,
            "aspect_agree": sum(1 for t in thread_results
                                if t["orig_agrees_has_mod"] and t["orig_agrees_aspects"]),
        },

        # Per-model reliability: how often does each model agree with majority?
        "per_model_reliability": {},

        # Pairwise agreement matrix
        "pairwise_agreement": {},

        # Examples for paper
        "examples": {
            "unanimous_with_mods": [],
            "unanimous_no_mods": [],
            "disagreements": [],
        },
    }

    # Per-model reliability
    model_agree_counts = defaultdict(lambda: {"total": 0, "agree_has_mod": 0, "agree_aspects": 0})
    for t in thread_results:
        for v in t["all_votes"]:
            m = v["model"]
            model_agree_counts[m]["total"] += 1
            if v["has_mod"] == t["consensus_has_mod"]:
                model_agree_counts[m]["agree_has_mod"] += 1
                if set(v["aspects"]) == set(t["consensus_aspects"]):
                    model_agree_counts[m]["agree_aspects"] += 1

    for m, counts in model_agree_counts.items():
        t = counts["total"]
        report["per_model_reliability"][m] = {
            "total_votes": t,
            "agree_with_majority_has_mod": counts["agree_has_mod"],
            "agree_rate_has_mod": round(counts["agree_has_mod"] / t, 4) if t else 0,
            "agree_with_majority_aspects": counts["agree_aspects"],
            "agree_rate_aspects": round(counts["agree_aspects"] / t, 4) if t else 0,
        }

    # Pairwise agreement
    model_list = sorted(all_models)
    for t in thread_results:
        vote_map = {v["model"]: v for v in t["all_votes"]}
        for j, m1 in enumerate(model_list):
            for m2 in model_list[j+1:]:
                if m1 in vote_map and m2 in vote_map:
                    pair = f"{m1} vs {m2}"
                    if pair not in report["pairwise_agreement"]:
                        report["pairwise_agreement"][pair] = {"total": 0, "agree_has_mod": 0, "agree_aspects": 0}
                    report["pairwise_agreement"][pair]["total"] += 1
                    if vote_map[m1]["has_mod"] == vote_map[m2]["has_mod"]:
                        report["pairwise_agreement"][pair]["agree_has_mod"] += 1
                        if set(vote_map[m1]["aspects"]) == set(vote_map[m2]["aspects"]):
                            report["pairwise_agreement"][pair]["agree_aspects"] += 1

    # Pairwise rates
    for pair, ps in report["pairwise_agreement"].items():
        ps["agree_rate_has_mod"] = round(ps["agree_has_mod"] / ps["total"], 4) if ps["total"] else 0
        ps["agree_rate_aspects"] = round(ps["agree_aspects"] / ps["total"], 4) if ps["total"] else 0

    # Examples
    for t in thread_results:
        if t["strength"] == "unanimous" and t["consensus_has_mod"] and len(report["examples"]["unanimous_with_mods"]) < 3:
            report["examples"]["unanimous_with_mods"].append({
                "thread_id": t["thread_id"],
                "voters": t["total_voters"],
                "aspects": t["consensus_aspects"],
            })
        elif t["strength"] == "unanimous" and not t["consensus_has_mod"] and len(report["examples"]["unanimous_no_mods"]) < 3:
            report["examples"]["unanimous_no_mods"].append({
                "thread_id": t["thread_id"], "voters": t["total_voters"],
            })
        elif t["strength"] in ("split", "majority") and len(report["examples"]["disagreements"]) < 5:
            report["examples"]["disagreements"].append({
                "thread_id": t["thread_id"],
                "yes_votes": t["has_mod_yes"], "no_votes": t["has_mod_no"],
                "votes": t["all_votes"],
            })

    # Save report
    report_path = output_dir / "agreement_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=list)

    # Save per-thread details
    details_path = output_dir / "per_thread_consensus.jsonl"
    with open(details_path, 'w', encoding='utf-8') as f:
        for t in thread_results:
            f.write(json.dumps(t, ensure_ascii=False, default=list) + '\n')

    # Print
    print(f"\n{'='*60}")
    print(f"CONSENSUS VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Threads validated:     {total}")
    print(f"Models used:           {len(all_models)}")
    print(f"")
    cs = report["consensus_strength"]
    print(f"Consensus strength:")
    print(f"  Unanimous:           {cs['unanimous']} ({cs['unanimous']/total*100:.0f}%)")
    print(f"  Strong (≥75%):       {cs['strong']} ({cs['strong']/total*100:.0f}%)")
    print(f"  Majority (>50%):     {cs['majority']} ({cs['majority']/total*100:.0f}%)")
    print(f"  Split (50/50):       {cs['split']} ({cs['split']/total*100:.0f}%)")
    print(f"")
    ovc = report["original_vs_consensus"]
    print(f"Original label vs consensus:")
    print(f"  has_mod agree:       {ovc['has_mod_agree']}/{total} ({ovc['has_mod_agree_rate']*100:.1f}%)")
    print(f"  full agree (+ asp):  {ovc['aspect_agree']}/{total}")
    print(f"")
    print(f"Per-model reliability (agree with majority):")
    for m in sorted(report["per_model_reliability"].keys()):
        r = report["per_model_reliability"][m]
        print(f"  {m:40s}  has_mod={r['agree_rate_has_mod']*100:.0f}%  "
              f"aspects={r['agree_rate_aspects']*100:.0f}%  (n={r['total_votes']})")
    print(f"")
    print(f"Pairwise agreement (top pairs):")
    sorted_pairs = sorted(report["pairwise_agreement"].items(),
                          key=lambda x: x[1]["total"], reverse=True)[:10]
    for pair, ps in sorted_pairs:
        print(f"  {pair}")
        print(f"    n={ps['total']}  has_mod={ps['agree_rate_has_mod']*100:.0f}%  "
              f"aspects={ps['agree_rate_aspects']*100:.0f}%")
    print(f"")
    print(f"Saved:")
    print(f"  Report:    {report_path}")
    print(f"  Details:   {details_path}")
    print(f"{'='*60}")


# =============================================================================
# HELPERS
# =============================================================================

def _next_active(pool, current=None):
    if current and not current["exhausted"]:
        return current
    for m in pool:
        if not m["exhausted"]:
            return m
    return None


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model labeler with full consensus validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
LABEL MODE (finish labeling all threads):
  python run_labeling.py --gemini-key KEY1 --groq-key KEY2
  python run_labeling.py --gemini-key KEY1 --limit 20  # test

VALIDATE MODE (all models re-label a sample):
  python run_labeling.py --validate --gemini-key KEY1 --groq-key KEY2
  python run_labeling.py --validate --gemini-key KEY1 --validate-n 50  # smaller sample
        """,
    )
    parser.add_argument("-i", "--input", default="data/raw_youtube/threads.jsonl")
    parser.add_argument("-o", "--output", default="data/silver_labels/teacher_output.jsonl")
    parser.add_argument("--gemini-key", help="Google AI Studio API key")
    parser.add_argument("--groq-key", help="Groq API key")
    parser.add_argument("--limit", type=int, help="Max threads (label mode)")
    parser.add_argument("--validate", action="store_true", help="Run validation mode")
    parser.add_argument("--validate-n", type=int, default=100,
                        help="Threads to validate (default: 100)")
    parser.add_argument("--validate-dir", default="data/silver_labels/validation")
    args = parser.parse_args()

    if not args.gemini_key and not args.groq_key:
        parser.error("At least one of --gemini-key or --groq-key required")

    pool = build_model_pool(gemini_key=args.gemini_key, groq_key=args.groq_key)
    print(f"Model pool: {len(pool)} models\n")

    if args.validate:
        run_validate(args.output, args.validate_dir, pool, validate_n=args.validate_n)
    else:
        run_label(args.input, args.output, pool, limit=args.limit)


if __name__ == "__main__":
    main()
