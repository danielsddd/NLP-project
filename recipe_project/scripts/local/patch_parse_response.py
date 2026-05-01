#!/usr/bin/env python3
"""
Patch parse_batch_response to handle Qwen tokenizer mangling YouTube thread_ids.

Bug: Qwen-3-235b's tokenizer rewrites parts of YouTube IDs (drops digits/letters
inside long alphanumeric strings). Pass 3 success rate dropped to ~30% because
the validator's `if tid in results:` check fails on the mangled IDs.

Fix: When response length matches batch length, prefer POSITIONAL alignment
over thread_id string matching. Order is preserved by the model — position is
more reliable than the (mangled) string ID.

Usage:
    python scripts/local/patch_parse_response.py
"""

import shutil
import sys
from pathlib import Path

TARGET = Path("src/teacher_labeling/generate_labels.py")
BACKUP = Path("src/teacher_labeling/generate_labels_v21_pre_parsefix.py")


# ─── OLD: ID-match-first, positional fallback only when matched_count == 0 ─────
OLD_BLOCK = '''    if isinstance(parsed, list):
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
                        results[tid] = validated'''


# ─── NEW: positional first when lengths match (handles Qwen ID mangling) ─────
NEW_BLOCK = '''    if isinstance(parsed, list):
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
                        results[tid] = validated'''


def main():
    if not TARGET.exists():
        print(f"❌ Target file not found: {TARGET}")
        print(f"   Run from project root (recipe_project/).")
        sys.exit(1)

    shutil.copy2(TARGET, BACKUP)
    print(f"✓ Backed up: {TARGET} → {BACKUP}")

    content = TARGET.read_text(encoding="utf-8")

    if OLD_BLOCK not in content:
        print(f"\n❌ Could not find target code block in {TARGET}")
        print(f"   The parse_batch_response function may have already been patched")
        print(f"   or modified. Aborting without changes.")
        sys.exit(1)

    if content.count(OLD_BLOCK) > 1:
        print(f"\n❌ Target block appears more than once — ambiguous match. Aborting.")
        sys.exit(1)

    content = content.replace(OLD_BLOCK, NEW_BLOCK)
    TARGET.write_text(content, encoding="utf-8")

    # Syntax check
    import ast
    try:
        ast.parse(content)
        print(f"✓ Python syntax valid")
    except SyntaxError as e:
        print(f"\n❌ SYNTAX ERROR after patch: {e}")
        print(f"   Restoring backup...")
        shutil.copy2(BACKUP, TARGET)
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  ✅ PATCH APPLIED")
    print(f"{'=' * 60}")
    print(f"  Backup: {BACKUP}")
    print(f"  Effect: Pass 3 will now use positional matching when batch lengths")
    print(f"          align (the common case). This recovers Qwen outputs that")
    print(f"          had thread_ids mangled by tokenization.")
    print(f"\n  Next: re-run Pass 3 to label the ~5,300 records currently null.")
    print(f"        Existing successful Pass 3 records will be skipped (resume).")
    print(f"\n        python -m src.teacher_labeling.generate_labels --third-pass \\")
    print(f"            -o data/silver_labels/teacher_output_v2.jsonl \\")
    print(f"            --batch-size 10")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()