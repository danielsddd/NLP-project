#!/usr/bin/env python3
"""
Patch SYSTEM_PROMPT v2 → v2.1 (ADDITION over-default fix + span length fix).

Applies three targeted edits based on pilot diagnostic:
  1. Rule 3: span length guidance — 2-4 tokens (too short) → 2-6 tokens with
     explicit "include comparison context" rule. Fixes boundary truncation.
  2. Rule 4 ADDITION: require an explicit add-verb. Without הוספתי / שמתי גם /
     etc., the model must not label ADDITION.
  3. Rule 4 callout: replace "default to ADDITION" with explicit decision
     order (SUBSTITUTION/QUANTITY before ADDITION). Fixes 3/7 wrong-aspect
     ADDITION cases observed in pilot.

Backs up the file before patching. Fails loudly if any edit doesn't match
its target (no silent partial patches).

Usage:
    python scripts/local/patch_prompt_v2.py
"""

import shutil
import sys
from pathlib import Path

TARGET = Path("src/teacher_labeling/generate_labels.py")
BACKUP = Path("src/teacher_labeling/generate_labels_v2_backup.py")


# =============================================================================
# THE THREE PATCHES
# =============================================================================

PATCHES = [
    # ─── PATCH 1: Span length guidance ───────────────────────────────────
    {
        "name": "P1: Span length 2-4 → 2-6 with comparison context",
        "old": (
            "  Prefer SHORT spans (2-4 tokens typical). Long spans are almost always wrong.\n"
            "\n"
            "  Example: comment says \"עדיף להוסיף כמה כפות מהרוטב\"\n"
            "           CORRECT span:   \"להוסיף כמה כפות מהרוטב\"\n"
            "           WRONG span:     \"עדיף להוסיף כמה כפות מהרוטב\""
        ),
        "new": (
            "  Span should fully include the modification AND its essential comparison context.\n"
            "  Typical length 2-6 tokens. Longer is OK when context demands it.\n"
            "  Critical: if the comment uses \"X במקום Y\" (X instead of Y), the span MUST\n"
            "  include BOTH X and Y. If it uses \"יותר X\" or \"פחות X\", include both words.\n"
            "  Do NOT chop the comparison and leave only the new value.\n"
            "\n"
            "  Example 1: \"עדיף להוסיף כמה כפות מהרוטב\"\n"
            "             CORRECT span: \"להוסיף כמה כפות מהרוטב\"\n"
            "             WRONG span:   \"עדיף להוסיף כמה כפות מהרוטב\"  (hedge included)\n"
            "\n"
            "  Example 2: \"300 גרם קמח במקום 700 גרם\"\n"
            "             CORRECT span: \"300 גרם קמח במקום 700 גרם\"\n"
            "             WRONG span:   \"300 גרם קמח\"  (comparison context lost)"
        ),
    },

    # ─── PATCH 2: ADDITION definition — require add-verb ─────────────────
    {
        "name": "P2: ADDITION requires an explicit add-verb",
        "old": (
            "  ADDITION     : adding a NEW ingredient that is NOT in the original recipe at all\n"
            "                 Example: \"הוספתי כוסברה טרייה בסוף\" → span: \"כוסברה טרייה\""
        ),
        "new": (
            "  ADDITION     : adding a NEW ingredient that is NOT in the original recipe.\n"
            "                 REQUIRES an explicit add-verb in Hebrew:\n"
            "                   הוספתי, שמתי גם, נתתי, פיזרתי, בנוסף, יחד עם, גם, ועוד\n"
            "                 If NO add-verb appears, do NOT label ADDITION even if a\n"
            "                 new ingredient is mentioned — likely SUBSTITUTION/TECHNIQUE.\n"
            "                 Example POSITIVE: \"הוספתי כוסברה טרייה בסוף\" → span: \"כוסברה טרייה\"\n"
            "                 Example NEGATIVE: \"במקום סוכר שמתי דבש\" → SUBSTITUTION not ADDITION\n"
            "                                    (\"במקום\" present, no add-verb)"
        ),
    },

    # ─── PATCH 3: Aspect decision order (replaces "default to ADDITION") ─
    {
        "name": "P3: Replace 'default to ADDITION' with explicit decision order",
        "old": (
            "  ⚠️  QUANTITY vs ADDITION:\n"
            "      \"more of X\" / a specific amount of X already in the recipe → QUANTITY\n"
            "      \"add X (new ingredient)\" → ADDITION\n"
            "      Default to ADDITION when you cannot tell whether X was already in the recipe."
        ),
        "new": (
            "  ⚠️  ASPECT DECISION ORDER (apply in this exact sequence):\n"
            "      1. If \"במקום\" (instead of) is present:\n"
            "         - Comparing amounts/times/temperatures → QUANTITY\n"
            "         - Comparing ingredients/tools         → SUBSTITUTION\n"
            "         (NEVER ADDITION when \"במקום\" appears)\n"
            "      2. If a comparative quantity word is present (יותר/פחות/כפול/חצי) → QUANTITY\n"
            "      3. If an explicit add-verb is present (הוספתי/שמתי גם/בנוסף/...) → ADDITION\n"
            "      4. If a method/tool/order/time/temperature change is described → TECHNIQUE\n"
            "      5. If NONE of the above clearly applies → set has_modification: false.\n"
            "         DO NOT default to ADDITION. DO NOT guess. Precision over recall."
        ),
    },
]


# =============================================================================
# APPLY
# =============================================================================

def main():
    if not TARGET.exists():
        print(f"❌ Target file not found: {TARGET}")
        print(f"   Run from project root (recipe_project/).")
        sys.exit(1)

    # Backup
    shutil.copy2(TARGET, BACKUP)
    print(f"✓ Backed up: {TARGET} → {BACKUP}")

    # Read
    content = TARGET.read_text(encoding="utf-8")
    original_len = len(content)
    original_md5 = hash(content)

    # Apply patches sequentially, fail loud if any miss
    for i, patch in enumerate(PATCHES, 1):
        name = patch["name"]
        old = patch["old"]
        new = patch["new"]

        if old not in content:
            print(f"\n❌ PATCH {i} FAILED: {name}")
            print(f"   Could not find target string in file.")
            print(f"   Has the prompt been edited since v2 was applied?")
            print(f"   Restoring backup...")
            shutil.copy2(BACKUP, TARGET)
            sys.exit(1)

        if content.count(old) > 1:
            print(f"\n❌ PATCH {i} AMBIGUOUS: {name}")
            print(f"   Target string appears {content.count(old)} times — must be unique.")
            shutil.copy2(BACKUP, TARGET)
            sys.exit(1)

        content = content.replace(old, new)
        print(f"✓ Applied patch {i}/3: {name}")

    # Sanity check: content actually changed
    if hash(content) == original_md5:
        print("\n❌ No changes made — patches matched but produced identical content?")
        sys.exit(1)

    # Write
    TARGET.write_text(content, encoding="utf-8")
    new_len = len(content)
    print(f"\n✓ Wrote {TARGET}")
    print(f"  Size: {original_len} → {new_len} chars (Δ {new_len - original_len:+d})")

    # Quick syntax check on the patched file
    import ast
    try:
        ast.parse(content)
        print(f"✓ Python syntax valid")
    except SyntaxError as e:
        print(f"\n❌ SYNTAX ERROR after patch: {e}")
        print(f"   Restoring backup...")
        shutil.copy2(BACKUP, TARGET)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ✅ PROMPT PATCHED v2 → v2.1")
    print("=" * 60)
    print("  Backup: src/teacher_labeling/generate_labels_v2_backup.py")
    print("  Next: re-run pilot to a NEW output path:")
    print()
    print("    python -m src.teacher_labeling.generate_labels \\")
    print("        -i data/silver_labels/pilot_input.jsonl \\")
    print("        -o data/silver_labels/pilot_output_v21.jsonl \\")
    print("        --batch-size 20")
    print()
    print("    python -m src.teacher_labeling.generate_labels --second-pass \\")
    print("        -o data/silver_labels/pilot_output_v21.jsonl --batch-size 20")
    print()
    print("    python -m src.teacher_labeling.generate_labels --third-pass \\")
    print("        -o data/silver_labels/pilot_output_v21.jsonl --batch-size 10")
    print()
    print("    python -m src.teacher_labeling.generate_labels --finalize \\")
    print("        -o data/silver_labels/pilot_output_v21.jsonl")
    print()
    print("    python scripts/evaluate_teacher.py \\")
    print("        --silver data/silver_labels/pilot_output_v21.jsonl \\")
    print("        --gold data/gold_validation/gold_final.jsonl \\")
    print("        --output results/pilot_teacher_bound_v21.json")
    print("=" * 60)


if __name__ == "__main__":
    main()