#!/bin/bash
# =============================================================================
# Download Silver Labels from HuggingFace
# =============================================================================
# Pulls the silver-labeled teacher outputs (NOT in git because of size) from:
#   https://huggingface.co/datasets/DanielDDDS/recipe-modifications
#
# Idempotent: if files already exist and match expected size, skips download.
# Safe to run on the cluster in a login node or inside a job.
#
# Usage:
#     bash scripts/download_silver_labels.sh
#     bash scripts/download_silver_labels.sh --force   # re-download even if present
#
# Requirements:
#     pip install huggingface_hub
#     (already in requirements.txt; installed in `recipe_nlp` conda env)
# =============================================================================

set -euo pipefail

REPO_ID="DanielDDDS/recipe-modifications"
REPO_TYPE="dataset"
DEST_DIR="data/silver_labels"

# Files we want from the HF repo. Format: "<hf_filename>:<local_name>"
# (The enriched file may not be uploaded yet — script handles missing files gracefully.)
FILES=(
    "teacher_output.jsonl:teacher_output.jsonl"
    "threads_positives_focus_labeled.jsonl:threads_positives_focus_labeled.jsonl"
)

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

# --- Sanity: run from project root ---
if [[ ! -d "src/preprocessing" ]]; then
    echo "ERROR: Must run from recipe_project/ root (no src/preprocessing found)"
    echo "       Current dir: $(pwd)"
    exit 1
fi

mkdir -p "$DEST_DIR"

# --- Sanity: huggingface_hub available? ---
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed in the current Python env."
    echo "       Activate the conda env first:"
    echo "         conda activate recipe_nlp"
    echo "       Then install:"
    echo "         pip install huggingface_hub"
    exit 1
fi

echo "=========================================="
echo "  Downloading silver labels"
echo "=========================================="
echo "  Source: huggingface.co/datasets/$REPO_ID"
echo "  Dest:   $DEST_DIR/"
echo "  Force:  $FORCE"
echo ""

for entry in "${FILES[@]}"; do
    hf_name="${entry%%:*}"
    local_name="${entry##*:}"
    local_path="$DEST_DIR/$local_name"

    if [[ -f "$local_path" && $FORCE -eq 0 ]]; then
        size=$(stat -c %s "$local_path" 2>/dev/null || stat -f %z "$local_path")
        echo "  [SKIP] $local_name already present (${size} bytes). Use --force to redownload."
        continue
    fi

    echo "  [GET]  $hf_name -> $local_path"

    # hf_hub_download returns the cache path; we then copy to DEST_DIR.
    # Using python one-liner for portability and clear error messages.
    python - <<PYEOF
import shutil, sys
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

try:
    cached = hf_hub_download(
        repo_id="$REPO_ID",
        repo_type="$REPO_TYPE",
        filename="$hf_name",
    )
    shutil.copyfile(cached, "$local_path")
    print(f"         OK: copied to $local_path")
except EntryNotFoundError:
    print(f"         WARN: '$hf_name' not found in repo $REPO_ID. Skipping.")
    sys.exit(0)
except Exception as e:
    print(f"         ERROR: {type(e).__name__}: {e}")
    sys.exit(1)
PYEOF
done

echo ""
echo "=========================================="
echo "  Local silver files:"
echo "=========================================="
ls -lh "$DEST_DIR"/ 2>/dev/null | grep -v '^total' | awk '{printf "    %-45s %s\n", $NF, $5}'

# Line counts (quick sanity check)
echo ""
echo "  Record counts:"
for f in "$DEST_DIR"/*.jsonl; do
    if [[ -f "$f" ]]; then
        count=$(wc -l < "$f")
        printf "    %-45s %6d lines\n" "$(basename "$f")" "$count"
    fi
done

echo ""
echo "  Done. Proceed to preprocessing:"
echo "    python -m src.preprocessing.prepare_data_merged \\"
echo "        --original data/silver_labels/teacher_output.jsonl \\"
echo "        --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \\"
echo "        --output-dir data/processed \\"
echo "        --model dicta-il/dictabert \\"
echo "        --downsample-ratio 3.0"