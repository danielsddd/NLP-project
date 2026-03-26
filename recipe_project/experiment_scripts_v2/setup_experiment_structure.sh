#!/bin/bash
# Run ONCE to create all directories
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"

echo "Creating experiment directory structure..."

# Log dirs
for m in mbert hebert xlm_roberta alephbert dictabert dictabert_large dictabert_crf preprocessing; do
    mkdir -p "$USER_DIR/logs/slurm_output/$m"
done

# Checkpoint dirs
for m in mbert hebert xlm_roberta alephbert dictabert dictabert_large dictabert_crf; do
    mkdir -p "$PROJECT_DIR/models/checkpoints/$m"
done

# Result dirs
for m in mbert hebert xlm_roberta alephbert dictabert dictabert_large dictabert_crf baselines; do
    mkdir -p "$PROJECT_DIR/results/$m"
done

# Data dirs
for s in "" _ds2 _ds4 _no_enriched _io _unanimous _thread_aware _thread_aware_no_enriched; do
    mkdir -p "$PROJECT_DIR/data/processed$s"
done

# Copy scripts
SRC="$(cd "$(dirname "$0")" && pwd)"
DST="$PROJECT_DIR/scripts/slurm"
for d in preprocessing mbert hebert xlm_roberta alephbert dictabert dictabert_large dictabert_crf; do
    cp -r "$SRC/$d" "$DST/" 2>/dev/null
done
cp "$SRC/submit_all.sh" "$DST/" 2>/dev/null

# Copy helper code
cp "$SRC/helper_code/prepare_thread_aware.py" "$PROJECT_DIR/src/preprocessing/" 2>/dev/null

echo "DONE! Next: sbatch scripts/slurm/preprocessing/01_preprocess_default.sbatch"
