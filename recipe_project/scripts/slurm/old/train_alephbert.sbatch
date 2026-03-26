#!/bin/bash
# =============================================================================
# SLURM Script: Train AlephBERT (Softmax + Focal Loss)
# =============================================================================
# Model: onlplab/alephbert-base
# Uses: train_student.py with class-weighted focal loss
# Submit: sbatch scripts/slurm/train_alephbert.sbatch
# =============================================================================

#SBATCH --job-name=alephbert
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_alephbert.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_alephbert.err

# =============================================================================
# CONFIGURATION
# =============================================================================
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="simanovsky2_DIR/recipe_modification_extraction/recipe_project"

MODEL="onlplab/alephbert-base"
OUTPUT_DIR="models/checkpoints/alephbert"

echo "=========================================="
echo "TRAINING: AlephBERT (Softmax + Focal Loss)"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "Model:     $MODEL"
echo "Output:    $OUTPUT_DIR"
echo "=========================================="

# Activate environment
source simanovsky2_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp

# Set cache paths (avoid home quota)
export HF_HOME=simanovsky2_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false

cd $PROJECT_DIR

# Show GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=========================================="

# Verify data
if [ ! -f "data/processed/train_merged.jsonl" ]; then
    echo "ERROR: data/processed/train_merged.jsonl not found!"
    echo "Run preprocessing first."
    exit 1
fi

echo ""
echo "Data files:"
wc -l data/processed/train_merged.jsonl data/processed/val.jsonl
echo ""

# Create output dir
mkdir -p $OUTPUT_DIR

# Train
python -m src.models.train_student \
    --model "$MODEL" \
    --data-dir data/processed \
    --output-dir "$OUTPUT_DIR" \
    --epochs 10 \
    --batch-size 16 \
    --lr 2e-5 \
    --seed 42 \
    --neg-ratio 3.0 \
    --class-weights sqrt \
    --fp16

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "TRAINING COMPLETED SUCCESSFULLY"
    echo "Model saved to: $OUTPUT_DIR/best_model"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "TRAINING FAILED (exit code: $EXIT_CODE)"
    echo "=========================================="
    exit $EXIT_CODE
fi

echo "End Time: $(date)"
echo "=========================================="
