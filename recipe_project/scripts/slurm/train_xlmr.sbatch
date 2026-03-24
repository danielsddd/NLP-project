#!/bin/bash
# =============================================================================
# SLURM Script: Train XLM-RoBERTa (Softmax + Focal Loss)
# =============================================================================
# Model: xlm-roberta-base
# Uses: train_student.py with class-weighted focal loss
# Submit: sbatch scripts/slurm/train_xlmr.sbatch
# =============================================================================

#SBATCH --job-name=xlmr
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_3090|a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_xlmr.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_xlmr.err

# =============================================================================
# CONFIGURATION
# =============================================================================
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction"

MODEL="xlm-roberta-base"
OUTPUT_DIR="models/checkpoints/xlm_roberta"

echo "=========================================="
echo "TRAINING: XLM-RoBERTa (Softmax + Focal Loss)"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "Model:     $MODEL"
echo "Output:    $OUTPUT_DIR"
echo "=========================================="

# Activate environment
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp

# Set cache paths (avoid home quota)
export HF_HOME=$USER_DIR/.cache/huggingface
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
    --class-weights balanced \
    --focal-loss \
    --focal-gamma 2.0 \
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
