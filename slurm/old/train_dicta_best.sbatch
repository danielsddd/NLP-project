#!/bin/bash
# =============================================================================
# SLURM Script: Train DictaBERT-Large (Softmax + Focal Loss)
# =============================================================================
# Model: dicta-il/dictabert-large
# Uses: train_student.py with class-weighted focal loss
# NOTE: batch-size=8 to avoid OOM on RTX 3090 (24GB)
# Submit: sbatch scripts/slurm/train_dicta_best.sbatch
# =============================================================================

#SBATCH --job-name=dicta_lg
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_dicta_large.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_dicta_large.err

# =============================================================================
# CONFIGURATION
# =============================================================================
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"

MODEL="dicta-il/dictabert-large"
OUTPUT_DIR="models/checkpoints/dictabert_large"

echo "=========================================="
echo "TRAINING: DictaBERT-Large (Softmax + Focal)"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "Model:     $MODEL"
echo "Output:    $OUTPUT_DIR"
echo "NOTE:      batch-size=8 (large model, OOM safety)"
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

# Train — batch-size 8 for large model (avoids OOM on 24GB GPUs)
python -m src.models.train_student \
    --model "$MODEL" \
    --data-dir data/processed \
    --output-dir "$OUTPUT_DIR" \
    --epochs 10 \
    --batch-size 4 \
    --lr 1e-5 \
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
