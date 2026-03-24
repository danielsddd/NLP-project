#!/bin/bash
# =============================================================================
# SLURM Script: Train BERT + CRF Model (V7 — no intent conditioning)
# =============================================================================
# Trains DictaBERT + CRF on merged (original + enriched) data.
# V7 CHANGE: Removed --intent-weight (no intent head).
#             Added --class-weights-file for emission weighting.
# Submit with: sbatch scripts/slurm/train_joint.sbatch
# =============================================================================

#SBATCH --job-name=joint_crf
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_3090|a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_joint.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_joint.err

# =============================================================================
# CONFIGURATION — Edit these for different experiments
# =============================================================================
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction"

MODEL_NAME="dicta-il/dictabert"
TRAIN_FILE="train_merged.jsonl"
OUTPUT_DIR="models/checkpoints/joint_crf"
EPOCHS=10
BATCH_SIZE=16
LR=2e-5
SEED=42

echo "=========================================="
echo "BERT + CRF TRAINING (V7 — no intent head)"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Start Time:   $(date)"
echo "Model:        $MODEL_NAME"
echo "Train File:   $TRAIN_FILE"
echo "Epochs:       $EPOCHS"
echo "Batch Size:   $BATCH_SIZE"
echo "Learning Rate: $LR (head: $(echo "$LR * 5" | bc))"
echo "Seed:         $SEED"
echo "=========================================="

# Show GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=========================================="

# Activate conda environment
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp

# Set HuggingFace cache paths (CRITICAL - avoid home quota)
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false

# Navigate to project
cd $PROJECT_DIR

# Create output directory
mkdir -p $OUTPUT_DIR

# Verify data exists
if [ ! -f "data/processed/$TRAIN_FILE" ]; then
    echo "ERROR: data/processed/$TRAIN_FILE not found!"
    echo "Run preprocessing first:"
    echo "  python -m src.preprocessing.prepare_data_merged \\"
    echo "      --original data/silver_labels/teacher_output.jsonl \\"
    echo "      --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \\"
    echo "      --output-dir data/processed --model $MODEL_NAME"
    exit 1
fi

if [ ! -f "data/processed/val.jsonl" ]; then
    echo "ERROR: data/processed/val.jsonl not found!"
    exit 1
fi

echo ""
echo "Data files:"
wc -l data/processed/$TRAIN_FILE data/processed/val.jsonl
echo ""

# Run training
python -m src.models.train_joint \
    --model "$MODEL_NAME" \
    --data-dir data/processed \
    --train-file "$TRAIN_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --class-weights-file data/processed/stats_merged.json

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "TRAINING COMPLETED SUCCESSFULLY"
    echo "Model saved to: $OUTPUT_DIR/best_model.pt"
    echo "=========================================="
    
    # Show result
    if [ -f "$OUTPUT_DIR/training_summary.json" ]; then
        echo "Training summary:"
        cat "$OUTPUT_DIR/training_summary.json"
    fi
else
    echo ""
    echo "=========================================="
    echo "TRAINING FAILED (exit code: $EXIT_CODE)"
    echo "Check error log for details"
    echo "=========================================="
    exit $EXIT_CODE
fi

echo ""
echo "End Time: $(date)"
echo "=========================================="
