#!/bin/bash
# =============================================================================
# SLURM Script: Train AlephBERT Student Model
# =============================================================================
# Main training script for the student model
# Submit with: sbatch train_alephbert.sbatch
# =============================================================================

#SBATCH --job-name=alephbert_train
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_2080_ti|geforce_rtx_3090|a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_alephbert.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_alephbert.err

# =============================================================================
# CONFIGURATION
# =============================================================================
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction"

# Training hyperparameters
MODEL_NAME="onlplab/alephbert-base"
EPOCHS=5
BATCH_SIZE=16
LEARNING_RATE=2e-5
WARMUP_RATIO=0.1
SEED=42

echo "=========================================="
echo "ALEPHBERT TRAINING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Model: $MODEL_NAME"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
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
mkdir -p models/checkpoints/student

# Run training
python -m src.models.train_student \
    --model "$MODEL_NAME" \
    --data-dir data/processed \
    --output-dir models/checkpoints/student \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --warmup $WARMUP_RATIO \
    --seed $SEED \
    --fp16

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "TRAINING COMPLETED SUCCESSFULLY"
    echo "Model saved to: models/checkpoints/student/best_model"
    echo "=========================================="
else
    echo "=========================================="
    echo "TRAINING FAILED"
    echo "Check the error log for details"
    echo "=========================================="
    exit 1
fi

echo "End Time: $(date)"
echo "=========================================="
