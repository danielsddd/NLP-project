#!/bin/bash
# =============================================================================
# SLURM Script: Train mBERT Baseline
# =============================================================================
# Train multilingual BERT for baseline comparison
# This demonstrates that Hebrew-specific pretraining (AlephBERT) matters
# Submit with: sbatch train_mbert.sbatch
# =============================================================================

#SBATCH --job-name=mbert_baseline
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_2080_ti|geforce_rtx_3090|a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_mbert.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_mbert.err

# =============================================================================
# CONFIGURATION
# =============================================================================
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction"

# mBERT model
MODEL_NAME="bert-base-multilingual-cased"
EPOCHS=5
BATCH_SIZE=16
LEARNING_RATE=2e-5
SEED=42

echo "=========================================="
echo "mBERT BASELINE TRAINING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Model: $MODEL_NAME"
echo "Purpose: Baseline to compare with AlephBERT"
echo "=========================================="

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=========================================="

# Activate conda environment
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp

# Set HuggingFace cache paths
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false

# Navigate to project
cd $PROJECT_DIR

# Create output directory
mkdir -p models/baselines/mbert

# Run training with mBERT
python -m src.models.train_student \
    --model "$MODEL_NAME" \
    --data-dir data/processed \
    --output-dir models/baselines/mbert \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --seed $SEED \
    --fp16

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "mBERT TRAINING COMPLETED"
    echo "Model saved to: models/baselines/mbert/best_model"
    echo "=========================================="
else
    echo "=========================================="
    echo "mBERT TRAINING FAILED"
    echo "=========================================="
    exit 1
fi

echo "End Time: $(date)"
