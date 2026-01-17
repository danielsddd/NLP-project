#!/bin/bash
# =============================================================================
# SLURM Training Script for AlephBERT
# =============================================================================
# Use this when you have SLURM cluster access
# Submit with: sbatch train_student.sbatch
# =============================================================================

#SBATCH --job-name=alephbert_train
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_2080_ti|geforce_rtx_3090|a5000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j.err

# =============================================================================
# SETUP - Update these paths when you have cluster access
# =============================================================================
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction"

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

# Activate conda environment
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp

# Set HuggingFace cache paths (CRITICAL - avoid home directory quota)
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false

# Navigate to project
cd $PROJECT_DIR

# Run training
python -m src.models.train_student \
    --data-dir data/processed \
    --output-dir models/checkpoints/student \
    --epochs 5 \
    --batch-size 16 \
    --lr 2e-5 \
    --fp16

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
