#!/bin/bash
# =============================================================================
# SLURM Script: Data Preprocessing
# =============================================================================
# Convert silver labels to BIO format for training
# This is CPU-intensive (tokenization), not GPU-intensive
# Submit with: sbatch preprocess_data.sbatch
# =============================================================================

#SBATCH --job-name=preprocess
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_preprocess.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_preprocess.err

# =============================================================================
# CONFIGURATION
# =============================================================================
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction"

echo "=========================================="
echo "DATA PREPROCESSING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start Time: $(date)"
echo "=========================================="

# Activate conda environment
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp

# Set HuggingFace cache paths
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=true  # Enable for CPU parallelism

# Navigate to project
cd $PROJECT_DIR

# Check if silver labels exist
if [ ! -f "data/silver_labels/teacher_output.jsonl" ]; then
    echo "ERROR: Silver labels not found!"
    echo "Run teacher labeling first (locally with API key)"
    exit 1
fi

echo "Converting silver labels to BIO format..."

# Run preprocessing
python -m src.preprocessing.prepare_data \
    --input data/silver_labels/teacher_output.jsonl \
    --output-dir data/processed \
    --model onlplab/alephbert-base \
    --scheme BIO \
    --max-length 128 \
    --val-split 0.1 \
    --test-split 0.1

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "PREPROCESSING COMPLETE"
    echo "=========================================="
    echo "Output files:"
    ls -lh data/processed/
    echo ""
    echo "Dataset statistics:"
    wc -l data/processed/*.jsonl
else
    echo "=========================================="
    echo "PREPROCESSING FAILED"
    echo "=========================================="
    exit 1
fi

echo ""
echo "End Time: $(date)"
