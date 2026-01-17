#!/bin/bash
# =============================================================================
# SLURM Script: Data Size Ablation Study
# =============================================================================
# Runs 6 parallel experiments with different training data sizes
# Uses SLURM job arrays for efficient parallel execution
# Submit with: sbatch ablation_data_size.sbatch
# =============================================================================

#SBATCH --job-name=ablation_size
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_2080_ti|geforce_rtx_3090|a5000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-6
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/ablation_size_%A_%a.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/ablation_size_%A_%a.err

# =============================================================================
# CONFIGURATION
# =============================================================================
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction"

# Data sizes for ablation (array index 1-6)
DATA_SIZES=(0 500 1000 1500 2000 2500 3000)  # Index 0 unused
DATA_SIZE=${DATA_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "ABLATION STUDY: Data Size"
echo "=========================================="
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Data Size: $DATA_SIZE examples"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=name --format=csv,noheader
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

# Create output directory for this ablation
OUTPUT_DIR="models/ablations/data_size_${DATA_SIZE}"
mkdir -p $OUTPUT_DIR

# Run training with limited data
python -m src.models.train_student \
    --model "onlplab/alephbert-base" \
    --data-dir data/processed \
    --output-dir $OUTPUT_DIR \
    --epochs 5 \
    --batch-size 16 \
    --lr 2e-5 \
    --max-examples $DATA_SIZE \
    --seed 42 \
    --fp16

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "ABLATION COMPLETE: Data Size $DATA_SIZE"
    echo "Model saved to: $OUTPUT_DIR"
    echo "=========================================="
    
    # Run evaluation
    echo "Running evaluation..."
    python -m src.evaluation.evaluate \
        --model-path $OUTPUT_DIR/best_model \
        --test-file data/processed/dataset_test.jsonl \
        --output-dir results/ablations/data_size_${DATA_SIZE}
else
    echo "=========================================="
    echo "ABLATION FAILED: Data Size $DATA_SIZE"
    echo "=========================================="
    exit 1
fi

echo "End Time: $(date)"
