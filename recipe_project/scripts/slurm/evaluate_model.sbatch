#!/bin/bash
# =============================================================================
# SLURM Script: Model Evaluation
# =============================================================================
# Evaluate trained models on test/gold set
# Submit with: sbatch evaluate_model.sbatch
# =============================================================================

#SBATCH --job-name=evaluate_model
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_eval.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_eval.err

# =============================================================================
# CONFIGURATION
# =============================================================================
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction"

# Models to evaluate (can be modified)
ALEPHBERT_MODEL="models/checkpoints/student/best_model"
MBERT_MODEL="models/baselines/mbert/best_model"

echo "=========================================="
echo "MODEL EVALUATION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "=========================================="

# Activate conda environment
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp

# Set HuggingFace cache paths
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false

# Navigate to project
cd $PROJECT_DIR

# Create results directory
mkdir -p results

echo ""
echo "=========================================="
echo "1. Evaluating AlephBERT Student Model"
echo "=========================================="

if [ -d "$ALEPHBERT_MODEL" ]; then
    python -m src.evaluation.evaluate \
        --model-path "$ALEPHBERT_MODEL" \
        --test-file data/processed/dataset_test.jsonl \
        --output-dir results/alephbert \
        --bootstrap
else
    echo "AlephBERT model not found at $ALEPHBERT_MODEL"
    echo "Run train_alephbert.sbatch first"
fi

echo ""
echo "=========================================="
echo "2. Evaluating mBERT Baseline"
echo "=========================================="

if [ -d "$MBERT_MODEL" ]; then
    python -m src.evaluation.evaluate \
        --model-path "$MBERT_MODEL" \
        --test-file data/processed/dataset_test.jsonl \
        --output-dir results/mbert \
        --bootstrap
else
    echo "mBERT model not found at $MBERT_MODEL"
    echo "Run train_mbert.sbatch first"
fi

echo ""
echo "=========================================="
echo "3. Running Simple Baselines"
echo "=========================================="

python -m src.baselines.run_baselines \
    --data-dir data/processed

echo ""
echo "=========================================="
echo "EVALUATION COMPLETE"
echo "Results saved to: results/"
echo "=========================================="
echo "End Time: $(date)"
