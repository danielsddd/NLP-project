#!/bin/bash
# =============================================================================
# Model: dictabert_large | Variant: A1b_uniform_weights
# =============================================================================

#SBATCH --job-name=dict_A1b_unifor
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_large/%j_A1b_uniform_weights.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_large/%j_A1b_uniform_weights.err

# --- Environment ---
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="simanovsky2_DIR/recipe_modification_extraction/recipe_project"
source simanovsky2_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=simanovsky2_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
cd $PROJECT_DIR

echo "=================================================="
echo "MODEL:   dictabert_large"
echo "VARIANT: A1b_uniform_weights"
echo "HF:      dicta-il/dictabert-large"
echo "OUTPUT:  models/checkpoints/dictabert_large/A1b_uniform_weights"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
echo "=================================================="

# --- Data Check ---
if [ ! -f "data/processed/train_merged.jsonl" ]; then
    echo "ERROR: data/processed/train_merged.jsonl not found!"
    echo "Run the corresponding preprocessing script first."
    exit 1
fi
echo "Data:"; wc -l data/processed/train_merged.jsonl data/processed/val.jsonl
echo ""

# --- A1b: Uniform weights (sanity check) ---
mkdir -p models/checkpoints/dictabert_large/A1b_uniform_weights

python -m src.models.train_student \
    --model "dicta-il/dictabert-large" \
    --data-dir data/processed \
    --output-dir models/checkpoints/dictabert_large/A1b_uniform_weights \
    --epochs 10 \
    --batch-size 8 \
    --lr 2e-5 \
    --seed 42 \
    --fp16 \
    --class-weights balanced \
    --weight-type uniform

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""; echo "=================================================="
    echo "SUCCESS — saved to: models/checkpoints/dictabert_large/A1b_uniform_weights"
    echo "=================================================="
    for f in models/checkpoints/dictabert_large/A1b_uniform_weights/training_summary.json models/checkpoints/dictabert_large/A1b_uniform_weights/all_results.json; do
        [ -f "$f" ] && echo "--- $f ---" && cat "$f" && echo ""
    done
else
    echo "FAILED (exit code: $EXIT_CODE)"; exit $EXIT_CODE
fi
echo "End: $(date)"; echo "=================================================="
