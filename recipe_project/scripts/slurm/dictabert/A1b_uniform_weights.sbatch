#!/bin/bash
# =============================================================================
# Model: dictabert | Variant: A1b_uniform_weights
# =============================================================================

#SBATCH --job-name=dict_A1b_unifor
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert/%j_A1b_uniform_weights.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert/%j_A1b_uniform_weights.err

# --- Environment ---
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
cd $PROJECT_DIR

echo "=================================================="
echo "MODEL:   dictabert"
echo "VARIANT: A1b_uniform_weights"
echo "HF:      dicta-il/dictabert"
echo "OUTPUT:  models/checkpoints/dictabert/A1b_uniform_weights"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
echo "=================================================="

# --- Data Check ---
if [ ! -f "data/processed_v2/train_merged.jsonl" ]; then
    echo "ERROR: data/processed_v2/train_merged.jsonl not found!"
    echo "Run the corresponding preprocessing script first."
    exit 1
fi
echo "Data:"; wc -l data/processed_v2/train_merged.jsonl data/processed_v2/val.jsonl
echo ""

# --- A1b: Uniform weights (sanity check) ---
mkdir -p models/checkpoints/dictabert/A1b_uniform_weights

python -m src.models.train_student \
    --model "dicta-il/dictabert" \
    --data-dir data/processed_v2 \
    --output-dir models/checkpoints/dictabert/A1b_uniform_weights \
    --epochs 10 \
    --batch-size 16 \
    --lr 2e-5 \
    --seed 42 \
    --class-weights balanced \
    --weight-type uniform

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""; echo "=================================================="
    echo "SUCCESS — saved to: models/checkpoints/dictabert/A1b_uniform_weights"
    echo "=================================================="
    for f in models/checkpoints/dictabert/A1b_uniform_weights/training_summary.json models/checkpoints/dictabert/A1b_uniform_weights/all_results.json; do
        [ -f "$f" ] && echo "--- $f ---" && cat "$f" && echo ""
    done
else
    echo "FAILED (exit code: $EXIT_CODE)"; exit $EXIT_CODE
fi
echo "End: $(date)"; echo "=================================================="
