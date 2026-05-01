#!/bin/bash
# =============================================================================
# Model: dictabert | Variant: A1c_weights_only
# =============================================================================

#SBATCH --job-name=dict_A1c_weight
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert/%j_A1c_weights_only.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert/%j_A1c_weights_only.err

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
echo "VARIANT: A1c_weights_only"
echo "HF:      dicta-il/dictabert"
echo "OUTPUT:  models/checkpoints/dictabert/A1c_weights_only"
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

# --- A1c: Inverse-freq weights, NO focal ---
mkdir -p models/checkpoints/dictabert/A1c_weights_only

python -m src.models.train_student \
    --model "dicta-il/dictabert" \
    --data-dir data/processed_v2 \
    --output-dir models/checkpoints/dictabert/A1c_weights_only \
    --epochs 10 \
    --batch-size 16 \
    --lr 2e-5 \
    --seed 42 \
    --class-weights balanced

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""; echo "=================================================="
    echo "SUCCESS — saved to: models/checkpoints/dictabert/A1c_weights_only"
    echo "=================================================="
    for f in models/checkpoints/dictabert/A1c_weights_only/training_summary.json models/checkpoints/dictabert/A1c_weights_only/all_results.json; do
        [ -f "$f" ] && echo "--- $f ---" && cat "$f" && echo ""
    done
else
    echo "FAILED (exit code: $EXIT_CODE)"; exit $EXIT_CODE
fi
echo "End: $(date)"; echo "=================================================="
