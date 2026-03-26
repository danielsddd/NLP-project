#!/bin/bash
# =============================================================================
# Model: dictabert | Variant: A5_data_50pct
# =============================================================================

#SBATCH --job-name=dictab_A5_data_
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_3090|a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/logs/slurm_output/dictabert/%j_A5_data_50pct.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/logs/slurm_output/dictabert/%j_A5_data_50pct.err

# --- Environment ---
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
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
echo "VARIANT: A5_data_50pct"
echo "HF:      dicta-il/dictabert"
echo "OUTPUT:  models/checkpoints/dictabert/A5_data_50pct"
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

# --- A5: 50% of training data (with weights+focal) ---
mkdir -p models/checkpoints/dictabert/A5_data_50pct

TOTAL=$(wc -l < data/processed/train_merged.jsonl)
MAX_EXAMPLES=$((TOTAL * 50 / 100))
echo "Using $MAX_EXAMPLES / $TOTAL training examples (50%)"

python -m src.models.train_student \
    --model "dicta-il/dictabert" \
    --data-dir data/processed \
    --output-dir models/checkpoints/dictabert/A5_data_50pct \
    --epochs 10 \
    --batch-size 16 \
    --lr 2e-5 \
    --seed 42 \
    --fp16 \
    --class-weights balanced \
    --focal-loss \
    --focal-gamma 2.0 \
    --max-examples $MAX_EXAMPLES

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""; echo "=================================================="
    echo "SUCCESS — saved to: models/checkpoints/dictabert/A5_data_50pct"
    echo "=================================================="
    for f in models/checkpoints/dictabert/A5_data_50pct/training_summary.json models/checkpoints/dictabert/A5_data_50pct/all_results.json; do
        [ -f "$f" ] && echo "--- $f ---" && cat "$f" && echo ""
    done
else
    echo "FAILED (exit code: $EXIT_CODE)"; exit $EXIT_CODE
fi
echo "End: $(date)"; echo "=================================================="
