#!/bin/bash
# =============================================================================
# Model: mbert | Variant: P4_add_thread_aware
# =============================================================================

#SBATCH --job-name=mber_P4_add_thr
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_3090|a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/mbert/%j_P4_add_thread_aware.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/mbert/%j_P4_add_thread_aware.err

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
echo "MODEL:   mbert"
echo "VARIANT: P4_add_thread_aware"
echo "HF:      bert-base-multilingual-cased"
echo "OUTPUT:  models/checkpoints/mbert/P4_add_thread_aware"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
echo "=================================================="

# --- Data Check ---
if [ ! -f "data/processed_thread_aware/train_merged.jsonl" ]; then
    echo "ERROR: data/processed_thread_aware/train_merged.jsonl not found!"
    echo "Run the corresponding preprocessing script first."
    exit 1
fi
echo "Data:"; wc -l data/processed_thread_aware/train_merged.jsonl data/processed_thread_aware/val.jsonl
echo ""

# --- P4: + Weights + Focal + Thread-aware context ---
mkdir -p models/checkpoints/mbert/P4_add_thread_aware

python -m src.models.train_student \
    --model "bert-base-multilingual-cased" \
    --data-dir data/processed_thread_aware \
    --output-dir models/checkpoints/mbert/P4_add_thread_aware \
    --epochs 10 \
    --batch-size 16 \
    --lr 2e-5 \
    --seed 42 \
    --fp16 \
    --class-weights balanced \
    --focal-loss \
    --focal-gamma 2.0

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""; echo "=================================================="
    echo "SUCCESS — saved to: models/checkpoints/mbert/P4_add_thread_aware"
    echo "=================================================="
    for f in models/checkpoints/mbert/P4_add_thread_aware/training_summary.json models/checkpoints/mbert/P4_add_thread_aware/all_results.json; do
        [ -f "$f" ] && echo "--- $f ---" && cat "$f" && echo ""
    done
else
    echo "FAILED (exit code: $EXIT_CODE)"; exit $EXIT_CODE
fi
echo "End: $(date)"; echo "=================================================="
