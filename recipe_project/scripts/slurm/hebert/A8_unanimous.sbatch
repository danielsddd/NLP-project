#!/bin/bash
# =============================================================================
# Model: hebert | Variant: A8_unanimous
# =============================================================================

#SBATCH --job-name=hebe_A8_unanimo
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_3090|a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/hebert/%j_A8_unanimous.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/hebert/%j_A8_unanimous.err

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
echo "MODEL:   hebert"
echo "VARIANT: A8_unanimous"
echo "HF:      avichr/heBERT"
echo "OUTPUT:  models/checkpoints/hebert/A8_unanimous"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
echo "=================================================="

# --- Data Check ---
if [ ! -f "data/processed_unanimous/train_merged.jsonl" ]; then
    echo "ERROR: data/processed_unanimous/train_merged.jsonl not found!"
    echo "Run the corresponding preprocessing script first."
    exit 1
fi
echo "Data:"; wc -l data/processed_unanimous/train_merged.jsonl data/processed_unanimous/val.jsonl
echo ""

# --- A8: Only 3/3 unanimous threads (with weights+focal) ---
mkdir -p models/checkpoints/hebert/A8_unanimous

python -m src.models.train_student \
    --model "avichr/heBERT" \
    --data-dir data/processed_unanimous \
    --output-dir models/checkpoints/hebert/A8_unanimous \
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
    echo "SUCCESS — saved to: models/checkpoints/hebert/A8_unanimous"
    echo "=================================================="
    for f in models/checkpoints/hebert/A8_unanimous/training_summary.json models/checkpoints/hebert/A8_unanimous/all_results.json; do
        [ -f "$f" ] && echo "--- $f ---" && cat "$f" && echo ""
    done
else
    echo "FAILED (exit code: $EXIT_CODE)"; exit $EXIT_CODE
fi
echo "End: $(date)"; echo "=================================================="
