#!/bin/bash
# =============================================================================
# Model: hebert | Variant: P3_add_downsample_4
# =============================================================================

#SBATCH --job-name=hebert_P3_add_d
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_3090|a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/logs/slurm_output/hebert/%j_P3_add_downsample_4.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/logs/slurm_output/hebert/%j_P3_add_downsample_4.err

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
echo "MODEL:   hebert"
echo "VARIANT: P3_add_downsample_4"
echo "HF:      avichr/heBERT"
echo "OUTPUT:  models/checkpoints/hebert/P3_add_downsample_4"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
echo "=================================================="

# --- Data Check ---
if [ ! -f "data/processed_ds4/train_merged.jsonl" ]; then
    echo "ERROR: data/processed_ds4/train_merged.jsonl not found!"
    echo "Run the corresponding preprocessing script first."
    exit 1
fi
echo "Data:"; wc -l data/processed_ds4/train_merged.jsonl data/processed_ds4/val.jsonl
echo ""

# --- P3b: + Weights + Focal + Downsample ratio 4:1 ---
mkdir -p models/checkpoints/hebert/P3_add_downsample_4

python -m src.models.train_student \
    --model "avichr/heBERT" \
    --data-dir data/processed_ds4 \
    --output-dir models/checkpoints/hebert/P3_add_downsample_4 \
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
    echo "SUCCESS — saved to: models/checkpoints/hebert/P3_add_downsample_4"
    echo "=================================================="
    for f in models/checkpoints/hebert/P3_add_downsample_4/training_summary.json models/checkpoints/hebert/P3_add_downsample_4/all_results.json; do
        [ -f "$f" ] && echo "--- $f ---" && cat "$f" && echo ""
    done
else
    echo "FAILED (exit code: $EXIT_CODE)"; exit $EXIT_CODE
fi
echo "End: $(date)"; echo "=================================================="
