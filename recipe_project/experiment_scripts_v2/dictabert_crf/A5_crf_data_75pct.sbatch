#!/bin/bash
# =============================================================================
# Model: dictabert_crf | Variant: A5_crf_data_75pct
# =============================================================================

#SBATCH --job-name=crf_A5_crf_dat
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_3090|a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/logs/slurm_output/dictabert_crf/%j_A5_crf_data_75pct.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/logs/slurm_output/dictabert_crf/%j_A5_crf_data_75pct.err

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
echo "MODEL:   dictabert_crf"
echo "VARIANT: A5_crf_data_75pct"
echo "HF:      dicta-il/dictabert"
echo "OUTPUT:  models/checkpoints/dictabert_crf/A5_crf_data_75pct"
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

# --- A5: CRF + 75% data ---
mkdir -p models/checkpoints/dictabert_crf/A5_crf_data_75pct

TOTAL=$(wc -l < data/processed/train_merged.jsonl)
MAX_EXAMPLES=$((TOTAL * 75 / 100))
echo "Using $MAX_EXAMPLES / $TOTAL training examples (75%)"
TEMP_TRAIN="models/checkpoints/dictabert_crf/A5_crf_data_75pct/train_75pct.jsonl"
head -n $MAX_EXAMPLES data/processed/train_merged.jsonl > "$TEMP_TRAIN"

python -m src.models.train_joint \
    --model "dicta-il/dictabert" \
    --data-dir data/processed \
    --train-file $TEMP_TRAIN \
    --output-dir models/checkpoints/dictabert_crf/A5_crf_data_75pct \
    --epochs 10 \
    --batch-size 16 \
    --lr 2e-5 \
    --seed 42 \
    --class-weights-file data/processed/stats_merged.json

rm -f "$TEMP_TRAIN"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""; echo "=================================================="
    echo "SUCCESS — saved to: models/checkpoints/dictabert_crf/A5_crf_data_75pct"
    echo "=================================================="
    for f in models/checkpoints/dictabert_crf/A5_crf_data_75pct/training_summary.json models/checkpoints/dictabert_crf/A5_crf_data_75pct/all_results.json; do
        [ -f "$f" ] && echo "--- $f ---" && cat "$f" && echo ""
    done
else
    echo "FAILED (exit code: $EXIT_CODE)"; exit $EXIT_CODE
fi
echo "End: $(date)"; echo "=================================================="
