#!/bin/bash
# =============================================================================
# Model: xlmr | Variant: A6_no_enriched
# =============================================================================

#SBATCH --job-name=xlm__A6_no_enri
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_3090|a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/xlmr/%j_A6_no_enriched.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/xlmr/%j_A6_no_enriched.err

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
echo "MODEL:   xlmr"
echo "VARIANT: A6_no_enriched"
echo "HF:      xlm-roberta-base"
echo "OUTPUT:  models/checkpoints/xlmr/A6_no_enriched"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
echo "=================================================="

# --- Data Check ---
if [ ! -f "data/processed_no_enriched/train_merged.jsonl" ]; then
    echo "ERROR: data/processed_no_enriched/train_merged.jsonl not found!"
    echo "Run the corresponding preprocessing script first."
    exit 1
fi
echo "Data:"; wc -l data/processed_no_enriched/train_merged.jsonl data/processed_no_enriched/val.jsonl
echo ""

# --- A6: Without enriched data (original only, with weights+focal) ---
mkdir -p models/checkpoints/xlmr/A6_no_enriched

python -m src.models.train_student \
    --model "xlm-roberta-base" \
    --data-dir data/processed_no_enriched \
    --output-dir models/checkpoints/xlmr/A6_no_enriched \
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
    echo "SUCCESS — saved to: models/checkpoints/xlmr/A6_no_enriched"
    echo "=================================================="
    for f in models/checkpoints/xlmr/A6_no_enriched/training_summary.json models/checkpoints/xlmr/A6_no_enriched/all_results.json; do
        [ -f "$f" ] && echo "--- $f ---" && cat "$f" && echo ""
    done
else
    echo "FAILED (exit code: $EXIT_CODE)"; exit $EXIT_CODE
fi
echo "End: $(date)"; echo "=================================================="
