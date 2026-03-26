#!/bin/bash
# =============================================================================
# Model: dictabert_large | Variant: combo_thread_aware_io
# =============================================================================

#SBATCH --job-name=dict_combo_thre
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_large/%j_combo_thread_aware_io.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_large/%j_combo_thread_aware_io.err

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
echo "VARIANT: combo_thread_aware_io"
echo "HF:      dicta-il/dictabert-large"
echo "OUTPUT:  models/checkpoints/dictabert_large/combo_thread_aware_io"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
echo "=================================================="

# --- Data Check ---
if [ ! -f "data/processed_thread_aware_io/train_merged.jsonl" ]; then
    echo "ERROR: data/processed_thread_aware_io/train_merged.jsonl not found!"
    echo "Run the corresponding preprocessing script first."
    exit 1
fi
echo "Data:"; wc -l data/processed_thread_aware_io/train_merged.jsonl data/processed_thread_aware_io/val.jsonl
echo ""

# --- COMBO: Thread-aware + IO scheme + weights + focal ---
mkdir -p models/checkpoints/dictabert_large/combo_thread_aware_io

python -m src.models.train_student \
    --model "dicta-il/dictabert-large" \
    --data-dir data/processed_thread_aware_io \
    --output-dir models/checkpoints/dictabert_large/combo_thread_aware_io \
    --epochs 10 \
    --batch-size 8 \
    --lr 2e-5 \
    --seed 42 \
    --fp16 \
    --class-weights balanced \
    --focal-loss \
    --focal-gamma 2.0

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""; echo "=================================================="
    echo "SUCCESS — saved to: models/checkpoints/dictabert_large/combo_thread_aware_io"
    echo "=================================================="
    for f in models/checkpoints/dictabert_large/combo_thread_aware_io/training_summary.json models/checkpoints/dictabert_large/combo_thread_aware_io/all_results.json; do
        [ -f "$f" ] && echo "--- $f ---" && cat "$f" && echo ""
    done
else
    echo "FAILED (exit code: $EXIT_CODE)"; exit $EXIT_CODE
fi
echo "End: $(date)"; echo "=================================================="
