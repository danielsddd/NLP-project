#!/bin/bash
#SBATCH --job-name=dictcrf_A5_50
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_A5_crf_data_50pct.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_A5_crf_data_50pct.err

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
echo "MODEL:   dictabert_crf"
echo "VARIANT: A5_crf_data_50pct"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true

if [ ! -f "data/processed_v2/train_merged.jsonl" ]; then
    echo "ERROR: data/processed_v2/train_merged.jsonl not found!"
    exit 1
fi

TOTAL=$(wc -l < data/processed_v2/train_merged.jsonl)
MAX_EXAMPLES=$((TOTAL * 50 / 100))
echo "Using $MAX_EXAMPLES / $TOTAL training examples (50%)"

TEMP_TRAIN="data/processed_v2/train_50pct.jsonl"
head -n $MAX_EXAMPLES data/processed_v2/train_merged.jsonl > "$TEMP_TRAIN"

mkdir -p models/checkpoints/dictabert_crf/A5_crf_data_50pct

python -m src.models.train_joint     --model "dicta-il/dictabert"     --data-dir data/processed_v2     --train-file train_50pct.jsonl     --output-dir models/checkpoints/dictabert_crf/A5_crf_data_50pct     --epochs 10     --batch-size 16     --lr 2e-5     --seed 42     --class-weights-file data/processed_v2/stats_merged.json

EXIT_CODE=$?
rm -f "$TEMP_TRAIN"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS — saved to: models/checkpoints/dictabert_crf/A5_crf_data_50pct"
else
    echo "FAILED (exit code: $EXIT_CODE)"
fi
echo "End: $(date)"
exit $EXIT_CODE
