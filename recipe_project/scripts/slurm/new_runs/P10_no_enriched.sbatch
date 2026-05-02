#!/bin/bash
#SBATCH --job-name=P10_noEnr
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_P10_no_enriched.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_P10_no_enriched.err

USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"

source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp

export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false

cd $PROJECT_DIR

# ── Data check ───────────────────────────────────────────────────────────────
# This dir contains ONLY original data (no enriched), thread-aware, IO scheme.
# Different from data/processed_thread_aware_io which has enriched mixed in.
DATA_DIR="data/processed_thread_aware_io_no_enriched"

if [ ! -f "$DATA_DIR/train_merged.jsonl" ]; then
    echo "ERROR: $DATA_DIR/train_merged.jsonl not found!"
    echo "Run preprocessing first:"
    echo "  python -m src.preprocessing.prepare_thread_aware \\"
    echo "      --original data/silver_labels/teacher_output_v2.jsonl \\"
    echo "      --output-dir $DATA_DIR \\"
    echo "      --model dicta-il/dictabert \\"
    echo "      --scheme IO \\"
    echo "      --downsample-ratio 3.0"
    echo "  (NO --enriched flag)"
    exit 1
fi

if [ ! -f "$DATA_DIR/id2label.json" ]; then
    echo "ERROR: $DATA_DIR/id2label.json not found!"
    echo "Re-run preprocessing with the patched prepare_thread_aware.py"
    exit 1
fi

echo "Data check OK:"
wc -l "$DATA_DIR/train_merged.jsonl" "$DATA_DIR/val.jsonl" "$DATA_DIR/test.jsonl"

mkdir -p models/checkpoints/dictabert_crf/P10_no_enriched
mkdir -p logs/slurm_output/dictabert_crf

echo "=================================================="
echo "MODEL:   DictaBERT+CRF"
echo "VARIANT: P10_no_enriched"
echo "HF:      dicta-il/dictabert"
echo "DATA:    $DATA_DIR  (original only, no enriched)"
echo "OUTPUT:  models/checkpoints/dictabert_crf/P10_no_enriched"
echo "Job ID:  $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=================================================="

python -m src.models.train_joint \
    --model dicta-il/dictabert \
    --data-dir "$DATA_DIR" \
    --train-file train_merged.jsonl \
    --output-dir models/checkpoints/dictabert_crf/P10_no_enriched \
    --epochs 15 \
    --batch-size 16 \
    --lr 2e-5 \
    --patience 5 \
    --seed 42

EXIT_CODE=$?
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS — saved to: models/checkpoints/dictabert_crf/P10_no_enriched"
    for f in models/checkpoints/dictabert_crf/P10_no_enriched/training_summary.json \
              models/checkpoints/dictabert_crf/P10_no_enriched/all_results.json; do
        [ -f "$f" ] && echo "--- $f ---" && cat "$f" && echo ""
    done
else
    echo "FAILED (exit code: $EXIT_CODE)"
    exit $EXIT_CODE
fi
echo "End: $(date)"
echo "=================================================="