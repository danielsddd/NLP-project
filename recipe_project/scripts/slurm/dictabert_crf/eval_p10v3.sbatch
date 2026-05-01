#!/bin/bash
#SBATCH --job-name=eval_P10v3
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_eval_P10v3.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_eval_P10v3.err

USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
cd $PROJECT_DIR

echo "=================================================="
echo "EVAL P10v3 (fixed weights, IO scheme)"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=================================================="

CKPT="models/checkpoints/dictabert_crf/P10v3_fixed_weights"
RESULT="results/dictabert_crf/P10v3_fixed_weights"
GOLD_IO="data/gold_validation/gold_tokenized_dictabert_io.jsonl"
SILVER="data/processed_thread_aware_io/test.jsonl"

# Gold eval
echo "--- Gold eval ---"
mkdir -p "$RESULT/gold"
python -m scripts.local.evaluate_crf \
    --ckpt-dir   "$CKPT" \
    --test-file  "$GOLD_IO" \
    --output-dir "$RESULT/gold" \
    --model-name "dicta-il/dictabert" 2>&1 | tee "$RESULT/gold_eval.log"

# Silver eval
echo "--- Silver eval ---"
mkdir -p "$RESULT/silver"
python -m scripts.local.evaluate_crf \
    --ckpt-dir   "$CKPT" \
    --test-file  "$SILVER" \
    --output-dir "$RESULT/silver" \
    --model-name "dicta-il/dictabert" 2>&1 | tee "$RESULT/silver_eval.log"

echo ""
echo "=================================================="
echo "DONE. End: $(date)"
echo "=================================================="
