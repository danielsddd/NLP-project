#!/bin/bash
#SBATCH --job-name=dlrg_P4_bio
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_large_crf/%j_P4_large_thread_bio.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_large_crf/%j_P4_large_thread_bio.err

USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
cd $PROJECT_DIR

mkdir -p models/checkpoints/dictabert_large_crf/P4_large_crf_thread_bio

echo "DictaBERT-Large+CRF + BIO + Threads | Job: $SLURM_JOB_ID | Start: $(date)"

python -m src.models.train_joint \
    --model dicta-il/dictabert-large \
    --data-dir data/processed_thread_aware \
    --train-file train_merged.jsonl \
    --output-dir models/checkpoints/dictabert_large_crf/P4_large_crf_thread_bio \
    --epochs 15 \
    --batch-size 8 \
    --lr 1e-5 \
    --patience 5 \
    --seed 42 \
    --class-weights-file data/processed_thread_aware/stats_merged.json

echo "DONE. End: $(date)"
