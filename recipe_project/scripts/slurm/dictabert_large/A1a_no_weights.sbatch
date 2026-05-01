#!/bin/bash
#SBATCH --job-name=dlarge_A1a
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_large/%j_A1a_no_weights.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_large/%j_A1a_no_weights.err

USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
cd $PROJECT_DIR

mkdir -p results/dictabert_large/A1a_no_weights

echo "DictaBERT-Large A1a (no weights) | Job: $SLURM_JOB_ID | Start: $(date)"

python -m src.models.train_student \
    --model dicta-il/dictabert-large \
    --data-dir data/processed_v2 \
    --train-file train_merged.jsonl \
    --output-dir results/dictabert_large/A1a_no_weights \
    --epochs 10 \
    --batch-size 8 \
    --lr 2e-5 \
    --patience 3 \
    --seed 42

echo "DONE. End: $(date)"
