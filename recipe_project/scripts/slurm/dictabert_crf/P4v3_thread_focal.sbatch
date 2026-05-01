#!/bin/bash
#SBATCH --job-name=dict_P4v3_focal
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_P4v3_thread_focal.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_P4v3_thread_focal.err

USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
cd $PROJECT_DIR

mkdir -p models/checkpoints/dictabert_crf/P4v3_thread_focal

echo "=================================================="
echo "P4v3: DictaBERT+CRF + BIO + Threads + Focal + FIXED weights"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=================================================="

python -m src.models.train_joint \
    --model dicta-il/dictabert \
    --data-dir data/processed_thread_aware \
    --train-file train_merged.jsonl \
    --output-dir models/checkpoints/dictabert_crf/P4v3_thread_focal \
    --epochs 15 \
    --batch-size 16 \
    --lr 2e-5 \
    --patience 5 \
    --seed 42 \
    --focal-gamma 2.0 \
    --class-weights-file data/processed_thread_aware/stats_merged.json

echo "DONE. End: $(date)"
