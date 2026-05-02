#!/bin/bash
#SBATCH --job-name=dict_noCRF
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_dictabert_no_crf_io_thread.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_dictabert_no_crf_io_thread.err

USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"

source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp

export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false

cd $PROJECT_DIR
mkdir -p models/checkpoints/dictabert/P10_io_thread_no_crf

echo "=================================================="
echo "dict_noCRF"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=================================================="

python -m src.models.train_student \
    --model dicta-il/dictabert \
    --data-dir data/processed_thread_aware_io \
    --output-dir models/checkpoints/dictabert/P10_io_thread_no_crf \
    --epochs 15 \
    --batch-size 16 \
    --lr 2e-5 \
    --warmup 0.1 \
    --weight-decay 0.01 \
    --patience 5 \
    --seed 42 \
    --class-weights sqrt

echo "=================================================="
echo "DONE. End: $(date)"
echo "=================================================="
