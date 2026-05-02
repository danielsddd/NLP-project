#!/bin/bash
#SBATCH --job-name=large_P10_IO
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_large_P10_IO.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_large_P10_IO.err

USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJ="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
cd $PROJ

DATA_DIR="data/processed_thread_aware_io_large"
mkdir -p models/checkpoints/dictabert_large_crf/P10_large_crf_thread_io

python -m src.models.train_joint \
    --model "dicta-il/dictabert-large" \
    --data-dir $DATA_DIR \
    --train-file train_merged.jsonl \
    --output-dir models/checkpoints/dictabert_large_crf/P10_large_crf_thread_io \
    --epochs 10 \
    --batch-size 8 \
    --lr 2e-5 \
    --seed 42 \
    --class-weights-file $DATA_DIR/stats_merged.json
