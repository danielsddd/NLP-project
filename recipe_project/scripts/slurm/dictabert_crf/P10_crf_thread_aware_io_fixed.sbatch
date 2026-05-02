#!/bin/bash
#SBATCH --job-name=P10_fix_IO
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_P10_fix.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_P10_fix.err

USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJ="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
cd $PROJ

DATA_DIR="data/processed_thread_aware_io"
mkdir -p models/checkpoints/dictabert_crf/P10_crf_thread_aware_io_fixed

python -m src.models.train_joint \
    --model "dicta-il/dictabert" \
    --data-dir $DATA_DIR \
    --train-file train_merged.jsonl \
    --output-dir models/checkpoints/dictabert_crf/P10_crf_thread_aware_io_fixed \
    --num-labels 5 \
    --epochs 10 \
    --batch-size 16 \
    --lr 2e-5 \
    --seed 42 \
    --class-weights-file $DATA_DIR/stats_merged.json
