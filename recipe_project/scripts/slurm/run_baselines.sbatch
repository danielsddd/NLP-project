#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/%j_baselines.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/%j_baselines.err

USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
cd $PROJECT_DIR

echo "=================================================="
echo "BASELINES: majority / random / keyword"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
echo "=================================================="

python -m src.baselines.run_baselines \
    --data-dir data/processed_v2 \
    --output-dir results/baselines \
    --split test

echo "DONE. End: $(date)"
