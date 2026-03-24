#!/bin/bash
#SBATCH --job-name=alephbert_v3
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_2080|geforce_rtx_3090|a5000|a6000|l40s"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/recipe_modification_extraction/logs/slurm_outputs/%j_alephbert_v3.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/recipe_modification_extraction/logs/slurm_outputs/%j_alephbert_v3.err

PROJ=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2
source $PROJ/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$PROJ/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd $PROJ/recipe_modification_extraction

echo "=== AlephBERT v3: lr=2e-5, 15 epochs, early stopping patience=3 ==="
echo "Node: $(hostname), Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m src.models.train_student \
    --model onlplab/alephbert-base \
    --data-dir data/processed \
    --output-dir models/checkpoints/student_v3 \
    --epochs 15 --batch-size 16 --lr 2e-5 --warmup 0.1 --seed 42 --fp16

echo "=== DONE: $(date) ==="
