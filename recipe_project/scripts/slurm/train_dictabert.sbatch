#!/bin/bash
#SBATCH --job-name=dictabert
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_2080|geforce_rtx_3090|a5000|a6000|l40s"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/recipe_modification_extraction/logs/slurm_outputs/%j_dictabert.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/recipe_modification_extraction/logs/slurm_outputs/%j_dictabert.err

PROJ=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2
source $PROJ/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$PROJ/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd $PROJ/recipe_modification_extraction

MODEL="dicta-il/dictabert"
DATA_OUT="data/processed_dictabert"

echo "=== DictaBERT Training ==="
echo "Node: $(hostname), Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Step 1: Preprocess with DictaBERT tokenizer
echo "Preprocessing with DictaBERT tokenizer..."
python -m src.preprocessing.prepare_data \
    --input data/silver_labels/teacher_output.jsonl \
    --output-dir $DATA_OUT \
    --model "$MODEL" \
    --scheme BIO

# Step 2: Train
echo "Training DictaBERT..."
python -m src.models.train_student \
    --model "$MODEL" \
    --data-dir $DATA_OUT \
    --output-dir models/checkpoints/dictabert \
    --epochs 10 --batch-size 16 --lr 2e-5 --warmup 0.1 --seed 42 --fp16

echo "=== DONE: $(date) ==="
