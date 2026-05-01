#!/bin/bash
#SBATCH --job-name=dict_P10v3_fixed
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_P10v3_fixed.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_P10v3_fixed.err

USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
cd $PROJECT_DIR

mkdir -p models/checkpoints/dictabert_crf/P10v3_fixed_weights

echo "=================================================="
echo "P10v3: DictaBERT+CRF + IO + Threads + FIXED class weights"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=================================================="

python -m src.models.train_joint \
    --model dicta-il/dictabert \
    --data-dir data/processed_thread_aware_io \
    --train-file train_merged.jsonl \
    --output-dir models/checkpoints/dictabert_crf/P10v3_fixed_weights \
    --epochs 15 \
    --batch-size 16 \
    --lr 2e-5 \
    --patience 5 \
    --seed 42 \
    --class-weights-file data/processed_thread_aware_io/stats_merged.json

echo "=================================================="
echo "DONE. End: $(date)"
echo "=================================================="
