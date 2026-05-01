#!/bin/bash
#SBATCH --job-name=dlrg_P10v4_foc
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_large_crf/%j_P10v4_large_focal.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_large_crf/%j_P10v4_large_focal.err

USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
cd $PROJECT_DIR

mkdir -p models/checkpoints/dictabert_large_crf/P10v4_large_focal

echo "=================================================="
echo "DictaBERT-Large+CRF + IO + Threads + Weights + Focal"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=================================================="

python -m src.models.train_joint \
    --model dicta-il/dictabert-large \
    --data-dir data/processed_thread_aware_io \
    --train-file train_merged.jsonl \
    --output-dir models/checkpoints/dictabert_large_crf/P10v4_large_focal \
    --epochs 15 \
    --batch-size 8 \
    --lr 1e-5 \
    --patience 5 \
    --seed 42 \
    --focal-gamma 2.0 \
    --class-weights-file data/processed_thread_aware_io/stats_merged.json

echo "DONE. End: $(date)"
