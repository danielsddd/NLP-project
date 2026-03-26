#!/bin/bash
# Preprocessing: A2 — downsample ratio 2:1
# Output: data/processed_ds2/

#SBATCH --job-name=prep_ratio_2
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/preprocessing/%j_ratio_2.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/preprocessing/%j_ratio_2.err

# --- Environment ---
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="simanovsky2_DIR/recipe_modification_extraction/recipe_project"
source simanovsky2_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=simanovsky2_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
cd $PROJECT_DIR

echo "PREPROCESSING: A2 — downsample ratio 2:1"
echo "Output: data/processed_ds2"

[ ! -f "data/silver_labels/teacher_output.jsonl" ] && echo "ERROR: teacher_output.jsonl missing!" && exit 1

mkdir -p data/processed_ds2

python -m src.preprocessing.prepare_data_merged \
    --original data/silver_labels/teacher_output.jsonl \
    --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \
    --output-dir data/processed_ds2 \
    --model dicta-il/dictabert \
    --downsample-ratio 2.0

[ $? -eq 0 ] && echo "DONE:" && wc -l data/processed_ds2/*.jsonl || (echo "FAILED"; exit 1)
echo "End: $(date)"
