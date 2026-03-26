#!/bin/bash
# Preprocessing: A2 — downsample ratio 4:1
# Output: data/processed_ds4/

#SBATCH --job-name=prep_ratio_4
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/logs/slurm_output/preprocessing/%j_ratio_4.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/logs/slurm_output/preprocessing/%j_ratio_4.err

# --- Environment ---
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
cd $PROJECT_DIR

echo "PREPROCESSING: A2 — downsample ratio 4:1"
echo "Output: data/processed_ds4"

[ ! -f "data/silver_labels/teacher_output.jsonl" ] && echo "ERROR: teacher_output.jsonl missing!" && exit 1

mkdir -p data/processed_ds4

python -m src.preprocessing.prepare_data_merged \
    --original data/silver_labels/teacher_output.jsonl \
    --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \
    --output-dir data/processed_ds4 \
    --model dicta-il/dictabert \
    --downsample-ratio 4.0

[ $? -eq 0 ] && echo "DONE:" && wc -l data/processed_ds4/*.jsonl || (echo "FAILED"; exit 1)
echo "End: $(date)"
