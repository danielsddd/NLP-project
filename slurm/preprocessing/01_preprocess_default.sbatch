#!/bin/bash
# Preprocessing: Default — ratio 3.0, BIO, merged
# Output: data/processed/

#SBATCH --job-name=prep_default
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/preprocessing/%j_default.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/preprocessing/%j_default.err

# --- Environment ---
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
cd $PROJECT_DIR

echo "PREPROCESSING: Default — ratio 3.0, BIO, merged"
echo "Output: data/processed"

[ ! -f "data/silver_labels/teacher_output.jsonl" ] && echo "ERROR: teacher_output.jsonl missing!" && exit 1

mkdir -p data/processed

python -m src.preprocessing.prepare_data_merged \
    --original data/silver_labels/teacher_output.jsonl \
    --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \
    --output-dir data/processed \
    --model dicta-il/dictabert \
    --downsample-ratio 3.0

[ $? -eq 0 ] && echo "DONE:" && wc -l data/processed/*.jsonl || (echo "FAILED"; exit 1)
echo "End: $(date)"
