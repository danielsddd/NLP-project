#!/bin/bash
# Preprocessing: Thread-aware context — prepend parent question to reply examples

#SBATCH --job-name=prep_thread
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/preprocessing/%j_thread_aware.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/preprocessing/%j_thread_aware.err

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

echo "PREPROCESSING: Thread-aware context"
echo "For reply-sourced examples, prepends [שאלה] <question> [SEP] <reply>"
echo "This gives the student model question→answer context"

mkdir -p data/processed_thread_aware

python -m src.preprocessing.prepare_thread_aware \
    --original data/silver_labels/teacher_output.jsonl \
    --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \
    --output-dir data/processed_thread_aware \
    --model dicta-il/dictabert \
    --downsample-ratio 3.0

[ $? -eq 0 ] && echo "DONE:" && wc -l data/processed_thread_aware/*.jsonl || (echo "FAILED"; exit 1)
echo "End: $(date)"
