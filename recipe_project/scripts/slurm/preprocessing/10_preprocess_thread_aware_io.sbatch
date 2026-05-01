#!/bin/bash
# Preprocessing: Thread-aware + IO tagging scheme (V2 DATA)

#SBATCH --job-name=prep_ta_io_v2
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/preprocessing/%j_thread_aware_io_v2.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/preprocessing/%j_thread_aware_io_v2.err

USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
cd $PROJECT_DIR

echo "PREPROCESSING: Thread-aware + IO scheme (USING V2)"
# Safety Check
[ ! -f "data/silver_labels/teacher_output_v2.jsonl" ] && echo "ERROR: teacher_output_v2.jsonl missing!" && exit 1

mkdir -p data/processed_thread_aware_io

python -m src.preprocessing.prepare_thread_aware     --original data/silver_labels/teacher_output_v2.jsonl     --enriched data/silver_labels/threads_positives_focus_labeled_v2.jsonl     --output-dir data/processed_thread_aware_io     --model dicta-il/dictabert     --scheme IO     --downsample-ratio 3.0

[ $? -eq 0 ] && echo "DONE: V2 Data Prepped" || (echo "FAILED"; exit 1)
