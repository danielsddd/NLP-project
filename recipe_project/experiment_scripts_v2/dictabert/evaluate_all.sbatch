#!/bin/bash
# =============================================================================
# Model: dictabert | Variant: evaluate_all
# =============================================================================

#SBATCH --job-name=eval_dictaber
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_3090|a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/logs/slurm_output/dictabert/%j_evaluate_all.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/logs/slurm_output/dictabert/%j_evaluate_all.err

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

echo "=================================================="
echo "EVALUATING ALL VARIANTS: dictabert"
echo "=================================================="

CKPT_BASE="models/checkpoints/dictabert"
RESULT_BASE="results/dictabert"
GOLD="data/gold_validation/gold_final.jsonl"
TEST="data/processed/test.jsonl"

for vdir in $CKPT_BASE/*/; do
    vname=$(basename "$vdir")
    
    # Find model
    MP=""
    [ -d "$vdir/best_model" ] && MP="$vdir/best_model"
    [ -f "$vdir/best_model.pt" ] && MP="$vdir/best_model.pt"
    [ -f "$vdir/pytorch_model.bin" ] && MP="$vdir"
    [ -f "$vdir/model.safetensors" ] && MP="$vdir"
    
    if [ -z "$MP" ]; then
        echo "SKIP: $vname (no model found)"; continue
    fi
    
    echo ""; echo "--- Evaluating: dictabert/$vname ---"
    RDIR="$RESULT_BASE/$vname"
    mkdir -p "$RDIR"
    
    if [ -f "$GOLD" ]; then
        echo "  Gold set..."
        python -m src.evaluation.evaluate \
            --model-path "$MP" --test-file "$GOLD" \
            --output-dir "$RDIR/gold" --bootstrap 2>&1 | tee "$RDIR/gold_eval.log"
    fi
    
    if [ -f "$TEST" ]; then
        echo "  Silver test..."
        python -m src.evaluation.evaluate \
            --model-path "$MP" --test-file "$TEST" \
            --output-dir "$RDIR/silver" --bootstrap 2>&1 | tee "$RDIR/silver_eval.log"
    fi
done

echo ""; echo "ALL DONE: results/$RESULT_BASE/"
echo "End: $(date)"
