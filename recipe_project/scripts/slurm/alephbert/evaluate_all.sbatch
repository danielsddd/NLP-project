#!/bin/bash
# =============================================================================
# Evaluate ALL variants of: alephbert
# =============================================================================

#SBATCH --job-name=alep_eval
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_3090|a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/alephbert/%j_evaluate_all.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/alephbert/%j_evaluate_all.err

USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
cd $PROJECT_DIR

echo "=================================================="
echo "EVALUATING ALL VARIANTS: alephbert"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=================================================="

CKPT_BASE="models/checkpoints/alephbert"
RESULT_BASE="results/alephbert"
GOLD="data/gold_validation/gold_final.jsonl"

# Default silver test file (BIO scheme)
DEFAULT_TEST="data/processed/test.jsonl"

for vdir in $CKPT_BASE/*/; do
    vname=$(basename "$vdir")

    # --- Find trained model ---
    MP=""
    [ -d "$vdir/best_model" ] && MP="$vdir/best_model"
    [ -f "$vdir/best_model.pt" ] && MP="$vdir/best_model.pt"
    [ -f "$vdir/pytorch_model.bin" ] && MP="$vdir"
    [ -f "$vdir/model.safetensors" ] && MP="$vdir"

    if [ -z "$MP" ]; then
        echo "SKIP: $vname (no model found)"; continue
    fi

    # --- Pick correct test file based on variant name ---
    # IO-scheme variants need IO-labeled test data
    # Thread-aware-IO combos also need IO test data
    # All others use default BIO test data
    TEST_FILE="$DEFAULT_TEST"

    case "$vname" in
        *io_scheme*|*io*)
            # Check for IO test files in order of specificity
            if [ -f "data/processed_thread_aware_io/test.jsonl" ] && [[ "$vname" == *thread* ]]; then
                TEST_FILE="data/processed_thread_aware_io/test.jsonl"
            elif [ -f "data/processed_io/test.jsonl" ]; then
                TEST_FILE="data/processed_io/test.jsonl"
            else
                echo "  ⚠ WARNING: No IO test file found for $vname, using BIO test (results may be wrong!)"
            fi
            ;;
        *downsample_2*|*ds2*)
            [ -f "data/processed_ds2/test.jsonl" ] && TEST_FILE="data/processed_ds2/test.jsonl"
            ;;
        *downsample_4*|*ds4*)
            [ -f "data/processed_ds4/test.jsonl" ] && TEST_FILE="data/processed_ds4/test.jsonl"
            ;;
        *no_enriched*)
            [ -f "data/processed_no_enriched/test.jsonl" ] && TEST_FILE="data/processed_no_enriched/test.jsonl"
            ;;
        *unanimous*)
            [ -f "data/processed_unanimous/test.jsonl" ] && TEST_FILE="data/processed_unanimous/test.jsonl"
            ;;
        *thread_aware*|P4_*|P5_*)
            [ -f "data/processed_thread_aware/test.jsonl" ] && TEST_FILE="data/processed_thread_aware/test.jsonl"
            ;;
    esac

    echo ""
    echo "=================================================="
    echo "Evaluating: alephbert / $vname"
    echo "Model:     $MP"
    echo "Test file: $TEST_FILE"
    echo "=================================================="

    RDIR="$RESULT_BASE/$vname"
    mkdir -p "$RDIR"

    # Evaluate on GOLD set (primary — always same format)
    if [ -f "$GOLD" ]; then
        echo "  → Gold set evaluation..."
        python -m src.evaluation.evaluate \
            --model-path "$MP" \
            --test-file "$GOLD" \
            --output-dir "$RDIR/gold" \
            --bootstrap 2>&1 | tee "$RDIR/gold_eval.log"
    else
        echo "  ⚠ Gold file not found: $GOLD"
    fi

    # Evaluate on silver test (secondary — matched to training data)
    if [ -f "$TEST_FILE" ]; then
        echo "  → Silver test ($TEST_FILE)..."
        python -m src.evaluation.evaluate \
            --model-path "$MP" \
            --test-file "$TEST_FILE" \
            --output-dir "$RDIR/silver" \
            --bootstrap 2>&1 | tee "$RDIR/silver_eval.log"
    else
        echo "  ⚠ Test file not found: $TEST_FILE"
    fi
done

echo ""
echo "=================================================="
echo "ALL EVALUATIONS COMPLETE: alephbert"
echo "Results in: $RESULT_BASE/"
echo "End: $(date)"
echo "=================================================="
