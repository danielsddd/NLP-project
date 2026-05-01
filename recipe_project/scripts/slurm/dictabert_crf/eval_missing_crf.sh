#!/bin/bash
#SBATCH --job-name=dictcrf_eval_all
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_eval_missing.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_eval_missing.err

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
echo "GOLD EVALUATION: all missing dictabert_crf configs"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=================================================="

GOLD="data/gold_validation/gold_tokenized_dictabert.jsonl"
if [ ! -s "$GOLD" ]; then
    echo "Regenerating tokenized gold..."
    python scripts/local/prepare_gold_for_eval.py --all \
        || { echo "FATAL: gold tokenization failed"; exit 1; }
fi
echo "Gold file: $GOLD ($(wc -l < $GOLD) records)"

# -------------------------------------------------------
# Helper: run gold eval for one CRF config
# Args: $1=ckpt_dir (containing best_model.pt)
#       $2=output_dir (results/dictabert_crf/<config>)
#       $3=silver_test_file
# -------------------------------------------------------
run_eval() {
    local ckpt_dir="$1"
    local out_dir="$2"
    local silver_test="$3"
    local vname=$(basename "$ckpt_dir")

    echo ""
    echo "--- Evaluating: $vname ---"
    echo "    ckpt:   $ckpt_dir"
    echo "    silver: $silver_test"

    if [ ! -f "$ckpt_dir/best_model.pt" ]; then
        echo "    SKIP: no best_model.pt found"
        return
    fi

    # Gold eval (always same gold file)
    if [ ! -f "$out_dir/gold/evaluation_results.json" ]; then
        echo "    → Gold eval..."
        mkdir -p "$out_dir/gold"
        python -m scripts.local.evaluate_crf \
            --ckpt-dir   "$ckpt_dir" \
            --test-file  "$GOLD" \
            --output-dir "$out_dir/gold" \
            --model-name "dicta-il/dictabert" 2>&1 | tee "$out_dir/gold_eval.log"
    else
        echo "    ⏭  Gold eval already exists — skipping"
    fi

    # Silver eval (matched to training data)
    if [ ! -f "$out_dir/silver/evaluation_results.json" ]; then
        if [ -f "$silver_test" ]; then
            echo "    → Silver eval ($silver_test)..."
            mkdir -p "$out_dir/silver"
            python -m scripts.local.evaluate_crf \
                --ckpt-dir   "$ckpt_dir" \
                --test-file  "$silver_test" \
                --output-dir "$out_dir/silver" \
                --model-name "dicta-il/dictabert" 2>&1 | tee "$out_dir/silver_eval.log"
        else
            echo "    ⚠  Silver test not found: $silver_test"
        fi
    else
        echo "    ⏭  Silver eval already exists — skipping"
    fi
}

# -------------------------------------------------------
# ABLATIONS (best_model.pt lives in results/)
# -------------------------------------------------------
RBASE="results/dictabert_crf"
MBASE="models/checkpoints/dictabert_crf"

run_eval "$RBASE/A1b_crf_uniform_weights"  "$RBASE/A1b_crf_uniform_weights"  "data/processed_v2/test.jsonl"
run_eval "$RBASE/A2_crf_downsample_2"      "$RBASE/A2_crf_downsample_2"      "data/processed_ds2/test.jsonl"
run_eval "$RBASE/A2_crf_downsample_4"      "$RBASE/A2_crf_downsample_4"      "data/processed_ds4/test.jsonl"
run_eval "$RBASE/A5_crf_data_25pct"        "$RBASE/A5_crf_data_25pct"        "data/processed_frac25/test.jsonl"
run_eval "$RBASE/A5_crf_data_50pct"        "$RBASE/A5_crf_data_50pct"        "data/processed_frac50/test.jsonl"
run_eval "$RBASE/A5_crf_data_75pct"        "$RBASE/A5_crf_data_75pct"        "data/processed_frac75/test.jsonl"
run_eval "$RBASE/A6_crf_no_enriched"       "$RBASE/A6_crf_no_enriched"       "data/processed_no_enriched/test.jsonl"
run_eval "$RBASE/A7_crf_io_scheme"         "$RBASE/A7_crf_io_scheme"         "data/processed_io/test.jsonl"
run_eval "$RBASE/A8_crf_unanimous"         "$RBASE/A8_crf_unanimous"         "data/processed_unanimous/test.jsonl"

# -------------------------------------------------------
# P-SERIES (best_model.pt lives in models/checkpoints/)
# -------------------------------------------------------
run_eval "$MBASE/P3_crf_downsample_2"      "$RBASE/P3_crf_downsample_2"      "data/processed_ds2/test.jsonl"
run_eval "$MBASE/P3_crf_downsample_4"      "$RBASE/P3_crf_downsample_4"      "data/processed_ds4/test.jsonl"
run_eval "$MBASE/P4_crf_thread_aware"      "$RBASE/P4_crf_thread_aware"      "data/processed_thread_aware/test.jsonl"
run_eval "$MBASE/P5_crf_thread_no_enriched" "$RBASE/P5_crf_thread_no_enriched" "data/processed_thread_aware/test.jsonl"
run_eval "$MBASE/P10_crf_thread_aware_io"  "$RBASE/P10_crf_thread_aware_io"  "data/processed_thread_aware_io/test.jsonl"

echo ""
echo "=================================================="
echo "ALL DONE. End: $(date)"
echo "=================================================="
