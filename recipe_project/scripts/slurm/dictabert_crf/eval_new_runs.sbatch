#!/bin/bash
#SBATCH --job-name=eval_new_runs
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_eval_new_runs.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_eval_new_runs.err

USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
cd $PROJECT_DIR

echo "=================================================="
echo "EVAL ALL NEW RUNS"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=================================================="

# ---- Gold files ----
GOLD_BIO="data/gold_validation/gold_tokenized_dictabert.jsonl"
GOLD_IO="data/gold_validation/gold_tokenized_dictabert_io.jsonl"
GOLD_LARGE_BIO="data/gold_validation/gold_tokenized_dictabert_large.jsonl"
GOLD_LARGE_IO="data/gold_validation/gold_tokenized_dictabert_large_io.jsonl"

# Generate any missing gold files
if [ ! -s "$GOLD_LARGE_BIO" ]; then
    echo "Generating gold for dictabert-large (BIO)..."
    python scripts/local/prepare_gold_for_eval.py \
        --model dicta-il/dictabert-large --scheme BIO \
        --output "$GOLD_LARGE_BIO"
fi
if [ ! -s "$GOLD_LARGE_IO" ]; then
    echo "Generating gold for dictabert-large (IO)..."
    python scripts/local/prepare_gold_for_eval.py \
        --model dicta-il/dictabert-large --scheme IO \
        --output "$GOLD_LARGE_IO"
fi

# ---- Helper function ----
eval_one() {
    local ckpt="$1"
    local result="$2"
    local gold="$3"
    local silver="$4"
    local model_name="$5"
    local name=$(basename "$ckpt")

    echo ""
    echo "============================================================"
    echo "  $name"
    echo "  ckpt:   $ckpt"
    echo "  gold:   $gold"
    echo "  silver: $silver"
    echo "============================================================"

    if [ ! -f "$ckpt/best_model.pt" ]; then
        echo "  SKIP: no best_model.pt"
        return
    fi

    if [ ! -f "$result/gold/evaluation_results.json" ]; then
        mkdir -p "$result/gold"
        python -m scripts.local.evaluate_crf \
            --ckpt-dir "$ckpt" \
            --test-file "$gold" \
            --output-dir "$result/gold" \
            --model-name "$model_name" 2>&1 | tee "$result/gold_eval.log"
        echo "  ✅ gold done"
    else
        echo "  ⏭  gold exists"
    fi

    if [ -f "$silver" ] && [ ! -f "$result/silver/evaluation_results.json" ]; then
        mkdir -p "$result/silver"
        python -m scripts.local.evaluate_crf \
            --ckpt-dir "$ckpt" \
            --test-file "$silver" \
            --output-dir "$result/silver" \
            --model-name "$model_name" 2>&1 | tee "$result/silver_eval.log"
        echo "  ✅ silver done"
    elif [ ! -f "$silver" ]; then
        echo "  ⚠  silver file not found: $silver"
    else
        echo "  ⏭  silver exists"
    fi
}

# ==============================================================
# SECTION 1: BUG-FIX RE-RUNS (dictabert + CRF)
# ==============================================================
echo ""
echo "================================================================"
echo "SECTION 1: BUG-FIX RE-RUNS"
echo "================================================================"

eval_one \
    "models/checkpoints/dictabert_crf/P4v2_thread_fixed" \
    "results/dictabert_crf/P4v2_thread_fixed" \
    "$GOLD_BIO" \
    "data/processed_thread_aware/test.jsonl" \
    "dicta-il/dictabert"

eval_one \
    "models/checkpoints/dictabert_crf/P5v2_thread_no_enriched_fixed" \
    "results/dictabert_crf/P5v2_thread_no_enriched_fixed" \
    "$GOLD_BIO" \
    "data/processed_thread_aware_no_enriched/test.jsonl" \
    "dicta-il/dictabert"

eval_one \
    "models/checkpoints/dictabert_crf/P10v3_fixed_weights" \
    "results/dictabert_crf/P10v3_fixed_weights" \
    "$GOLD_IO" \
    "data/processed_thread_aware_io/test.jsonl" \
    "dicta-il/dictabert"

# ==============================================================
# SECTION 2: NEW COMBOS (dictabert + CRF)
# ==============================================================
echo ""
echo "================================================================"
echo "SECTION 2: NEW COMBOS (dictabert + CRF)"
echo "================================================================"

eval_one \
    "models/checkpoints/dictabert_crf/P10v4_focal" \
    "results/dictabert_crf/P10v4_focal" \
    "$GOLD_IO" \
    "data/processed_thread_aware_io/test.jsonl" \
    "dicta-il/dictabert"

eval_one \
    "models/checkpoints/dictabert_crf/P4v3_thread_focal" \
    "results/dictabert_crf/P4v3_thread_focal" \
    "$GOLD_BIO" \
    "data/processed_thread_aware/test.jsonl" \
    "dicta-il/dictabert"

eval_one \
    "models/checkpoints/dictabert_crf/P10v3_lr1e5" \
    "results/dictabert_crf/P10v3_lr1e5" \
    "$GOLD_IO" \
    "data/processed_thread_aware_io/test.jsonl" \
    "dicta-il/dictabert"

# ==============================================================
# SECTION 3: SEED SWEEP (for confidence intervals)
# ==============================================================
echo ""
echo "================================================================"
echo "SECTION 3: SEED SWEEP"
echo "================================================================"

eval_one \
    "models/checkpoints/dictabert_crf/P10v3_seed123" \
    "results/dictabert_crf/P10v3_seed123" \
    "$GOLD_IO" \
    "data/processed_thread_aware_io/test.jsonl" \
    "dicta-il/dictabert"

eval_one \
    "models/checkpoints/dictabert_crf/P10v3_seed2026" \
    "results/dictabert_crf/P10v3_seed2026" \
    "$GOLD_IO" \
    "data/processed_thread_aware_io/test.jsonl" \
    "dicta-il/dictabert"

eval_one \
    "models/checkpoints/dictabert_crf/P10v3_seed7777" \
    "results/dictabert_crf/P10v3_seed7777" \
    "$GOLD_IO" \
    "data/processed_thread_aware_io/test.jsonl" \
    "dicta-il/dictabert"

# ==============================================================
# SECTION 4: DICTABERT-LARGE + CRF (all variants)
# ==============================================================
echo ""
echo "================================================================"
echo "SECTION 4: DICTABERT-LARGE + CRF"
echo "================================================================"

eval_one \
    "models/checkpoints/dictabert_large_crf/P2_crf_weighted" \
    "results/dictabert_large_crf/P2_crf_weighted" \
    "$GOLD_LARGE_BIO" \
    "data/processed_v2/test.jsonl" \
    "dicta-il/dictabert-large"

eval_one \
    "models/checkpoints/dictabert_large_crf/P10_large_crf_thread_io" \
    "results/dictabert_large_crf/P10_large_crf_thread_io" \
    "$GOLD_LARGE_IO" \
    "data/processed_thread_aware_io/test.jsonl" \
    "dicta-il/dictabert-large"

eval_one \
    "models/checkpoints/dictabert_large_crf/P10v4_large_focal" \
    "results/dictabert_large_crf/P10v4_large_focal" \
    "$GOLD_LARGE_IO" \
    "data/processed_thread_aware_io/test.jsonl" \
    "dicta-il/dictabert-large"

eval_one \
    "models/checkpoints/dictabert_large_crf/P4_large_crf_thread_bio" \
    "results/dictabert_large_crf/P4_large_crf_thread_bio" \
    "$GOLD_LARGE_BIO" \
    "data/processed_thread_aware/test.jsonl" \
    "dicta-il/dictabert-large"

eval_one \
    "models/checkpoints/dictabert_large_crf/P10_large_lr3e5" \
    "results/dictabert_large_crf/P10_large_lr3e5" \
    "$GOLD_LARGE_IO" \
    "data/processed_thread_aware_io/test.jsonl" \
    "dicta-il/dictabert-large"

# ==============================================================
# SECTION 5: RE-EVAL BROKEN OLD RUNS (IO gold was wrong)
# ==============================================================
echo ""
echo "================================================================"
echo "SECTION 5: RE-EVAL BROKEN OLD RUNS"
echo "================================================================"

rm -f results/dictabert_crf/P10_crf_thread_aware_io/gold/evaluation_results.json
eval_one \
    "models/checkpoints/dictabert_crf/P10_crf_thread_aware_io" \
    "results/dictabert_crf/P10_crf_thread_aware_io" \
    "$GOLD_IO" \
    "data/processed_thread_aware_io/test.jsonl" \
    "dicta-il/dictabert"

rm -f results/dictabert_crf/A7_crf_io_scheme/gold/evaluation_results.json
eval_one \
    "results/dictabert_crf/A7_crf_io_scheme" \
    "results/dictabert_crf/A7_crf_io_scheme" \
    "$GOLD_IO" \
    "data/processed_io/test.jsonl" \
    "dicta-il/dictabert"

# ==============================================================
# SECTION 6: BINARY CLASSIFICATION FROM SPANS
# ==============================================================
echo ""
echo "================================================================"
echo "SECTION 6: BINARY CLASSIFICATION METRICS"
echo "================================================================"

python scripts/local/compute_binary_from_spans.py --batch

# ==============================================================
# SUMMARY TABLE
# ==============================================================
echo ""
echo "================================================================"
echo "RESULTS SUMMARY TABLE"
echo "================================================================"
echo ""
printf "%-50s %-8s %-8s %-8s %-8s\n" "Config" "Split" "F1" "Prec" "Recall"
echo "--------------------------------------------------------------------------------------------------------------"

for result_dir in \
    results/dictabert_crf/P4v2_thread_fixed \
    results/dictabert_crf/P5v2_thread_no_enriched_fixed \
    results/dictabert_crf/P10v3_fixed_weights \
    results/dictabert_crf/P10v4_focal \
    results/dictabert_crf/P4v3_thread_focal \
    results/dictabert_crf/P10v3_lr1e5 \
    results/dictabert_crf/P10v3_seed123 \
    results/dictabert_crf/P10v3_seed2026 \
    results/dictabert_crf/P10v3_seed7777 \
    results/dictabert_large_crf/P2_crf_weighted \
    results/dictabert_large_crf/P10_large_crf_thread_io \
    results/dictabert_large_crf/P10v4_large_focal \
    results/dictabert_large_crf/P4_large_crf_thread_bio \
    results/dictabert_large_crf/P10_large_lr3e5 \
    results/dictabert_crf/P10_crf_thread_aware_io \
    results/dictabert_crf/A7_crf_io_scheme; do

    name=$(echo "$result_dir" | sed 's|results/||')
    for split in gold silver; do
        f="$result_dir/$split/evaluation_results.json"
        if [ -f "$f" ]; then
            python3 -c "
import json
with open('$f') as fp:
    d = json.load(fp)
f1 = d.get('entity_f1', 'N/A')
p = d.get('entity_precision', 'N/A')
r = d.get('entity_recall', 'N/A')
print(f'$name|$split|{f1}|{p}|{r}')
" | awk -F'|' '{printf "%-50s %-8s %-8s %-8s %-8s\n", $1, $2, $3, $4, $5}'
        fi
    done
done

echo ""
echo "=================================================="
echo "ALL DONE. End: $(date)"
echo "=================================================="
