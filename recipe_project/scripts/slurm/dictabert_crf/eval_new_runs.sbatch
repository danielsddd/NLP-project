#!/bin/bash
#SBATCH --job-name=eval_new_runs
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
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

# ---------- Gold files ----------
GOLD_BIO_THREAD="data/gold_validation/gold_tokenized_dictabert_bio_thread.jsonl"
GOLD_IO_THREAD="data/gold_validation/gold_tokenized_dictabert_io_thread.jsonl"
GOLD_IO_PLAIN="data/gold_validation/gold_tokenized_dictabert_io.jsonl"
GOLD_LARGE_BIO="data/gold_validation/gold_tokenized_dictabert_large.jsonl"
GOLD_LARGE_IO="data/gold_validation/gold_tokenized_dictabert_large_io.jsonl"
GOLD_LARGE_BIO_THREAD="data/gold_validation/gold_tokenized_dictabert_large_bio_thread.jsonl"
GOLD_LARGE_IO_THREAD="data/gold_validation/gold_tokenized_dictabert_large_io_thread.jsonl"

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

# ---------- Helper function ----------
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
# SECTION 1: RETRAINED THREAD‑AWARE (P4, P5, P10, P10v3 seeds)
# ==============================================================
echo ""
echo "================================================================"
echo "SECTION 1: RETRAINED THREAD-AWARE MODELS"
echo "================================================================"

eval_one \
    "models/checkpoints/dictabert_crf/P4_crf_thread_aware" \
    "results/dictabert_crf/P4_crf_thread_aware" \
    "$GOLD_BIO_THREAD" \
    "data/processed_thread_aware/test.jsonl" \
    "dicta-il/dictabert"

eval_one \
    "models/checkpoints/dictabert_crf/P5_crf_thread_no_enriched" \
    "results/dictabert_crf/P5_crf_thread_no_enriched" \
    "$GOLD_BIO_THREAD" \
    "data/processed_thread_aware_no_enriched/test.jsonl" \
    "dicta-il/dictabert"

eval_one \
    "models/checkpoints/dictabert_crf/P10_crf_thread_aware_io_fixed" \
    "results/dictabert_crf/P10_crf_thread_aware_io_fixed" \
    "$GOLD_IO_THREAD" \
    "data/processed_thread_aware_io/test.jsonl" \
    "dicta-il/dictabert"

for seed in 123 2026 7777; do
    eval_one \
        "models/checkpoints/dictabert_crf/P10v3_seed${seed}_fixed" \
        "results/dictabert_crf/P10v3_seed${seed}_fixed" \
        "$GOLD_IO_THREAD" \
        "data/processed_thread_aware_io/test.jsonl" \
        "dicta-il/dictabert"
done

for variant in P10v3_fixed_weights P10v3_lr1e5; do
    eval_one \
        "models/checkpoints/dictabert_crf/$variant" \
        "results/dictabert_crf/$variant" \
        "$GOLD_IO_THREAD" \
        "data/processed_thread_aware_io/test.jsonl" \
        "dicta-il/dictabert"
done

# ==============================================================
# SECTION 2: OLD THREAD‑AWARE BASELINES
# ==============================================================
echo ""
echo "================================================================"
echo "SECTION 2: OLD THREAD-AWARE (no real context)"
echo "================================================================"

rm -f results/dictabert_crf/P10_crf_thread_aware_io/gold/evaluation_results.json
eval_one \
    "models/checkpoints/dictabert_crf/P10_crf_thread_aware_io" \
    "results/dictabert_crf/P10_crf_thread_aware_io" \
    "$GOLD_IO_THREAD" \
    "data/processed_thread_aware_io/test.jsonl" \
    "dicta-il/dictabert"

# ==============================================================
# SECTION 3: NON‑THREAD MODELS
# ==============================================================
echo ""
echo "================================================================"
echo "SECTION 3: NON-THREAD MODELS"
echo "================================================================"

rm -f results/dictabert_crf/A7_crf_io_scheme/gold/evaluation_results.json
eval_one \
    "results/dictabert_crf/A7_crf_io_scheme" \
    "results/dictabert_crf/A7_crf_io_scheme" \
    "$GOLD_IO_PLAIN" \
    "data/processed_io/test.jsonl" \
    "dicta-il/dictabert"

# ==============================================================
# SECTION 4: DICTABERT‑LARGE + CRF
# ==============================================================
echo ""
echo "================================================================"
echo "SECTION 4: DICTABERT-LARGE + CRF (thread-aware)"
echo "================================================================"

eval_one \
    "models/checkpoints/dictabert_large_crf/P4_large_crf_thread_bio" \
    "results/dictabert_large_crf/P4_large_crf_thread_bio" \
    "$GOLD_LARGE_BIO_THREAD" \
    "data/processed_thread_aware_large/test.jsonl" \
    "dicta-il/dictabert-large"

eval_one \
    "models/checkpoints/dictabert_large_crf/P10_large_crf_thread_io" \
    "results/dictabert_large_crf/P10_large_crf_thread_io" \
    "$GOLD_LARGE_IO_THREAD" \
    "data/processed_thread_aware_io_large/test.jsonl" \
    "dicta-il/dictabert-large"

eval_one \
    "models/checkpoints/dictabert_large_crf/P2_crf_weighted" \
    "results/dictabert_large_crf/P2_crf_weighted" \
    "$GOLD_LARGE_BIO" \
    "data/processed_v2/test.jsonl" \
    "dicta-il/dictabert-large"

# ==============================================================
# SECTION 5: BINARY CLASSIFICATION FROM SPANS
# ==============================================================
echo ""
echo "================================================================"
echo "SECTION 5: BINARY CLASSIFICATION METRICS"
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
printf "%-55s %-8s %-8s %-8s %-8s\n" "Config" "Split" "F1" "Prec" "Recall"
echo "--------------------------------------------------------------------------------------------------------------"

for result_dir in \
    results/dictabert_crf/P4_crf_thread_aware \
    results/dictabert_crf/P5_crf_thread_no_enriched \
    results/dictabert_crf/P10_crf_thread_aware_io_fixed \
    results/dictabert_crf/P10v3_seed123_fixed \
    results/dictabert_crf/P10v3_seed2026_fixed \
    results/dictabert_crf/P10v3_seed7777_fixed \
    results/dictabert_crf/P10v3_fixed_weights \
    results/dictabert_crf/P10v3_lr1e5 \
    results/dictabert_crf/P10_crf_thread_aware_io \
    results/dictabert_crf/A7_crf_io_scheme \
    results/dictabert_large_crf/P4_large_crf_thread_bio \
    results/dictabert_large_crf/P10_large_crf_thread_io \
    results/dictabert_large_crf/P2_crf_weighted; do

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
" | awk -F'|' '{printf "%-55s %-8s %-8s %-8s %-8s\n", $1, $2, $3, $4, $5}'
        fi
    done
done

echo ""
echo "=================================================="
echo "ALL DONE. End: $(date)"
echo "=================================================="
