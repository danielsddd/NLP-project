#!/bin/bash
#SBATCH --job-name=eval_all_models
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/%j_eval_all_models.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/%j_eval_all_models.err

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
echo "EVALUATING ALL MODELS"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=================================================="

# -------------------------------------------------------
# eval_crf: evaluate a .pt CRF model
#   $1 = ckpt_dir    (dir containing best_model.pt)
#   $2 = result_dir  (where to write gold/ and silver/)
#   $3 = gold_file
#   $4 = silver_file
# -------------------------------------------------------
eval_crf() {
    local ckpt_dir="$1"
    local result_dir="$2"
    local gold_file="$3"
    local silver_file="$4"
    local name=$(basename "$ckpt_dir")

    if [ ! -f "$ckpt_dir/best_model.pt" ]; then
        echo "  [SKIP] $name — no best_model.pt at $ckpt_dir"
        return
    fi

    echo ""
    echo "  [CRF] $name"

    if [ ! -f "$result_dir/gold/evaluation_results.json" ]; then
        mkdir -p "$result_dir/gold"
        python -m scripts.local.evaluate_crf \
            --ckpt-dir   "$ckpt_dir" \
            --test-file  "$gold_file" \
            --output-dir "$result_dir/gold" \
            --model-name "dicta-il/dictabert" \
            2>&1 | tee "$result_dir/gold_eval.log"
        echo "    ✅ gold done"
    else
        echo "    ⏭  gold already exists"
    fi

    if [ -n "$silver_file" ] && [ -f "$silver_file" ] && [ ! -f "$result_dir/silver/evaluation_results.json" ]; then
        mkdir -p "$result_dir/silver"
        python -m scripts.local.evaluate_crf \
            --ckpt-dir   "$ckpt_dir" \
            --test-file  "$silver_file" \
            --output-dir "$result_dir/silver" \
            --model-name "dicta-il/dictabert" \
            2>&1 | tee "$result_dir/silver_eval.log"
        echo "    ✅ silver done"
    elif [ ! -f "$result_dir/silver/evaluation_results.json" ]; then
        echo "    ⚠  silver file not found: $silver_file"
    else
        echo "    ⏭  silver already exists"
    fi
}

# -------------------------------------------------------
# eval_hf: evaluate a HuggingFace model (safetensors)
#   $1 = model_dir   (dir containing model.safetensors)
#   $2 = result_dir
#   $3 = gold_file
#   $4 = silver_file
# -------------------------------------------------------
eval_hf() {
    local model_dir="$1"
    local result_dir="$2"
    local gold_file="$3"
    local silver_file="$4"
    local name=$(basename "$(dirname "$model_dir")")/$(basename "$model_dir")

    if [ ! -f "$model_dir/model.safetensors" ] && [ ! -f "$model_dir/pytorch_model.bin" ]; then
        echo "  [SKIP] $name — no model weights at $model_dir"
        return
    fi

    echo ""
    echo "  [HF]  $name"

    if [ ! -f "$result_dir/gold/evaluation_results.json" ]; then
        mkdir -p "$result_dir/gold"
        python -m src.evaluation.evaluate \
            --model-path "$model_dir" \
            --test-file  "$gold_file" \
            --output-dir "$result_dir/gold" \
            --bootstrap \
            2>&1 | tee "$result_dir/gold_eval.log"
        echo "    ✅ gold done"
    else
        echo "    ⏭  gold already exists"
    fi

    if [ -n "$silver_file" ] && [ -f "$silver_file" ] && [ ! -f "$result_dir/silver/evaluation_results.json" ]; then
        mkdir -p "$result_dir/silver"
        python -m src.evaluation.evaluate \
            --model-path "$model_dir" \
            --test-file  "$silver_file" \
            --output-dir "$result_dir/silver" \
            --bootstrap \
            2>&1 | tee "$result_dir/silver_eval.log"
        echo "    ✅ silver done"
    elif [ ! -f "$result_dir/silver/evaluation_results.json" ]; then
        echo "    ⚠  silver file not found: $silver_file"
    else
        echo "    ⏭  silver already exists"
    fi
}

# ================================================================
# GOLD FILES (per tokenizer)
# ================================================================
GOLD_DIR="data/gold_validation"
GOLD_DICTABERT="$GOLD_DIR/gold_tokenized_dictabert.jsonl"
GOLD_DICTA_LARGE="$GOLD_DIR/gold_tokenized_dictabert_large.jsonl"
GOLD_ALEPHBERT="$GOLD_DIR/gold_tokenized_alephbert.jsonl"
GOLD_HEBERT="$GOLD_DIR/gold_tokenized_hebert.jsonl"
GOLD_MBERT="$GOLD_DIR/gold_tokenized_mbert.jsonl"
GOLD_XLMR="$GOLD_DIR/gold_tokenized_xlmr.jsonl"

# ================================================================
# SILVER TEST FILES
# ================================================================
SIL_V2="data/processed_v2/test.jsonl"
SIL_DICTA="data/processed_dictabert/test.jsonl"
SIL_LARGE="data/processed_dicta_large/test.jsonl"
SIL_DS2="data/processed_ds2/test.jsonl"
SIL_DS4="data/processed_ds4/test.jsonl"
SIL_F25="data/processed_frac25/test.jsonl"
SIL_F50="data/processed_frac50/test.jsonl"
SIL_F75="data/processed_frac75/test.jsonl"
SIL_NO_ENR="data/processed_no_enriched/test.jsonl"
SIL_IO="data/processed_io/test.jsonl"
SIL_UNANI="data/processed_unanimous/test.jsonl"
SIL_TA="data/processed_thread_aware/test.jsonl"
SIL_TA_IO="data/processed_thread_aware_io/test.jsonl"
SIL_TA_NO_ENR="data/processed_thread_aware_no_enriched/test.jsonl"
SIL_MBERT="data/processed_mbert/test.jsonl"
SIL_HEBERT="data/processed_hebert/test.jsonl"
SIL_XLMR="data/processed_xlmr/test.jsonl"

CKPT="models/checkpoints"
RES="results"

echo ""
echo "================================================================"
echo "SECTION 1: dictabert (HF)"
echo "================================================================"
for config in P0_baseline P1_add_weights P2_add_focal; do
    eval_hf "$CKPT/dictabert/$config/best_model" \
            "$RES/dictabert/$config" \
            "$GOLD_DICTABERT" "$SIL_DICTA"
done

echo ""
echo "================================================================"
echo "SECTION 2: dictabert_large (HF)"
echo "================================================================"
# P-series
for config in P0_baseline P1_add_weights P2_add_focal; do
    eval_hf "$CKPT/dictabert_large/$config/best_model" \
            "$RES/dictabert_large/$config" \
            "$GOLD_DICTA_LARGE" "$SIL_LARGE"
done
eval_hf "$CKPT/dictabert_large/P3_add_downsample_2/best_model" \
        "$RES/dictabert_large/P3_add_downsample_2" \
        "$GOLD_DICTA_LARGE" "$SIL_DS2"
eval_hf "$CKPT/dictabert_large/P3_add_downsample_4/best_model" \
        "$RES/dictabert_large/P3_add_downsample_4" \
        "$GOLD_DICTA_LARGE" "$SIL_DS4"
# Ablations
eval_hf "$RES/dictabert_large/A1b_uniform_weights/best_model"  "$RES/dictabert_large/A1b_uniform_weights"  "$GOLD_DICTA_LARGE" "$SIL_LARGE"
eval_hf "$RES/dictabert_large/A2_downsample_2/best_model"      "$RES/dictabert_large/A2_downsample_2"      "$GOLD_DICTA_LARGE" "$SIL_DS2"
eval_hf "$RES/dictabert_large/A2_downsample_4/best_model"      "$RES/dictabert_large/A2_downsample_4"      "$GOLD_DICTA_LARGE" "$SIL_DS4"
eval_hf "$RES/dictabert_large/A5_data_25pct/best_model"        "$RES/dictabert_large/A5_data_25pct"        "$GOLD_DICTA_LARGE" "$SIL_F25"
eval_hf "$RES/dictabert_large/A5_data_50pct/best_model"        "$RES/dictabert_large/A5_data_50pct"        "$GOLD_DICTA_LARGE" "$SIL_F50"
eval_hf "$RES/dictabert_large/A5_data_75pct/best_model"        "$RES/dictabert_large/A5_data_75pct"        "$GOLD_DICTA_LARGE" "$SIL_F75"
eval_hf "$RES/dictabert_large/A6_no_enriched/best_model"       "$RES/dictabert_large/A6_no_enriched"       "$GOLD_DICTA_LARGE" "$SIL_NO_ENR"
eval_hf "$RES/dictabert_large/A7_io_scheme/best_model"         "$RES/dictabert_large/A7_io_scheme"         "$GOLD_DICTA_LARGE" "$SIL_IO"
eval_hf "$RES/dictabert_large/A8_unanimous/best_model"         "$RES/dictabert_large/A8_unanimous"         "$GOLD_DICTA_LARGE" "$SIL_UNANI"

echo ""
echo "================================================================"
echo "SECTION 3: dictabert_crf (CRF)"
echo "================================================================"
# P-series (checkpoints in models/)
eval_crf "$CKPT/dictabert_crf/P0_crf_baseline"          "$RES/dictabert_crf/P0_crf_baseline"          "$GOLD_DICTABERT" "$SIL_V2"
eval_crf "$CKPT/dictabert_crf/P1_crf_weighted"          "$RES/dictabert_crf/P1_crf_weighted"          "$GOLD_DICTABERT" "$SIL_V2"
eval_crf "$CKPT/dictabert_crf/P2_crf_weighted"          "$RES/dictabert_crf/P2_crf_weighted"          "$GOLD_DICTABERT" "$SIL_V2"
eval_crf "$CKPT/dictabert_crf/P3_crf_downsample_2"      "$RES/dictabert_crf/P3_crf_downsample_2"      "$GOLD_DICTABERT" "$SIL_DS2"
eval_crf "$CKPT/dictabert_crf/P3_crf_downsample_4"      "$RES/dictabert_crf/P3_crf_downsample_4"      "$GOLD_DICTABERT" "$SIL_DS4"
eval_crf "$CKPT/dictabert_crf/P4_crf_thread_aware"      "$RES/dictabert_crf/P4_crf_thread_aware"      "$GOLD_DICTABERT" "$SIL_TA"
eval_crf "$CKPT/dictabert_crf/P5_crf_thread_no_enriched" "$RES/dictabert_crf/P5_crf_thread_no_enriched" "$GOLD_DICTABERT" "$SIL_TA_NO_ENR"
eval_crf "$CKPT/dictabert_crf/P10_crf_thread_aware_io"  "$RES/dictabert_crf/P10_crf_thread_aware_io"  "$GOLD_DICTABERT" "$SIL_TA_IO"
# Ablations (checkpoints in results/)
eval_crf "$RES/dictabert_crf/A1b_crf_uniform_weights"   "$RES/dictabert_crf/A1b_crf_uniform_weights"  "$GOLD_DICTABERT" "$SIL_V2"
eval_crf "$RES/dictabert_crf/A2_crf_downsample_2"       "$RES/dictabert_crf/A2_crf_downsample_2"      "$GOLD_DICTABERT" "$SIL_DS2"
eval_crf "$RES/dictabert_crf/A2_crf_downsample_4"       "$RES/dictabert_crf/A2_crf_downsample_4"      "$GOLD_DICTABERT" "$SIL_DS4"
eval_crf "$RES/dictabert_crf/A5_crf_data_25pct"         "$RES/dictabert_crf/A5_crf_data_25pct"        "$GOLD_DICTABERT" "$SIL_F25"
eval_crf "$RES/dictabert_crf/A5_crf_data_50pct"         "$RES/dictabert_crf/A5_crf_data_50pct"        "$GOLD_DICTABERT" "$SIL_F50"
eval_crf "$RES/dictabert_crf/A5_crf_data_75pct"         "$RES/dictabert_crf/A5_crf_data_75pct"        "$GOLD_DICTABERT" "$SIL_F75"
eval_crf "$RES/dictabert_crf/A6_crf_no_enriched"        "$RES/dictabert_crf/A6_crf_no_enriched"       "$GOLD_DICTABERT" "$SIL_NO_ENR"
eval_crf "$RES/dictabert_crf/A7_crf_io_scheme"          "$RES/dictabert_crf/A7_crf_io_scheme"         "$GOLD_DICTABERT" "$SIL_IO"
eval_crf "$RES/dictabert_crf/A8_crf_unanimous"          "$RES/dictabert_crf/A8_crf_unanimous"         "$GOLD_DICTABERT" "$SIL_UNANI"

echo ""
echo "================================================================"
echo "SECTION 4: alephbert (HF)"
echo "================================================================"
for config in P0_baseline P1_add_weights P2_add_focal; do
    eval_hf "$CKPT/alephbert/$config/best_model" \
            "$RES/alephbert/$config" \
            "$GOLD_ALEPHBERT" ""
done

echo ""
echo "================================================================"
echo "SECTION 5: hebert (HF)"
echo "================================================================"
for config in P0_baseline P1_add_weights P2_add_focal; do
    eval_hf "$CKPT/hebert/$config/best_model" \
            "$RES/hebert/$config" \
            "$GOLD_HEBERT" "$SIL_HEBERT"
done

echo ""
echo "================================================================"
echo "SECTION 6: mbert (HF)"
echo "================================================================"
for config in P0_baseline P1_add_weights P2_add_focal; do
    eval_hf "$CKPT/mbert/$config/best_model" \
            "$RES/mbert/$config" \
            "$GOLD_MBERT" "$SIL_MBERT"
done

echo ""
echo "================================================================"
echo "SECTION 7: xlmr (HF)"
echo "================================================================"
for config in P0_baseline P1_add_weights P2_add_focal; do
    eval_hf "$CKPT/xlmr/$config/best_model" \
            "$RES/xlmr/$config" \
            "$GOLD_XLMR" "$SIL_XLMR"
done

# ================================================================
# FINAL RESULTS TABLE
# ================================================================
echo ""
echo "================================================================"
echo "RESULTS TABLE"
echo "================================================================"

python3 - << 'PYEOF'
import json, os, glob

BASE = "/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/recipe_modification_extraction/recipe_project"
RES  = f"{BASE}/results"

rows = []

def read_result(path):
    try:
        with open(path) as f:
            d = json.load(f)
        f1 = d.get("entity_f1",       d.get("overall_f1",        d.get("f1",        None)))
        pr = d.get("entity_precision", d.get("overall_precision", d.get("precision", None)))
        re = d.get("entity_recall",    d.get("overall_recall",    d.get("recall",    None)))
        cl = d.get("ci_95_low",   d.get("ci_95", [None, None])[0] if isinstance(d.get("ci_95"), list) else None)
        ch = d.get("ci_95_high",  d.get("ci_95", [None, None])[1] if isinstance(d.get("ci_95"), list) else None)
        return f1, pr, re, cl, ch
    except:
        return None, None, None, None, None

# Scan results/model/config/gold|silver/evaluation_results.json
SKIP_CONFIGS = {"best_model"}
SKIP_PATTERN = r"checkpoint-\d+"

import re
for path in sorted(glob.glob(f"{RES}/**/evaluation_results.json", recursive=True)):
    rel   = os.path.relpath(path, RES)
    parts = rel.split(os.sep)
    if len(parts) < 4:
        continue
    model, config, split = parts[0], parts[1], parts[2]
    if config in SKIP_CONFIGS or re.match(r"checkpoint-\d+", config):
        continue
    f1, pr, re_, cl, ch = read_result(path)
    ci = f"[{cl:.3f},{ch:.3f}]" if cl is not None else ""
    rows.append((model, config, split, f1, pr, re_, ci))

MODEL_ORDER = ["mbert","hebert","alephbert","xlmr","dictabert","dictabert_large","dictabert_crf"]
rows.sort(key=lambda r: (
    MODEL_ORDER.index(r[0]) if r[0] in MODEL_ORDER else 99,
    r[1], r[2]
))

def f(v):
    return f"{v:.4f}" if isinstance(v, float) else ("N/A" if v is None else str(v))

print(f"\n{'Model':<22} {'Config':<34} {'Split':<7} {'F1':<8} {'Prec':<8} {'Rec':<8} {'95% CI'}")
print("=" * 105)
last = None
for model, config, split, f1, pr, re_, ci in rows:
    if model != last:
        if last: print()
        last = model
    print(f"{model:<22} {config:<34} {split:<7} {f(f1):<8} {f(pr):<8} {f(re_):<8} {ci}")

print(f"\nTotal: {len(rows)} rows")
PYEOF

echo ""
echo "=================================================="
echo "ALL DONE. End: $(date)"
echo "=================================================="
