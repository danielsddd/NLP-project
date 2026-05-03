#!/bin/bash
# =============================================================================
# eval_all_complete.sbatch
# MERGED: eval_all_models + eval_new_runs — covers every trained variant.
# Skips already-completed evals (idempotent — safe to re-run).
# Expected runtime: ~6-8h on a single GPU.
# =============================================================================

#SBATCH --job-name=eval_all_complete
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/%j_eval_all_complete.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/%j_eval_all_complete.err

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
echo "FULL EVALUATION — ALL MODELS + LAST NIGHT RUNS"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=================================================="

# ================================================================
# GOLD FILES (per tokenizer × scheme)
# ================================================================
GOLD_DIR="data/gold_validation"
GOLD_DICTABERT="data/gold_validation/gold_tokenized_dictabert_v2.jsonl"
GOLD_DICTABERT_IO="data/gold_validation/gold_tokenized_dictabert_io_v2.jsonl"
GOLD_DICTA_LARGE="data/gold_validation/gold_tokenized_dictabert_large_v2.jsonl"
GOLD_DICTA_LARGE_IO="data/gold_validation/gold_tokenized_dictabert_large_io_v2.jsonl"
GOLD_ALEPHBERT="data/gold_validation/gold_tokenized_alephbert_v2.jsonl"
GOLD_HEBERT="data/gold_validation/gold_tokenized_hebert_v2.jsonl"
GOLD_MBERT="data/gold_validation/gold_tokenized_mbert_v2.jsonl"
GOLD_XLMR="data/gold_validation/gold_tokenized_xlmr_v2.jsonl"

# Generate any missing gold tokenizations
for spec in \
    "dicta-il/dictabert-large|BIO|$GOLD_DICTA_LARGE" \
    "dicta-il/dictabert-large|IO|$GOLD_DICTA_LARGE_IO" \
    "dicta-il/dictabert|IO|$GOLD_DICTABERT_IO"; do
    IFS='|' read -r mdl scheme out <<< "$spec"
    if [ ! -s "$out" ]; then
        echo "Generating gold: $out"
        python scripts/local/prepare_gold_for_eval.py \
            --model "$mdl" --scheme "$scheme" --output "$out"
    fi
done

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

# ================================================================
# HELPER: eval a CRF checkpoint (best_model.pt)
# eval_crf <ckpt_dir> <result_dir> <gold_file> <silver_file> <model_name>
# ================================================================
eval_crf() {
    local ckpt_dir="$1"
    local result_dir="$2"
    local gold_file="$3"
    local silver_file="$4"
    local model_name="${5:-dicta-il/dictabert}"
    local name=$(basename "$ckpt_dir")

    if [ ! -f "$ckpt_dir/best_model.pt" ]; then
        echo "  [SKIP] $name — no best_model.pt"
        return
    fi
    echo ""
    echo "  [CRF] $name  (model=$model_name)"

    local gold_done=0
    if [ -f "$result_dir/gold/evaluation_results.json" ]; then
        if python3 -c "import json,sys; d=json.load(open('$result_dir/gold/evaluation_results.json')); sys.exit(0 if 'relaxed_f1' in d else 1)" 2>/dev/null; then
            echo "    ⏭  gold exists (relaxed_f1 present)"
            gold_done=1
        else
            echo "    ♻  gold exists but missing relaxed_f1 — re-running"
        fi
    fi

    if [ "$gold_done" -eq 0 ]; then
        mkdir -p "$result_dir/gold"
        python -m scripts.local.evaluate_crf \
            --ckpt-dir   "$ckpt_dir" \
            --test-file  "$gold_file" \
            --output-dir "$result_dir/gold" \
            --model-name "$model_name" 2>&1 | tee "$result_dir/gold_eval.log"
        echo "    ✅ gold done"
    fi

    local silver_done=0
    if [ -n "$silver_file" ] && [ -f "$silver_file" ]; then
        if [ -f "$result_dir/silver/evaluation_results.json" ]; then
            if python3 -c "import json,sys; d=json.load(open('$result_dir/silver/evaluation_results.json')); sys.exit(0 if 'relaxed_f1' in d else 1)" 2>/dev/null; then
                echo "    ⏭  silver exists (relaxed_f1 present)"
                silver_done=1
            else
                echo "    ♻  silver exists but missing relaxed_f1 — re-running"
            fi
        fi
        if [ "$silver_done" -eq 0 ]; then
            mkdir -p "$result_dir/silver"
            python -m scripts.local.evaluate_crf \
                --ckpt-dir   "$ckpt_dir" \
                --test-file  "$silver_file" \
                --output-dir "$result_dir/silver" \
                --model-name "$model_name" 2>&1 | tee "$result_dir/silver_eval.log"
            echo "    ✅ silver done"
        fi
    else
        echo "    ⚠  silver not found: $silver_file"
    fi
}

# ================================================================
# HELPER: eval a HuggingFace model (safetensors / pytorch_model.bin)
# eval_hf <model_dir> <result_dir> <gold_file> <silver_file>
# ================================================================
eval_hf() {
    local model_dir="$1"
    local result_dir="$2"
    local gold_file="$3"
    local silver_file="$4"
    local name=$(basename "$(dirname "$model_dir")")/$(basename "$model_dir")

    if [ ! -f "$model_dir/model.safetensors" ] \
       && [ ! -f "$model_dir/pytorch_model.bin" ]; then
        echo "  [SKIP] $name — no weights"
        return
    fi
    echo ""
    echo "  [HF]  $name"

    local hf_gold_done=0
    if [ -f "$result_dir/gold/evaluation_results.json" ]; then
        if python3 -c "import json,sys; d=json.load(open('$result_dir/gold/evaluation_results.json')); sys.exit(0 if 'relaxed_f1' in d else 1)" 2>/dev/null; then
            echo "    ⏭  gold exists (relaxed_f1 present)"
            hf_gold_done=1
        else
            echo "    ♻  gold exists but missing relaxed_f1 — re-running"
        fi
    fi
    if [ "$hf_gold_done" -eq 0 ]; then
        mkdir -p "$result_dir/gold"
        python -m src.evaluation.evaluate \
            --model-path "$model_dir" \
            --test-file  "$gold_file" \
            --output-dir "$result_dir/gold" \
            --bootstrap  2>&1 | tee "$result_dir/gold_eval.log"
        echo "    ✅ gold done"
    fi

    local hf_silver_done=0
    if [ -n "$silver_file" ] && [ -f "$silver_file" ]; then
        if [ -f "$result_dir/silver/evaluation_results.json" ]; then
            if python3 -c "import json,sys; d=json.load(open('$result_dir/silver/evaluation_results.json')); sys.exit(0 if 'relaxed_f1' in d else 1)" 2>/dev/null; then
                echo "    ⏭  silver exists (relaxed_f1 present)"
                hf_silver_done=1
            else
                echo "    ♻  silver exists but missing relaxed_f1 — re-running"
            fi
        fi
        if [ "$hf_silver_done" -eq 0 ]; then
            mkdir -p "$result_dir/silver"
            python -m src.evaluation.evaluate \
                --model-path "$model_dir" \
                --test-file  "$silver_file" \
                --output-dir "$result_dir/silver" \
                --bootstrap  2>&1 | tee "$result_dir/silver_eval.log"
            echo "    ✅ silver done"
        fi
    else
        echo "    ⚠  silver not found: $silver_file"
    fi
}

# ================================================================
# SECTION 1: mbert
# ================================================================
echo ""; echo "================================================================"
echo "SECTION 1: mbert"; echo "================================================================"
for cfg in P0_baseline P1_add_weights P2_add_focal; do
    eval_hf "$CKPT/mbert/$cfg/best_model" "$RES/mbert/$cfg" "$GOLD_MBERT" "$SIL_MBERT"
done

# ================================================================
# SECTION 2: hebert
# ================================================================
echo ""; echo "================================================================"
echo "SECTION 2: hebert"; echo "================================================================"
for cfg in P0_baseline P1_add_weights P2_add_focal; do
    eval_hf "$CKPT/hebert/$cfg/best_model" "$RES/hebert/$cfg" "$GOLD_HEBERT" "$SIL_HEBERT"
done

# ================================================================
# SECTION 3: alephbert
# ================================================================
echo ""; echo "================================================================"
echo "SECTION 3: alephbert"; echo "================================================================"
for cfg in P0_baseline P1_add_weights P2_add_focal; do
    eval_hf "$CKPT/alephbert/$cfg/best_model" "$RES/alephbert/$cfg" "$GOLD_ALEPHBERT" "data/processed_alephbert/test.jsonl"
done

# ================================================================
# SECTION 4: xlmr
# ================================================================
echo ""; echo "================================================================"
echo "SECTION 4: xlmr"; echo "================================================================"
for cfg in P0_baseline P1_add_weights P2_add_focal; do
    eval_hf "$CKPT/xlmr/$cfg/best_model" "$RES/xlmr/$cfg" "$GOLD_XLMR" "$SIL_XLMR"
done

# ================================================================
# SECTION 5: dictabert (HF, no CRF)
# ================================================================
echo ""; echo "================================================================"
echo "SECTION 5: dictabert (HF)"; echo "================================================================"
for cfg in P0_baseline P1_add_weights P2_add_focal; do
    eval_hf "$CKPT/dictabert/$cfg/best_model" "$RES/dictabert/$cfg" "$GOLD_DICTABERT" "$SIL_DICTA"
done

# ================================================================
# SECTION 6: dictabert_large (HF, no CRF)
# ================================================================
echo ""; echo "================================================================"
echo "SECTION 6: dictabert_large (HF)"; echo "================================================================"
for cfg in P0_baseline P1_add_weights P2_add_focal; do
    eval_hf "$CKPT/dictabert_large/$cfg/best_model" \
            "$RES/dictabert_large/$cfg" "$GOLD_DICTA_LARGE" "$SIL_LARGE"
done
eval_hf "$CKPT/dictabert_large/P3_add_downsample_2/best_model" \
        "$RES/dictabert_large/P3_add_downsample_2" "$GOLD_DICTA_LARGE" "$SIL_DS2"
eval_hf "$CKPT/dictabert_large/P3_add_downsample_4/best_model" \
        "$RES/dictabert_large/P3_add_downsample_4" "$GOLD_DICTA_LARGE" "$SIL_DS4"
# Large ablations
eval_hf "$RES/dictabert_large/A1b_uniform_weights/best_model" \
        "$RES/dictabert_large/A1b_uniform_weights" "$GOLD_DICTA_LARGE" "$SIL_LARGE"
eval_hf "$RES/dictabert_large/A2_downsample_2/best_model" \
        "$RES/dictabert_large/A2_downsample_2"     "$GOLD_DICTA_LARGE" "$SIL_DS2"
eval_hf "$RES/dictabert_large/A2_downsample_4/best_model" \
        "$RES/dictabert_large/A2_downsample_4"     "$GOLD_DICTA_LARGE" "$SIL_DS4"
eval_hf "$RES/dictabert_large/A5_data_25pct/best_model" \
        "$RES/dictabert_large/A5_data_25pct"       "$GOLD_DICTA_LARGE" "$SIL_F25"
eval_hf "$RES/dictabert_large/A5_data_50pct/best_model" \
        "$RES/dictabert_large/A5_data_50pct"       "$GOLD_DICTA_LARGE" "$SIL_F50"
eval_hf "$RES/dictabert_large/A5_data_75pct/best_model" \
        "$RES/dictabert_large/A5_data_75pct"       "$GOLD_DICTA_LARGE" "$SIL_F75"
eval_hf "$RES/dictabert_large/A6_no_enriched/best_model" \
        "$RES/dictabert_large/A6_no_enriched"      "$GOLD_DICTA_LARGE" "$SIL_NO_ENR"
eval_hf "$RES/dictabert_large/A7_io_scheme/best_model" \
        "$RES/dictabert_large/A7_io_scheme"        "$GOLD_DICTA_LARGE" "$SIL_IO"
eval_hf "$RES/dictabert_large/A8_unanimous/best_model" \
        "$RES/dictabert_large/A8_unanimous"        "$GOLD_DICTA_LARGE" "$SIL_UNANI"

# ================================================================
# SECTION 7: dictabert_crf — original P-series + ablations
# ================================================================
echo ""; echo "================================================================"
echo "SECTION 7: dictabert_crf — P-series + ablations"; echo "================================================================"
eval_crf "$CKPT/dictabert_crf/P0_crf_baseline"           "$RES/dictabert_crf/P0_crf_baseline"           "$GOLD_DICTABERT"    "$SIL_V2"
eval_crf "$CKPT/dictabert_crf/P1_crf_weighted"           "$RES/dictabert_crf/P1_crf_weighted"           "$GOLD_DICTABERT"    "$SIL_V2"
eval_crf "$CKPT/dictabert_crf/P2_crf_weighted"           "$RES/dictabert_crf/P2_crf_weighted"           "$GOLD_DICTABERT"    "$SIL_V2"
eval_crf "$CKPT/dictabert_crf/P3_crf_downsample_2"       "$RES/dictabert_crf/P3_crf_downsample_2"       "$GOLD_DICTABERT"    "$SIL_DS2"
eval_crf "$CKPT/dictabert_crf/P3_crf_downsample_4"       "$RES/dictabert_crf/P3_crf_downsample_4"       "$GOLD_DICTABERT"    "$SIL_DS4"
eval_crf "$CKPT/dictabert_crf/P4_crf_thread_aware"       "$RES/dictabert_crf/P4_crf_thread_aware"       "$GOLD_DICTABERT"    "$SIL_TA"
eval_crf "$CKPT/dictabert_crf/P5_crf_thread_no_enriched" "$RES/dictabert_crf/P5_crf_thread_no_enriched" "$GOLD_DICTABERT"    "$SIL_TA_NO_ENR"
# Ablations
eval_crf "$RES/dictabert_crf/A1b_crf_uniform_weights"    "$RES/dictabert_crf/A1b_crf_uniform_weights"   "$GOLD_DICTABERT"    "$SIL_V2"
eval_crf "$RES/dictabert_crf/A2_crf_downsample_2"        "$RES/dictabert_crf/A2_crf_downsample_2"       "$GOLD_DICTABERT"    "$SIL_DS2"
eval_crf "$RES/dictabert_crf/A2_crf_downsample_4"        "$RES/dictabert_crf/A2_crf_downsample_4"       "$GOLD_DICTABERT"    "$SIL_DS4"
eval_crf "$RES/dictabert_crf/A5_crf_data_25pct"          "$RES/dictabert_crf/A5_crf_data_25pct"         "$GOLD_DICTABERT"    "$SIL_F25"
eval_crf "$RES/dictabert_crf/A5_crf_data_50pct"          "$RES/dictabert_crf/A5_crf_data_50pct"         "$GOLD_DICTABERT"    "$SIL_F50"
eval_crf "$RES/dictabert_crf/A5_crf_data_75pct"          "$RES/dictabert_crf/A5_crf_data_75pct"         "$GOLD_DICTABERT"    "$SIL_F75"
eval_crf "$RES/dictabert_crf/A6_crf_no_enriched"         "$RES/dictabert_crf/A6_crf_no_enriched"        "$GOLD_DICTABERT"    "$SIL_NO_ENR"
eval_crf "$RES/dictabert_crf/A8_crf_unanimous"           "$RES/dictabert_crf/A8_crf_unanimous"          "$GOLD_DICTABERT"    "$SIL_UNANI"

# ================================================================
# SECTION 8: dictabert_crf — IO scheme variants (last night)
# Uses GOLD_DICTABERT_IO — correct tokenization for IO scheme
# ================================================================
echo ""; echo "================================================================"
echo "SECTION 8: dictabert_crf — IO + thread-aware (last night runs)"; echo "================================================================"

# Re-eval broken IO runs with correct gold file
rm -f "$RES/dictabert_crf/P10_crf_thread_aware_io/gold/evaluation_results.json"
rm -f "$RES/dictabert_crf/A7_crf_io_scheme/gold/evaluation_results.json"

eval_crf "$CKPT/dictabert_crf/P10_crf_thread_aware_io"   "$RES/dictabert_crf/P10_crf_thread_aware_io"   "$GOLD_DICTABERT_IO" "$SIL_TA_IO"
eval_crf "$RES/dictabert_crf/A7_crf_io_scheme"           "$RES/dictabert_crf/A7_crf_io_scheme"          "$GOLD_DICTABERT_IO" "$SIL_IO"

# Bug-fix re-runs
eval_crf "$CKPT/dictabert_crf/P4v2_thread_fixed"              "$RES/dictabert_crf/P4v2_thread_fixed"              "$GOLD_DICTABERT"    "$SIL_TA"
eval_crf "$CKPT/dictabert_crf/P5v2_thread_no_enriched_fixed"  "$RES/dictabert_crf/P5v2_thread_no_enriched_fixed"  "$GOLD_DICTABERT"    "$SIL_TA_NO_ENR"

# NEW: P10v3 (best model — IO gold)
eval_crf "$CKPT/dictabert_crf/P10v3_fixed_weights"       "$RES/dictabert_crf/P10v3_fixed_weights"       "$GOLD_DICTABERT_IO" "$SIL_TA_IO"

# NEW: P10v4 focal
eval_crf "$CKPT/dictabert_crf/P10v4_focal"               "$RES/dictabert_crf/P10v4_focal"               "$GOLD_DICTABERT_IO" "$SIL_TA_IO"

# NEW: P10v3 lr sweep
eval_crf "$CKPT/dictabert_crf/P10v3_lr1e5"               "$RES/dictabert_crf/P10v3_lr1e5"               "$GOLD_DICTABERT_IO" "$SIL_TA_IO"

# NEW: P4v3 focal (BIO)
eval_crf "$CKPT/dictabert_crf/P4v3_thread_focal"         "$RES/dictabert_crf/P4v3_thread_focal"         "$GOLD_DICTABERT"    "$SIL_TA"

# NEW: Seed sweep
eval_crf "$CKPT/dictabert_crf/P10v3_seed123"             "$RES/dictabert_crf/P10v3_seed123"             "$GOLD_DICTABERT_IO" "$SIL_TA_IO"
eval_crf "$CKPT/dictabert_crf/P10v3_seed2026"            "$RES/dictabert_crf/P10v3_seed2026"            "$GOLD_DICTABERT_IO" "$SIL_TA_IO"
eval_crf "$CKPT/dictabert_crf/P10v3_seed7777"            "$RES/dictabert_crf/P10v3_seed7777"            "$GOLD_DICTABERT_IO" "$SIL_TA_IO"

# ================================================================
# SECTION 9: dictabert_large_crf — all variants (last night)
# ================================================================
echo ""; echo "================================================================"
echo "SECTION 9: dictabert_large_crf — all variants"; echo "================================================================"

eval_crf "$CKPT/dictabert_large_crf/P2_crf_weighted"          "$RES/dictabert_large_crf/P2_crf_weighted"          "$GOLD_DICTA_LARGE"    "$SIL_V2"        "dicta-il/dictabert-large"
eval_crf "$CKPT/dictabert_large_crf/P4_large_crf_thread_bio"  "$RES/dictabert_large_crf/P4_large_crf_thread_bio"  "$GOLD_DICTA_LARGE"    "$SIL_TA"        "dicta-il/dictabert-large"
eval_crf "$CKPT/dictabert_large_crf/P10_large_crf_thread_io"  "$RES/dictabert_large_crf/P10_large_crf_thread_io"  "$GOLD_DICTA_LARGE_IO" "$SIL_TA_IO"     "dicta-il/dictabert-large"
eval_crf "$CKPT/dictabert_large_crf/P10v4_large_focal"        "$RES/dictabert_large_crf/P10v4_large_focal"        "$GOLD_DICTA_LARGE_IO" "$SIL_TA_IO"     "dicta-il/dictabert-large"
eval_crf "$CKPT/dictabert_large_crf/P10_large_lr3e5"          "$RES/dictabert_large_crf/P10_large_lr3e5"          "$GOLD_DICTA_LARGE_IO" "$SIL_TA_IO"     "dicta-il/dictabert-large"

# Large+CRF ablations (A1a / A1c from pending jobs 305177 / 305178)
eval_crf "$CKPT/dictabert_large_crf/A1a_no_weights"           "$RES/dictabert_large_crf/A1a_no_weights"           "$GOLD_DICTA_LARGE"    "$SIL_V2"        "dicta-il/dictabert-large"
eval_crf "$CKPT/dictabert_large_crf/A1c_weights_only"         "$RES/dictabert_large_crf/A1c_weights_only"         "$GOLD_DICTA_LARGE"    "$SIL_V2"        "dicta-il/dictabert-large"

# ================================================================
# SECTION 10: Binary classification from spans
# ================================================================
echo ""; echo "================================================================"
echo "SECTION 10: Binary classification metrics"; echo "================================================================"
python scripts/local/compute_binary_from_spans.py --batch

# ================================================================
# FINAL RESULTS TABLE
# ================================================================
echo ""; echo "================================================================"
echo "RESULTS TABLE — GOLD SET"; echo "================================================================"

python3 - << 'PYEOF'
import json, os, glob, re

BASE = os.getcwd()
RES  = os.path.join(BASE, "results")

MODEL_ORDER = ["mbert", "hebert", "alephbert", "xlmr",
               "dictabert", "dictabert_large",
               "dictabert_crf", "dictabert_large_crf"]

SKIP_CONFIGS = {"best_model"}

def read_result(path):
    try:
        d = json.load(open(path))
        f1      = d.get("entity_f1",       d.get("overall_f1", d.get("f1")))
        pr      = d.get("entity_precision", d.get("precision"))
        re      = d.get("entity_recall",    d.get("recall"))
        relaxed = d.get("relaxed_f1")
        r_lo    = d.get("relaxed_ci_95_low")
        r_hi    = d.get("relaxed_ci_95_high")
        return f1, pr, re, relaxed, r_lo, r_hi
    except:
        return None, None, None, None, None, None

rows = []
for path in sorted(glob.glob(f"{RES}/**/gold/evaluation_results.json", recursive=True)):
    rel   = os.path.relpath(path, RES)
    parts = rel.split(os.sep)
    if len(parts) < 3:
        continue
    model, config = parts[0], parts[1]
    if config in SKIP_CONFIGS or re.match(r"checkpoint-\d+", config):
        continue
    f1, pr, re_, relaxed, r_lo, r_hi = read_result(path)
    if f1 is None:
        continue
    rows.append((model, config, f1, pr, re_, relaxed, r_lo, r_hi))

rows.sort(key=lambda r: (
    MODEL_ORDER.index(r[0]) if r[0] in MODEL_ORDER else 99,
    -(r[3] or r[2] or 0)
))

def fmt(v):
    return f"{v*100:.2f}%" if isinstance(v, float) else "—"

print(f"\n{'Model':<22} {'Config':<36} {'Exact F1':>9} {'Relaxed F1':>11} {'Relaxed CI':>22}")
print("=" * 110)
last_model = None
for model, config, f1, pr, re_, relaxed, r_lo, r_hi in rows:
    if model != last_model:
        if last_model is not None:
            print()
        last_model = model
    ci_str = f"[{r_lo:.3f}, {r_hi:.3f}]" if r_lo is not None else "—"
    r_str  = fmt(relaxed) if relaxed else "pending"
    print(f"{model:<22} {config:<36} {fmt(f1):>9} {r_str:>11} {ci_str:>22}")

print(f"\n{'─'*110}")
if rows:
    best = max(rows, key=lambda r: r[3] or r[2] or 0)
    print(f"★  Best: {best[0]}/{best[1]}  →  Exact={fmt(best[2])}  Relaxed={fmt(best[3])}")
print(f"Total variants with gold results: {len(rows)}")
PYEOF

echo ""
echo "================================================================"
echo "RESULTS TABLE — SILVER TEST SET"
echo "================================================================"

python3 - << 'PYEOF'
import json, os, glob, re

RES = os.path.join(os.getcwd(), "results")
MODEL_ORDER = ["mbert","hebert","alephbert","xlmr",
               "dictabert","dictabert_large",
               "dictabert_crf","dictabert_large_crf"]
SKIP = {"best_model"}

def read_result(path):
    try:
        d = json.load(open(path))
        f1 = d.get("entity_f1", d.get("f1"))
        pr = d.get("entity_precision", d.get("precision"))
        re = d.get("entity_recall", d.get("recall"))
        return f1, pr, re
    except:
        return None, None, None

rows = []
for path in sorted(glob.glob(f"{RES}/**/silver/evaluation_results.json", recursive=True)):
    rel   = os.path.relpath(path, RES)
    parts = rel.split(os.sep)
    if len(parts) < 3: continue
    model, config = parts[0], parts[1]
    if config in SKIP or re.match(r"checkpoint-\d+", config): continue
    f1, pr, re_ = read_result(path)
    if f1 is None: continue
    rows.append((model, config, f1, pr, re_))

rows.sort(key=lambda r: (
    MODEL_ORDER.index(r[0]) if r[0] in MODEL_ORDER else 99,
    -(r[2] or 0)
))

def fmt(v):
    return f"{v*100:.2f}%" if isinstance(v, float) else "—"

print(f"\n{'Model':<22} {'Config':<38} {'Silver F1':>10} {'Prec':>9} {'Recall':>9}")
print("=" * 95)
last = None
for model, config, f1, pr, re_ in rows:
    if model != last:
        if last: print()
        last = model
    print(f"{model:<22} {config:<38} {fmt(f1):>10} {fmt(pr):>9} {fmt(re_):>9}")
print(f"\nTotal: {len(rows)}")
PYEOF

echo ""
echo "=================================================="
echo "ALL DONE. End: $(date)"
echo "=================================================="
