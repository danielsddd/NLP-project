#!/bin/bash
# =============================================================================
# eval_p2_seeds.sbatch
# Evaluates seed stability runs for P1_crf_weighted (seeds 7777, 2026, 123).
# Mirrors eval_all_complete.sbatch exactly — same eval_crf helper,
# same gold file, same silver file, same module path.
#
# Submit AFTER all three P2_crf_seed*.sbatch jobs complete.
# Expected runtime: ~30-45min on a single GPU.
# =============================================================================

#SBATCH --job-name=eval_p2_seeds
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/%j_eval_p2_seeds.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/%j_eval_p2_seeds.err

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
echo "P1 SEED STABILITY EVALUATION"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=================================================="

# Gold/silver files — same as used for P1_crf_weighted in eval_all_complete
GOLD_DICTABERT="data/gold_validation/gold_tokenized_dictabert_v2.jsonl"
SIL_V2="data/processed_v2/test.jsonl"

CKPT="models/checkpoints"
RES="results"

# ================================================================
# HELPER — exact copy from eval_all_complete.sbatch
# ================================================================
eval_crf() {
    local ckpt_dir="$1"
    local result_dir="$2"
    local gold_file="$3"
    local silver_file="$4"
    local model_name="${5:-dicta-il/dictabert}"
    local name=$(basename "$ckpt_dir")

    if [ ! -f "$ckpt_dir/best_model.pt" ]; then
        echo "  [SKIP] $name — no best_model.pt (training still running?)"
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
# P1 SEED STABILITY — seeds 7777, 2026, 123
# (seed 42 = P1_crf_weighted, already evaluated in eval_all_complete)
# ================================================================
echo ""
echo "================================================================"
echo "P1_crf_weighted — seed stability (7777 / 2026 / 123)"
echo "================================================================"

eval_crf "$CKPT/dictabert_crf/P2_crf_seed7777" \
         "$RES/dictabert_crf/P2_crf_seed7777"  \
         "$GOLD_DICTABERT" "$SIL_V2"

eval_crf "$CKPT/dictabert_crf/P2_crf_seed2026" \
         "$RES/dictabert_crf/P2_crf_seed2026"  \
         "$GOLD_DICTABERT" "$SIL_V2"

eval_crf "$CKPT/dictabert_crf/P2_crf_seed123"  \
         "$RES/dictabert_crf/P2_crf_seed123"   \
         "$GOLD_DICTABERT" "$SIL_V2"

# ================================================================
# SUMMARY TABLE — all four P1 seeds side by side
# ================================================================
echo ""
echo "================================================================"
echo "SEED STABILITY SUMMARY — P1_crf_weighted"
echo "================================================================"

python3 - << 'PYEOF'
import json, os

RES = os.path.join(os.getcwd(), "results/dictabert_crf")

# seed 42 is P1_crf_weighted; seeds 7777/2026/123 are new
RUNS = [
    ("42  (ref)", "P1_crf_weighted"),
    ("7777",      "P2_crf_seed7777"),
    ("2026",      "P2_crf_seed2026"),
    ("123",       "P2_crf_seed123"),
]

def fmt(v):
    return f"{v*100:.2f}%" if isinstance(v, float) else "—"

def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}

print(f"\n{'Seed':<12}  {'GoldExact':>10}  {'GoldRelaxed':>12}  {'Relaxed CI':>22}  {'SilvExact':>10}")
print("─" * 75)

for seed_label, config in RUNS:
    gd = load(f"{RES}/{config}/gold/evaluation_results.json")
    sd = load(f"{RES}/{config}/silver/evaluation_results.json")

    ge  = gd.get("entity_f1")
    gr  = gd.get("relaxed_f1")
    lo  = gd.get("relaxed_ci_95_low")
    hi  = gd.get("relaxed_ci_95_high")
    se  = sd.get("entity_f1")

    ci  = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "—"
    print(f"{seed_label:<12}  {fmt(ge):>10}  {fmt(gr):>12}  {ci:>22}  {fmt(se):>10}")

print()
PYEOF

echo ""
echo "=================================================="
echo "DONE. End: $(date)"
echo "=================================================="
