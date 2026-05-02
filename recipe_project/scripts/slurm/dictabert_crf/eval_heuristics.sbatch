#!/bin/bash
# =============================================================================
# eval_heuristics.sbatch
# Evaluates P10v3_fixed_weights with and without post-processing heuristics.
# Reports raw vs filtered F1 on both gold and silver test sets.
# Run from recipe_project root.
# =============================================================================

#SBATCH --job-name=heuristics_eval
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_xp|geforce_rtx_2080"
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_heuristics_eval.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf/%j_heuristics_eval.err

USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction/recipe_project"
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
cd $PROJECT_DIR

MODEL="models/checkpoints/dictabert_crf/P10v3_fixed_weights"
GOLD="data/gold_validation/gold_final.jsonl"
SILVER_TEST="data/processed_thread_aware_io/test.jsonl"
VAL="data/processed_thread_aware_io/val.jsonl"

echo "=================================================="
echo "HEURISTICS EVALUATION"
echo "Model:  $MODEL"
echo "Job ID: $SLURM_JOB_ID  |  Node: $(hostname)"
echo "Start:  $(date)"
echo "=================================================="

# ── Step 0: Sanity check —validate model and data exist ───────────────────────
for path in "$MODEL" "$GOLD" "$SILVER_TEST" "$VAL"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: Missing: $path"
        exit 1
    fi
done

# ── Step 1: Tune heuristics on SILVER VAL (no gold leakage) ──────────────────
echo ""
echo "=== STEP 1: Tuning on silver val (no gold leakage) ==="

# Default config (all passes on, window=4)
python -m src.postprocessing.eval_with_heuristics \
    --model-path "$MODEL" \
    --test-file  "$VAL" \
    --output-dir "results/heuristics/P10v3/val_default" \
    --scheme IO \
    --quantity-window 4 \
    --bootstrap 2>&1 | tee results/heuristics/P10v3/val_default.log

# Ablate: quantity filter only (disable boundary trim + contiguity)
python -m src.postprocessing.eval_with_heuristics \
    --model-path "$MODEL" \
    --test-file  "$VAL" \
    --output-dir "results/heuristics/P10v3/val_qty_only" \
    --scheme IO \
    --no-contiguity \
    --no-boundary-trim \
    --quantity-window 4 2>&1 | tee results/heuristics/P10v3/val_qty_only.log

# Ablate: boundary trim only
python -m src.postprocessing.eval_with_heuristics \
    --model-path "$MODEL" \
    --test-file  "$VAL" \
    --output-dir "results/heuristics/P10v3/val_trim_only" \
    --scheme IO \
    --no-contiguity \
    --no-quantity-filter 2>&1 | tee results/heuristics/P10v3/val_trim_only.log

# Ablate: wider quantity window
python -m src.postprocessing.eval_with_heuristics \
    --model-path "$MODEL" \
    --test-file  "$VAL" \
    --output-dir "results/heuristics/P10v3/val_qty_window8" \
    --scheme IO \
    --quantity-window 8 2>&1 | tee results/heuristics/P10v3/val_qty_window8.log

echo ""
echo "=== VAL RESULTS SUMMARY ==="
python3 - <<'PYEOF'
import json, glob, os

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'; BOLD='\033[1m'
base = "results/heuristics/P10v3"

for config_dir in sorted(glob.glob(f"{base}/val_*")):
    name = os.path.basename(config_dir)
    raw_f  = f"{config_dir}/raw_evaluation_results.json"
    heur_f = f"{config_dir}/heuristic_evaluation_results.json"
    if not os.path.exists(raw_f): continue
    raw  = json.load(open(raw_f))
    heur = json.load(open(heur_f)) if os.path.exists(heur_f) else {}
    df1  = (heur.get('f1',0) - raw.get('f1',0))
    dp   = (heur.get('precision',0) - raw.get('precision',0))
    dr   = (heur.get('recall',0) - raw.get('recall',0))
    col  = GREEN if df1 > 0 else (RED if df1 < 0 else NC)
    print(f"  {BOLD}{name:<25}{NC}  raw F1={raw.get('f1','—'):.4f}  "
          f"→  heur F1={heur.get('f1','—'):.4f}  "
          f"{col}Δ={df1:+.4f}{NC}  ΔP={dp:+.4f}  ΔR={dr:+.4f}")
PYEOF

# ── Step 2: Apply best config to SILVER TEST ──────────────────────────────────
echo ""
echo "=== STEP 2: Silver test set (1,518 examples, 137 spans) ==="
python -m src.postprocessing.eval_with_heuristics \
    --model-path "$MODEL" \
    --test-file  "$SILVER_TEST" \
    --output-dir "results/heuristics/P10v3/silver_default" \
    --scheme IO \
    --quantity-window 4 \
    --bootstrap 2>&1 | tee results/heuristics/P10v3/silver_default.log

# ── Step 3: Apply to GOLD SET ─────────────────────────────────────────────────
echo ""
echo "=== STEP 3: Gold set evaluation (496 examples, 27 spans) ==="
python -m src.postprocessing.eval_with_heuristics \
    --model-path "$MODEL" \
    --test-file  "$GOLD" \
    --output-dir "results/heuristics/P10v3/gold_default" \
    --scheme IO \
    --quantity-window 4 \
    --bootstrap 2>&1 | tee results/heuristics/P10v3/gold_default.log

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "FINAL SUMMARY"
echo "=================================================="
python3 - <<'PYEOF'
import json, os

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'; BOLD='\033[1m'

sets = [
    ("Val (silver, 1510ex)",   "results/heuristics/P10v3/val_default"),
    ("Test (silver, 1518ex)",  "results/heuristics/P10v3/silver_default"),
    ("Gold (human, 496ex)",    "results/heuristics/P10v3/gold_default"),
]

print(f"\n  {BOLD}{'Set':<28} {'Raw F1':>8} {'Heur F1':>8} {'ΔF1':>8} {'ΔPrec':>8} {'ΔRecall':>8}{NC}")
print("  " + "─"*68)
for name, d in sets:
    rf = f"{d}/raw_evaluation_results.json"
    hf = f"{d}/heuristic_evaluation_results.json"
    if not os.path.exists(rf):
        print(f"  {name:<28} {'—':>8}")
        continue
    raw  = json.load(open(rf))
    heur = json.load(open(hf)) if os.path.exists(hf) else raw
    df1 = heur['f1'] - raw['f1']
    dp  = heur['precision'] - raw['precision']
    dr  = heur['recall'] - raw['recall']
    col = GREEN if df1 > 0.005 else (RED if df1 < -0.005 else NC)
    print(f"  {name:<28} {raw['f1']:>8.4f} {heur['f1']:>8.4f} "
          f"{col}{df1:>+8.4f}{NC} {dp:>+8.4f} {dr:>+8.4f}")
print()
PYEOF

echo "Results in: results/heuristics/P10v3/"
echo "End: $(date)"
echo "=================================================="
