#!/bin/bash
# =============================================================================
# quick_check.sh  —  fast status check while eval_new_runs.sbatch is running
#
# Usage (from recipe_project root):
#   bash scripts/local/quick_check.sh
#   bash scripts/local/quick_check.sh --tail 305412   # tail a specific job log
#   bash scripts/local/quick_check.sh --dup           # check 305156 vs 305160
# =============================================================================

LOG_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output"
RESULTS_BASE="results"
CKPT_BASE="models/checkpoints"

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ─── Float colour via python (avoids awk locale issues entirely) ───────────────
f1_colour() {
    local f1="$1"
    if [[ "$f1" == "" || "$f1" == "—" ]]; then echo -e "${YELLOW}"; return; fi
    python3 -c "
v=float('$f1')
if   v >= 0.50: print('\033[0;32m')
elif v >= 0.35: print('\033[0;36m')
else:           print('\033[0;31m')
" 2>/dev/null || echo -e "${NC}"
}

# ─── Parse args ───────────────────────────────────────────────────────────────
TAIL_JOB=""
CHECK_DUP=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tail) TAIL_JOB="$2"; shift 2 ;;
        --dup)  CHECK_DUP=1;   shift   ;;
        *)      shift ;;
    esac
done

# ─── --tail MODE ──────────────────────────────────────────────────────────────
if [[ -n "$TAIL_JOB" ]]; then
    log=$(find "$LOG_BASE" -name "${TAIL_JOB}_*.out" 2>/dev/null | head -1)
    [[ -z "$log" ]] && log=$(find "$LOG_BASE" -name "${TAIL_JOB}.out" 2>/dev/null | head -1)
    if [[ -n "$log" ]]; then
        echo -e "${CYAN}Tailing: $log${NC}"
        tail -60 "$log"
    else
        echo -e "${RED}No log found for job $TAIL_JOB under $LOG_BASE${NC}"
    fi
    exit 0
fi

# ─── --dup MODE ───────────────────────────────────────────────────────────────
if [[ $CHECK_DUP -eq 1 ]]; then
    echo -e "${BOLD}${CYAN}=== Checking 305156 vs 305160 (are they duplicates?) ===${NC}"
    for jid in 305156 305160; do
        log=$(find "$LOG_BASE" -name "${jid}_*.out" 2>/dev/null | head -1)
        if [[ -z "$log" ]]; then
            echo -e "  ${YELLOW}[$jid] Log not found${NC}"
            continue
        fi
        echo -e "\n  ${CYAN}[$jid] $log${NC}"
        echo "  --- SBATCH header ---"
        head -40 "$log" | grep -iE 'VARIANT|OUTPUT|MODEL|Job.ID|Start|job.name' \
            | head -8 | sed 's/^/    /'
        echo "  --- Python call ---"
        grep -m5 -E 'python -m|train_joint|train_student|evaluate' "$log" 2>/dev/null \
            | sed 's/^/    /'
        echo "  --- output-dir line ---"
        grep -m3 -E 'output.dir|OUTPUT|Saving' "$log" 2>/dev/null | sed 's/^/    /'
        echo "  --- Last 5 lines ---"
        tail -5 "$log" | sed 's/^/    /'
    done
    echo
    v1=$(find "$LOG_BASE" -name "305156_*.out" 2>/dev/null | head -1 | xargs basename 2>/dev/null)
    v2=$(find "$LOG_BASE" -name "305160_*.out" 2>/dev/null | head -1 | xargs basename 2>/dev/null)
    echo -e "  ${BOLD}Filenames:${NC}"
    echo -e "    305156 → ${CYAN}${v1:-NOT FOUND}${NC}"
    echo -e "    305160 → ${CYAN}${v2:-NOT FOUND}${NC}"
    if [[ "$v1" == "$v2" && -n "$v1" ]]; then
        echo -e "  ${RED}★ CONFIRMED DUPLICATES — same filename/variant!${NC}"
        echo -e "  ${YELLOW}  Recommendation: scancel 305160${NC}"
    else
        echo -e "  ${GREEN}★ Different filenames — likely NOT duplicates.${NC}"
    fi
    exit 0
fi

# ─── MAIN STATUS CHECK ────────────────────────────────────────────────────────
echo -e "\n${BOLD}${CYAN}════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  QUICK RESULTS CHECK — $(date '+%H:%M:%S')${NC}"
echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════${NC}"

# ── [1] evaluation_results.json ───────────────────────────────────────────────
echo -e "\n${BOLD}[1] evaluation_results.json  (eval job output — most reliable)${NC}"
echo    "─────────────────────────────────────────────────────────────────"

found_any=0
while IFS= read -r json_path; do
    rel="${json_path#$RESULTS_BASE/}"
    model=$(echo "$rel" | cut -d'/' -f1)
    variant=$(echo "$rel" | cut -d'/' -f2)
    split=$(echo "$rel" | cut -d'/' -f3)

    read -r f1 precision recall < <(python3 - "$json_path" <<'PYEOF'
import json, sys
path = sys.argv[1]
try:
    d = json.load(open(path))
    f1 = prec = rec = None
    for k in ['f1','span_f1','micro_f1','overall_f1','eval_f1']:
        if k in d and d[k] is not None:
            f1 = float(d[k]); break
    if f1 is None and 'overall' in d:
        for k in ['f1','f1-score']:
            if k in d.get('overall',{}):
                f1 = float(d['overall'][k]); break
    for k in ['precision','span_precision']:
        if k in d and d[k] is not None:
            prec = float(d[k]); break
    for k in ['recall','span_recall']:
        if k in d and d[k] is not None:
            rec = float(d[k]); break
    f1s   = f"{f1:.4f}"   if f1   is not None else "—"
    precs = f"{prec:.4f}" if prec is not None else "—"
    recs  = f"{rec:.4f}"  if rec  is not None else "—"
    print(f"{f1s} {precs} {recs}")
except Exception as e:
    print("— — —")
PYEOF
)

    col=$(f1_colour "$f1")
    printf "  %-32s %-30s %-8s ${col}%8s${NC}  prec=%-8s  rec=%s\n" \
        "$model" "$variant" "$split" "$f1" "$precision" "$recall"
    found_any=1
done < <(find "$RESULTS_BASE" -name "evaluation_results.json" 2>/dev/null | sort)

[[ $found_any -eq 0 ]] && echo -e "  ${YELLOW}None found yet — eval job 305412 still running.${NC}"

# ── [2] training_summary.json ─────────────────────────────────────────────────
echo -e "\n${BOLD}[2] training_summary.json  (val F1 from training — ranked)${NC}"
echo    "─────────────────────────────────────────────────────────────────"

python3 - "$CKPT_BASE" <<'PYEOF'
import json, sys
from pathlib import Path

GREEN = '\033[0;32m'; CYAN = '\033[0;36m'; RED = '\033[0;31m'
YELLOW = '\033[0;33m'; BOLD = '\033[1m'; DIM = '\033[2m'; NC = '\033[0m'

ckpt_base = Path(sys.argv[1])
seen = {}

for json_path in (list(ckpt_base.rglob("training_summary.json")) +
                  list(ckpt_base.rglob("all_results.json"))):
    parts = json_path.relative_to(ckpt_base).parts
    if len(parts) < 2:
        continue
    model, variant = parts[0], parts[1]
    key = (model, variant)
    if key in seen:
        continue  # first file wins (training_summary > all_results)
    try:
        d = json.load(open(json_path))
    except Exception:
        continue
    f1 = None
    for k in ['best_val_f1','best_f1','eval_f1','f1','span_f1','val_f1','final_f1']:
        if k in d and d[k] is not None:
            f1 = float(d[k]); break
    if f1 is None:
        for nk in ['best_metrics','eval_metrics','metrics']:
            if isinstance(d.get(nk), dict):
                for k in ['f1','eval_f1','span_f1']:
                    if d[nk].get(k) is not None:
                        f1 = float(d[nk][k]); break
            if f1 is not None:
                break
    seen[key] = f1

ranked = sorted(seen.items(), key=lambda x: (x[1] is None, -(x[1] or -1)))

for (model, variant), f1 in ranked:
    f1_str = f"{f1:.4f}" if f1 is not None else "—"
    if f1 is None:   col = YELLOW
    elif f1 >= 0.50: col = GREEN
    elif f1 >= 0.35: col = CYAN
    else:            col = RED
    print(f"  {model:<32} {variant:<34} {col}{f1_str:>8}{NC}")

above50 = sum(1 for _,f in ranked if f and f >= 0.50)
above35 = sum(1 for _,f in ranked if f and 0.35 <= f < 0.50)
below35 = sum(1 for _,f in ranked if f and f < 0.35)
nodata  = sum(1 for _,f in ranked if f is None)
print(f"\n  {DIM}Total={len(ranked)}  |  "
      f"{GREEN}>=0.50: {above50}{NC}{DIM}  |  "
      f"{CYAN}0.35-0.50: {above35}{NC}{DIM}  |  "
      f"{RED}<0.35: {below35}{NC}{DIM}  |  "
      f"no data: {nodata}{NC}")
PYEOF

# ── [3] Recent SLURM .out logs ────────────────────────────────────────────────
echo -e "\n${BOLD}[3] Recent SLURM .out logs (modified in last 8h)${NC}"
echo    "─────────────────────────────────────────────────────────────────"

found_any=0
while IFS= read -r log; do
    find "$log" -mmin -480 2>/dev/null | grep -q . || continue

    stem=$(basename "$log" .out)
    job_id=$(echo "$stem" | grep -oE '^[0-9]+')
    short_name="${stem#${job_id}_}"
    model=$(basename "$(dirname "$log")")

    # Extract LAST F1 from log
    f1_val=$(grep -oiE '"f1"\s*:\s*[0-9]+\.[0-9]+' "$log" 2>/dev/null \
             | grep -oE '[0-9]+\.[0-9]+$' | tail -1)
    if [[ -z "$f1_val" ]]; then
        f1_val=$(grep -oiE '(f1|span_f1|eval_f1)\s*[=:]\s*[0-9]+\.[0-9]+' "$log" 2>/dev/null \
                 | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    fi

    status="${YELLOW}running${NC}"
    grep -qE 'SUCCESS|ALL EVALUATIONS COMPLETE|Training complete|Saved best model' "$log" 2>/dev/null \
        && status="${GREEN}done${NC}"
    grep -qE 'FAILED|Traceback|RuntimeError|exit code: [^0]' "$log" 2>/dev/null \
        && status="${RED}FAILED${NC}"

    col=$(f1_colour "${f1_val}")
    printf "  [%-6s] %-28s %-30s ${col}%8s${NC}  %b\n" \
        "${job_id:-?}" "$model" "$short_name" "${f1_val:-—}" "$status"
    found_any=1
done < <(find "$LOG_BASE" -name "*.out" 2>/dev/null | sort -r)

[[ $found_any -eq 0 ]] && echo -e "  ${YELLOW}No recent log files found.${NC}"

# ── [4] Live squeue ───────────────────────────────────────────────────────────
echo -e "\n${BOLD}[4] Live squeue${NC}"
echo    "─────────────────────────────────────────────────────────────────"
squeue -u "$USER" -o "%.8i %.20j %.2t %.10M %R" 2>/dev/null \
    || echo -e "  ${DIM}(squeue not available — run on the cluster)${NC}"

# ── [5] Analysis ──────────────────────────────────────────────────────────────
echo -e "\n${BOLD}[5] Key findings${NC}"
echo    "─────────────────────────────────────────────────────────────────"
python3 - "$CKPT_BASE" <<'PYEOF'
import json, sys
from pathlib import Path

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

ckpt_base = Path(sys.argv[1])
seen = {}

for json_path in (list(ckpt_base.rglob("training_summary.json")) +
                  list(ckpt_base.rglob("all_results.json"))):
    parts = json_path.relative_to(ckpt_base).parts
    if len(parts) < 2: continue
    model, variant = parts[0], parts[1]
    key = (model, variant)
    if key in seen: continue
    try: d = json.load(open(json_path))
    except: continue
    f1 = None
    for k in ['best_val_f1','best_f1','eval_f1','f1','span_f1','val_f1','final_f1']:
        if k in d and d[k] is not None: f1 = float(d[k]); break
    if f1 is None:
        for nk in ['best_metrics','eval_metrics','metrics']:
            if isinstance(d.get(nk), dict):
                for k in ['f1','eval_f1','span_f1']:
                    if d[nk].get(k) is not None: f1 = float(d[nk][k]); break
            if f1 is not None: break
    seen[key] = f1

if not seen:
    print("  No data."); sys.exit(0)

ranked = sorted(seen.items(), key=lambda x: (x[1] is None, -(x[1] or -1)))
best = ranked[0]
f1_str = f"{best[1]:.4f}" if best[1] else "—"
col = GREEN if best[1] and best[1] >= 0.50 else CYAN
print(f"\n  {BOLD}Best (val F1):{NC}  {col}★ {best[0][0]}/{best[0][1]}  →  {f1_str}{NC}")

above50 = [(k,v) for k,v in ranked if v and v >= 0.50]
above35 = [(k,v) for k,v in ranked if v and 0.35 <= v < 0.50]
below35 = [(k,v) for k,v in ranked if v and v < 0.35]

if above50:
    print(f"\n  {GREEN}Above 0.50 target:{NC}")
    for (m,v), f1 in above50:
        print(f"    {GREEN}✓ {m}/{v}  →  {f1:.4f}{NC}")

if above35:
    print(f"\n  {CYAN}Close (0.35–0.50) — top 5:{NC}")
    for (m,v), f1 in above35[:5]:
        print(f"    {CYAN}● {m}/{v}  →  {f1:.4f}{NC}")

if below35:
    print(f"\n  {RED}Struggling (<0.35) — count: {len(below35)}{NC}")
    for (m,v), f1 in below35[:3]:
        print(f"    {RED}▼ {m}/{v}  →  {f1:.4f}{NC}")

# Seed stability check
p10v3_seeds = [(k,v) for k,v in ranked if 'P10v3' in k[1] and 'seed' in k[1]]
if p10v3_seeds:
    vals = [v for _,v in p10v3_seeds if v]
    spread = max(vals) - min(vals) if len(vals) > 1 else 0
    print(f"\n  {BOLD}P10v3 seed stability:{NC}")
    for (m,v_), f1 in p10v3_seeds:
        print(f"    {m}/{v_}  →  {f1:.4f if f1 else '—'}")
    if vals:
        print(f"    Spread: {spread:.4f}  Mean: {sum(vals)/len(vals):.4f}")

print()
PYEOF

echo -e "${BOLD}Useful commands:${NC}"
echo -e "  Tail eval job:      ${CYAN}bash scripts/local/quick_check.sh --tail 305412${NC}"
echo -e "  Check 305156/160:   ${CYAN}bash scripts/local/quick_check.sh --dup${NC}"
echo -e "  Full ranked table:  ${CYAN}python scripts/local/check_results.py${NC}"
echo