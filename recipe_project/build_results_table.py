import json
import os
import glob
import re

MY_BASE = "/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/recipe_modification_extraction/recipe_project"

rows = []

def add_row(model, config, split, f1, precision, recall, ci_low=None, ci_high=None, source=""):
    rows.append({
        "model": model,
        "confi        "split": split,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "CI": f"[{ci_low:.4f},{ci_high:.4f}]" if ci_low is not None else "N/A",
        "source": source,
    })

# -----------------------------------------------
# SOURCE 1: evaluation_results.json (gold/silver test eval — authoritative)
# Pattern: results/<model>/<config>/gold/evaluation_results.json
#          results/<model>/<config>/silver/evaluation_results.json
# -----------------------------------------------
for path in sorted(glob.glob(f"{MY_BASE}/results/**/evaluation_results.json", recursive=True)):
    rel = os.path.relpath(path, f"{MY_BASE}/results")
    parts = rel.split(os.sep)
    # parts: [model, config, gold|silver, evaluation_results.json]
    if len(parts) < 4:
        continue
    model  = parts[0]
    config = parts[1]
    split  = parts[2]  # gold or silver

    # Skip checkpoint dirs — we only want named configs (P0, P1, A1b, best_model etc.)
    if re.match(r'^checkpoint-\d+$', config):
        continue

    try:
        with open(path) as f:
            d = json.load(f)
        f1        = d.get("entity_f1",        d.get("overall_f1",   d.get("f1",        None)))
        precision = d.get("entity_precision",  d.get("overall_precision", d.get("precision", None)))
        recall    = d.get("entity_recall",     d.get("overall_recall",    d.get("recall",    None)))
        ci        = d.get("confidence_interval", d.get("ci_95", None))
        ci_low    = ci[0] if isinstance(ci, list) and len(ci) == 2 else None
        ci_high   = ci[1] if isinstance(ci, list) and len(ci) == 2 else None
        add_row(model, config, split, f1, precision, recall, ci_low, ci_high, "eval_json")
    except Exception as e:
        add_row(model, config, split, f"ERR:{e}", None, None, source="eval_json")

# -----------------------------------------------
# SOURCE 2: training_summary.json (dictabert_crf custom trainer)
# Has best_f1 on VALIDATION set — mark as [val]
# Covers: P0-P5, P10, A1b-A8 ablations
# Only add if no gold eval_json row already exists for this config
# -----------------------------------------------
already_have = {(r["model"], r["config"]) for r in rows}

for path in sorted(glob.glob(f"{MY_BASE}/models/checkpoints/dictabert_crf/*/training_summary.json") +
                   glob.glob(f"{MY_BASE}/results/dictabert_crf/*/training_summary.json")):
    config = os.path.basename(os.path.dirname(path))
    if ("dictabert_crf", config) in already_have:
        continue  # already have gold eval for this config
    try:
        with open(path) as f:
            d = json.load(f)
        f1 = d.get("best_f1")
        add_row("dictabert_crf", config, "val", f1, None, None, source="train_summary")
    except Exception as e:
        add_row("dictabert_crf", config, "val", f"ERR:{e}", None, None, source="train_summary")

# -----------------------------------------------
# SOURCE 3: training_metrics.json (HuggingFace trainer — dictabert_large ablations)
# Only add if no gold eval_json row already exists
# -----------------------------------------------
for path in sorted(glob.glob(f"{MY_BASE}/results/dictabert_large/*/training_metrics.json") +
                   glob.glob(f"{MY_BASE}/models/checkpoints/dictabert_large/*/training_metrics.json")):
    config = os.path.basename(os.path.dirname(path))
    if ("dictabert_large", config) in already_have:
        continue
    try:
        with open(path) as f:
            d = json.load(f)
        f1        = d.get("eval_f1")
        precision = d.get("eval_precision")
        recall    = d.get("eval_recall")
        add_row("dictabert_large", config, "val", f1, precision, recall, source="train_metrics")
    except Exception as e:
        add_row("dictabert_large", config, "val", f"ERR:{e}", None, None, source="train_metrics")

# -----------------------------------------------
# Print table
# -----------------------------------------------
MODEL_ORDER = ["mbert", "hebert", "alephbert", "xlmr", "dictabert",
               "dictabert_large", "dictabert_crf"]

def model_sort_key(r):
    m = r["model"]
    idx = MODEL_ORDER.index(m) if m in MODEL_ORDER else 99
    return (idx, r["config"], r["split"])

rows.sort(key=model_sort_key)

def fmt(v, digits=4):
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:.{digits}f}"
    return str(v)

print(f"\n{'Model':<22} {'Config':<32} {'Split':<7} {'F1':<8} {'Prec':<8} {'Rec':<8} {'95% CI':<18} {'Src'}")
print("=" * 115)

last_model = None
for r in rows:
    if r["model"] != last_model:
        if last_model is not None:
            print()
        last_model = r["model"]
    src_tag = "" if r["source"] == "eval_json" else f"[{r['source']}]"
    print(f"{r['model']:<22} {r['config']:<32} {r['split']:<7} "
          f"{fmt(r['F1']):<8} {fmt(r['Precision']):<8} {fmt(r['Recall']):<8} "
          f"{r['CI']:<18} {src_tag}")

print(f"\nTotal rows: {len(rows)}")
print("\nNOTE: [train_summary] = validation F1 (not yet gold-evaluated)")
print("      [train_metrics]  = HuggingFace val F1 (not yet gold-evaluated)")
print("      (no tag)         = gold test set evaluation (authoritative)")
