#!/usr/bin/env python3
import json
from pathlib import Path

EXCLUDE = {
    "dictabert_crf/P4v3_thread_focal",
    "dictabert_crf/P10v4_focal",
    "dictabert_crf/P10v3_lr1e5",
    "dictabert_crf/P4v2_thread_fixed",
    "dictabert_crf/P5v2_thread_no_enriched_fixed",
}

results_root = Path("results")
rows = {}

# gather gold
for gf in sorted(results_root.rglob("**/gold/evaluation_results.json")):
    model = str(gf.parent.parent.relative_to(results_root))
    if model in EXCLUDE:
        continue
    d = json.loads(gf.read_text())
    rows.setdefault(model, {})["gold"] = d

# gather silver
for sf in sorted(results_root.rglob("**/silver/evaluation_results.json")):
    model = str(sf.parent.parent.relative_to(results_root))
    if model in EXCLUDE:
        continue
    d = json.loads(sf.read_text())
    rows.setdefault(model, {})["silver"] = d

def fmt_pct(v):
    return f"{v*100:5.1f}%" if isinstance(v, float) else "   ·  "

def fmt_ci(lo, hi):
    if lo is None or hi is None:
        return "       ·        "
    return f"[{lo*100:.1f}%,{hi*100:.1f}%]"

table = []
for model, data in sorted(rows.items()):
    g = data.get("gold", {})
    s = data.get("silver", {})
    exact_g = g.get("entity_f1")
    relax_g = g.get("relaxed_f1")
    ci_lo   = g.get("relaxed_ci_95_low")
    ci_hi   = g.get("relaxed_ci_95_high")
    exact_s = s.get("entity_f1") if s else None
    relax_s = s.get("relaxed_f1") if s else None
    table.append((model, exact_g, relax_g, ci_lo, ci_hi, exact_s, relax_s))

# sort by relaxed gold F1 desc
table.sort(key=lambda r: r[2] if r[2] is not None else 0, reverse=True)

header = f"{'Model':<45s} {'GoldExact':>9s} {'GoldRelax':>9s} {'Relaxed CI':>18s} {'SilvExact':>9s} {'SilvRelax':>9s}"
print(header)
print("-" * len(header))
for model, eg, rg, clo, chi, es, rs in table:
    print(f"{model:<45s} {fmt_pct(eg):>9s} {fmt_pct(rg):>9s} {fmt_ci(clo, chi):>18s} {fmt_pct(es):>9s} {fmt_pct(rs):>9s}")

print(f"\nTotal models: {len(table)}")