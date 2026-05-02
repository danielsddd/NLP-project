#!/usr/bin/env python3
"""Compute Cohen's kappa on gold overlap region (threads 225-274)."""

import json, argparse

def has_mod(annot):
    if isinstance(annot, bool): return annot
    if isinstance(annot, dict): return bool(annot.get("has_modification", False))
    return None

def cohens_kappa(y1, y2):
    n = len(y1)
    if n == 0: return 0.0
    agree = sum(1 for a,b in zip(y1,y2) if a == b)
    po = agree / n
    p1 = sum(y1)/n
    p2 = sum(y2)/n
    pe = p1*p2 + (1-p1)*(1-p2)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-final", default="data/gold_validation/gold_final.jsonl")
    ap.add_argument("--overlap-start", type=int, default=225)
    ap.add_argument("--overlap-end", type=int, default=274)
    args = ap.parse_args()

    bd, br = [], []
    n_missing = 0
    with open(args.gold_final) as f:
        for i, line in enumerate(f, 1):
            if not (args.overlap_start <= i <= args.overlap_end): continue
            rec = json.loads(line)
            d = rec.get("annotator_daniel")
            r = rec.get("annotator_roei")
            if d is None or r is None:
                n_missing += 1
                continue
            hd = has_mod(d)
            hr = has_mod(r)
            if hd is None or hr is None:
                n_missing += 1
                continue
            bd.append(int(hd))
            br.append(int(hr))

    k = cohens_kappa(bd, br)
    print(f"Records in overlap: {len(bd)}")
    print(f"Missing: {n_missing}")
    print(f"Binary Cohen's kappa: {k:.4f}")

if __name__ == "__main__":
    main()
