#!/usr/bin/env python3
"""
Re-tokenize the silver test set with each model's own tokenizer.
Produces data/processed_{slug}/test.jsonl for xlmr, alephbert, mbert, hebert.
"""
import json, os, argparse
from transformers import AutoTokenizer

MODELS = {
    "xlmr":      "xlm-roberta-base",
    "alephbert": "onlplab/alephbert-base",
    "mbert":     "bert-base-multilingual-cased",
    "hebert":    "avichr/heBERT",
}

BIO_LABELS = ["O",
              "B-SUBSTITUTION","I-SUBSTITUTION",
              "B-ADDITION","I-ADDITION",
              "B-QUANTITY","I-QUANTITY",
              "B-TECHNIQUE","I-TECHNIQUE"]
LABEL2ID   = {l: i for i, l in enumerate(BIO_LABELS)}
ID2LABEL   = {i: l for i, l in enumerate(BIO_LABELS)}
MAX_LEN    = 128
IGNORE_IDX = -100


def align_word_labels(word_labels, word_ids):
    out, prev = [], None
    for wid in word_ids:
        if wid is None:
            out.append(IGNORE_IDX)
        elif wid != prev:
            out.append(LABEL2ID.get(word_labels[wid], 0))
        else:
            lbl = word_labels[wid]
            if lbl.startswith("B-"):
                lbl = "I-" + lbl[2:]
            out.append(LABEL2ID.get(lbl, 0))
        prev = wid
    return out


def process_model(slug, model_name, examples, out_path, src_tok):
    print(f"\n[{slug}]  model: {model_name}")
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"  ERROR loading tokenizer: {e}")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_written = n_word_aligned = n_fallback = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex in examples:
            text        = ex.get("text", "")
            int_labels  = ex.get("labels", [])
            word_tokens = ex.get("tokens")
            ner_tags    = ex.get("ner_tags")

            # Path A: word tokens + word-level BIO strings (best)
            if word_tokens and ner_tags:
                enc = tok(word_tokens, is_split_into_words=True,
                          max_length=MAX_LEN, truncation=True, padding="max_length")
                new_labels = align_word_labels(ner_tags, enc.word_ids())
                n_word_aligned += 1

            # Path B: word tokens only — recover word labels from DictaBERT alignment
            elif word_tokens and int_labels:
                src_enc = src_tok(word_tokens, is_split_into_words=True,
                                  max_length=MAX_LEN, truncation=True, padding="max_length")
                recovered = {}
                for wid, lbl in zip(src_enc.word_ids(), int_labels):
                    if wid is not None and wid not in recovered and lbl != IGNORE_IDX:
                        recovered[wid] = ID2LABEL.get(lbl, "O")
                n_words = max((w for w in src_enc.word_ids() if w is not None), default=-1) + 1
                approx = [recovered.get(i, "O") for i in range(n_words)]
                enc = tok(word_tokens, is_split_into_words=True,
                          max_length=MAX_LEN, truncation=True, padding="max_length")
                new_labels = align_word_labels(approx, enc.word_ids())
                n_fallback += 1

            # Path C: raw text only (input_ids correct, labels approximate)
            else:
                enc = tok(text, max_length=MAX_LEN, truncation=True, padding="max_length")
                new_labels = (int_labels[:MAX_LEN] +
                              [IGNORE_IDX] * (MAX_LEN - len(int_labels)))[:MAX_LEN]
                n_fallback += 1

            rec = {
                "text":           text,
                "input_ids":      enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels":         new_labels,
            }

            if "token_type_ids" in enc:
                try:
                    from transformers import AutoConfig
                    cfg = AutoConfig.from_pretrained(model_name)
                    cap = max(0, cfg.type_vocab_size - 1)
                except Exception:
                    cap = 0
                rec["token_type_ids"] = [min(v, cap) for v in enc["token_type_ids"]]

            for k in ("ner_tags", "tokens", "thread_id", "post_id", "label"):
                if k in ex:
                    rec[k] = ex[k]

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"  {n_written} examples -> {out_path}")
    print(f"     word-aligned: {n_word_aligned}  |  fallback: {n_fallback}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    default="data/processed_v2/test.jsonl")
    parser.add_argument("--src-model", default="dicta-il/dictabert")
    parser.add_argument("--models",    nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    examples = [json.loads(l) for l in open(args.source, encoding="utf-8")]
    print(f"Loaded {len(examples)} examples from {args.source}")

    sample = examples[0]
    print(f"Fields: {sorted(sample.keys())}")
    print(f"  has 'tokens':   {'tokens' in sample}")
    print(f"  has 'ner_tags': {'ner_tags' in sample}")

    src_tok = AutoTokenizer.from_pretrained(args.src_model)

    for slug in args.models:
        out_path = f"data/processed_{slug}/test.jsonl"
        if os.path.exists(out_path) and not args.overwrite:
            print(f"\n[{slug}]  already exists, skipping (use --overwrite to redo)")
            continue
        process_model(slug, MODELS[slug], examples, out_path, src_tok)

    print("\nDone.")

if __name__ == "__main__":
    main()
