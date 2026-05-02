import json
from transformers import AutoTokenizer
from pathlib import Path

# ---------- settings ----------
SILVER_PATH = "data/silver_labels/teacher_output_v2.jsonl"
DATA_DIR    = Path("data/processed_thread_aware_io")
TOKENIZER   = "dicta-il/dictabert"
MAX_LEN     = 128

# ---------- load parent texts ----------
print("Loading parent texts from silver file...")
parent_texts = {}
with open(SILVER_PATH) as f:
    for line in f:
        rec = json.loads(line)
        tid = rec["thread_id"]
        parent = rec.get("top_comment_text", "").strip()
        if parent:
            parent_texts[tid] = parent
print(f"  Loaded {len(parent_texts)} threads with parent text")

tok = AutoTokenizer.from_pretrained(TOKENIZER)
# Special tokens: [CLS] at 0, [SEP] at end – we'll keep them, 
# but the prefix goes after [CLS] and before the reply text.
# In your existing tokenization, the reply text starts at index 1 (after [CLS]).
# We'll compute prefix_len as number of tokens in the string " [שאלה] parent [תשובה] "
# (the space is because the reply text originally had no leading space after [CLS]).

def inject_and_shift(rec):
    tid = rec["thread_id"]
    source = rec["source_comment"]
    old_text = rec["text"]   # original text (already may contain prefix if previously injected)
    labels = rec["labels"]

    # Avoid double injection
    if source != "top" and tid in parent_texts and not old_text.startswith("[שאלה]"):
        parent = parent_texts[tid]
        # Build prefix string with space separator that matches tokenizer
        prefix_str = f"[שאלה] {parent} [תשובה] "
        # Tokenize prefix (include special tokens? No, we'll insert after [CLS])
        prefix_tokens = tok.encode(prefix_str, add_special_tokens=False)
        prefix_len = len(prefix_tokens)

        # Build new full text
        new_text = prefix_str + old_text
        rec["text"] = new_text

        # Re-tokenize the new text as a replacement for input_ids
        encoded = tok(
            new_text,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )
        rec["input_ids"] = encoded["input_ids"]
        rec["attention_mask"] = encoded["attention_mask"]

        # Shift labels: the original labels (without [CLS] and padding) start at index 1.
        # We'll cut off the [CLS] label (-100) and the padding, keep only the reply part.
        # Original labels: [-100, L1, L2, ..., Ln, -100, ...] (padded to MAX_LEN)
        # The reply labels are L1...Ln (non-padding). We need to count only the non-(-100) labels.
        # A safer method: extract the original reply labels list (excluding -100, excluding O maybe? No, include all).
        # Actually, we can just shift the whole list right by prefix_len, putting -100 for prefix positions.
        # But the original list is already MAX_LEN long. After shifting, we need to truncate.
        new_labels = [-100] * prefix_len
        for l in labels:
            if l != -100:   # skip special tokens (including the initial [CLS] which is -100)
                new_labels.append(l)
        # Pad/truncate to MAX_LEN
        if len(new_labels) < MAX_LEN:
            new_labels += [-100] * (MAX_LEN - len(new_labels))
        else:
            new_labels = new_labels[:MAX_LEN]
        rec["labels"] = new_labels
        rec["is_thread_aware"] = True
    else:
        # Ensure correct padding for non-replies
        if len(labels) < MAX_LEN:
            rec["labels"] = labels + [-100] * (MAX_LEN - len(labels))
        else:
            rec["labels"] = labels[:MAX_LEN]
        rec["is_thread_aware"] = False
    return rec

# ---------- process each split ----------
for split in ["train_merged.jsonl", "val.jsonl", "test.jsonl"]:
    infile  = DATA_DIR / split
    outfile = DATA_DIR / (split + ".tmp")
    print(f"\nProcessing {split} ...")
    count = 0
    with open(infile) as fin, open(outfile, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            rec = inject_and_shift(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    outfile.rename(infile)
    print(f"  -> {split} updated ({count} records)")

print("\nDone. Context injected with label alignment via shift.")
