import json
from transformers import AutoTokenizer
from pathlib import Path

SRC_DIR = Path("data/processed_thread_aware_io")
DST_DIR = Path("data/processed_thread_aware_io_large")
TOKENIZER_NAME = "dicta-il/dictabert-large"
MAX_LEN = 128

# Ensure destination exists
for split in ["train_merged.jsonl", "val.jsonl", "test.jsonl"]:
    (DST_DIR / split).parent.mkdir(parents=True, exist_ok=True)

base_tok = AutoTokenizer.from_pretrained("dicta-il/dictabert")
large_tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def convert_labels_by_chars(example):
    # Get original tokens and labels from base tokenizer
    base_ids = example["input_ids"]
    base_labels = example["labels"]
    # Remove padding and special tokens by decoding the full text exactly
    base_tokens = base_tok.convert_ids_to_tokens(base_ids)
    # Reconstruct character offsets for each base token
    # We'll use the tokenizer's decode method with clean_up_tokenization_spaces=False
    # to get the actual text for each token.
    # Simpler: use the fact that 'example["text"]' is the complete text.
    text = example["text"]
    
    # Encode with large tokenizer (with special tokens)
    large_enc = large_tok(
        text,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,  # we need character offsets
        return_tensors=None
    )
    large_ids = large_enc["input_ids"]
    large_offsets = large_enc["offset_mapping"]  # list of (start, end) char pairs

    # Map base token labels to character positions
    # First, get base offsets
    base_enc = base_tok(
        text,
        return_offsets_mapping=True,
        return_tensors=None
    )
    base_offsets = base_enc["offset_mapping"]
    # We need base labels for each token (excluding specials)
    # The labels list corresponds to input_ids; we must align to offsets.
    # Remove the first and last token offsets (CLS, SEP) and pad
    base_label_seq = []
    for off, lbl in zip(base_offsets, base_labels):
        if off == (0, 0):  # special token
            continue
        base_label_seq.append((off, lbl))
    
    # Now assign labels to large tokens based on overlap
    large_labels = [-100] * len(large_ids)  # start with -100 (ignore)
    for i, (l_start, l_end) in enumerate(large_offsets):
        if l_end == 0:  # special token
            continue
        # Find which base token(s) overlap with this character span
        overlapping_labels = []
        for (b_start, b_end), blbl in base_label_seq:
            if b_end <= l_start or l_end <= b_start:
                continue
            # Overlap
            overlapping_labels.append(blbl)
        if overlapping_labels:
            # Majority vote: pick the most frequent non-O label, or O if all O
            non_zero = [l for l in overlapping_labels if l != 0]
            if non_zero:
                # Choose the most common (if tie, first)
                label = max(set(non_zero), key=non_zero.count)
            else:
                label = 0
            large_labels[i] = label
        else:
            large_labels[i] = 0   # no overlap -> O
    
    # Finally, truncate labels to match length
    large_labels = large_labels[:MAX_LEN] + [-100] * (MAX_LEN - len(large_labels))
    
    new_example = {
        "input_ids": large_ids,
        "attention_mask": large_enc["attention_mask"],
        "labels": large_labels,
        "text": text,
        "thread_id": example.get("thread_id", ""),
        "source_comment": example.get("source_comment", ""),
        "is_thread_aware": True,
    }
    return new_example

for split in ["train_merged.jsonl", "val.jsonl", "test.jsonl"]:
    infile = SRC_DIR / split
    outfile = DST_DIR / split
    print(f"Converting {infile} -> {outfile}")
    with open(infile, "r") as fin, open(outfile, "w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            new_ex = convert_labels_by_chars(ex)
            fout.write(json.dumps(new_ex, ensure_ascii=False) + "\n")
    print(f"  Done {split}")

# Also copy stats_merged.json if present, but we can just use class weights from this new data later
print("Done conversion for large models.")
