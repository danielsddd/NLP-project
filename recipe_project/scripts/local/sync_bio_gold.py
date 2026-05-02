import json

bio_in = "data/gold_validation/gold_tokenized_dictabert.jsonl"
io_thread = "data/gold_validation/gold_enriched_tokenized_dictabert_io_thread_fixed.jsonl"
bio_out = "data/gold_validation/gold_tokenized_dictabert_bio_thread.jsonl"

count = 0
with open(bio_in) as fb, open(io_thread) as fi, open(bio_out, 'w') as fo:
    for lb, li in zip(fb, fi):
        db = json.loads(lb)
        di = json.loads(li)
        
        # Extract the actual unpadded labels from the original BIO file
        orig_bio_labels = [lbl for lbl in db['labels'] if lbl != -100]
        
        # Reconstruct the labels array matching the thread-injected token structure
        new_labels = []
        bio_idx = 0
        for io_lbl in di['labels']:
            if io_lbl == -100:
                new_labels.append(-100) # Keep the padding (including the new [שאלה] prefix)
            else:
                if bio_idx < len(orig_bio_labels):
                    new_labels.append(orig_bio_labels[bio_idx])
                    bio_idx += 1
                else:
                    new_labels.append(-100) # Handle edge-case truncation
        
        # Save the new BIO labels into the thread-aware structure
        di['labels'] = new_labels
        fo.write(json.dumps(di, ensure_ascii=False) + '\n')
        count += 1

print(f"Successfully generated {bio_out} with {count} records!")
