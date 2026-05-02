import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

import torch, json
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from scripts.local.evaluate_crf import EvalDataset, collate_fn
from src.models.joint_model import BertCRFModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained("dicta-il/dictabert")

ckpt = torch.load("models/checkpoints/dictabert_crf/P10_crf_thread_aware_io/best_model.pt", map_location=device)
model = BertCRFModel(model_name="dicta-il/dictabert", num_labels=9)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.to(device)
model.eval()

ds = EvalDataset("data/gold_validation/gold_tokenized_dictabert_bio_thread.jsonl")
loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)

found = 0
with torch.no_grad():
    for batch in loader:
        labels = batch["labels"][0]
        mask_valid = (labels != -100)
        if mask_valid.sum() == 0 or labels[mask_valid].max() <= 0:
            continue
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        preds = model(input_ids, mask)
        for i in range(input_ids.size(0)):
            found += 1
            if found > 3:
                break
            tokens = tok.convert_ids_to_tokens(input_ids[i])
            gold = batch["labels"][i]
            pred_seq = preds[i]
            print(f"--- Positive {found} ---")
            for j, (t, g, p) in enumerate(zip(tokens, gold, pred_seq)):
                if mask[i][j] == 0:
                    break
                if g == -100:
                    continue
                gid = g.item()
                if gid > 0 or p > 0:
                    print(f"{t:20s} gold={gid} pred={p}")
            print()
        if found >= 3:
            break
print("Done.")
