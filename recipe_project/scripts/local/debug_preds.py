import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from scripts.local.evaluate_crf import EvalDataset, collate_fn
from src.models.joint_model import BertCRFModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained("dicta-il/dictabert")

BIO_ID2LABEL = {
    0: "O",
    1: "B-SUBSTITUTION", 2: "I-SUBSTITUTION",
    3: "B-QUANTITY",      4: "I-QUANTITY",
    5: "B-TECHNIQUE",     6: "I-TECHNIQUE",
    7: "B-ADDITION",      8: "I-ADDITION",
}

ckpt = torch.load(
    "models/checkpoints/dictabert_crf/P10_crf_thread_aware_io/best_model.pt",
    map_location=device
)
num_labels = 9
model = BertCRFModel(model_name="dicta-il/dictabert", num_labels=num_labels)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.to(device)
model.eval()

ds = EvalDataset("data/gold_validation/gold_tokenized_dictabert_io_thread.jsonl")
loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)

print("Scanning gold for REAL positives (gold entity present)...")
found = 0
with torch.no_grad():
    for batch in loader:
        labels_tensor = batch["labels"][0]
        # True positive = max label ID (excluding -100) > 0
        mask_valid = (labels_tensor != -100)
        if mask_valid.sum() == 0:
            continue   # no valid tokens
        max_label = labels_tensor[mask_valid].max().item()
        if max_label <= 0:
            continue   # no entity (all O)
        # Now we have a genuine positive example
        found += 1
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        preds = model(input_ids, mask)
        for i in range(input_ids.size(0)):
            raw_text = tok.decode(input_ids[i], skip_special_tokens=False)
            print(f"\n--- Positive example {found} ---")
            print("TEXT:", raw_text[:300])
            tokens = tok.convert_ids_to_tokens(input_ids[i])
            gold = batch["labels"][i]
            pred_seq = preds[i]
            print(f"{'Token':<20s} {'Gold_id':>6s} {'Pred_id':>6s} {'Pred_label':<20s}")
            for j in range(len(tokens)):
                if mask[i][j] == 0:
                    break
                t = tokens[j]
                gid = gold[j].item() if gold[j] != -100 else -100
                pid = pred_seq[j] if j < len(pred_seq) else -1
                if gid == -100:
                    continue
                gold_label = "O" if gid == 0 else f"ENT({gid})"
                pred_label = BIO_ID2LABEL.get(pid, "?")
                print(f"{t:<20s} {gold_label:>6s} {pid:>6d} {pred_label:<20s}")
            print("-" * 60)
        if found >= 3:
            break
print("Done.")
