# Recipe Mod Detector — Hebrew (Chrome Extension)

Automatically detects **recipe modifications** (substitutions, additions, quantity changes, technique adjustments) in Hebrew YouTube cooking comments and displays them in a sleek, dark‑theme sidebar.

---

## How It Works

1. **Page detection** – The extension injects a content script into any `youtube.com/watch` page.  
   It heuristically identifies Hebrew recipe videos by scanning the title and description for Hebrew letters and cooking keywords.

2. **Comment loading** – The script scrolls down the page to trigger YouTube's lazy‑loading of comments, then clicks all « View replies » buttons to expand threaded conversations.

3. **Batch inference** – Up to 200 Hebrew comments are sent to the background service worker, which forwards them one‑by‑one to a **HuggingFace Space** running a fine‑tuned DictaBERT‑large model.

4. **Result display** – Predictions are grouped into four categories:
   - 🥄 **תחליפים** (Substitutions)
   - ➕ **תוספות** (Additions)
   - ⚖ **כמויות** (Quantities)
   - ✦ **טכניקות** (Techniques)

   Each extracted phrase is shown together with a **confidence score** (see below).

---

## Model Information

- **Architecture**: `dicta‑il/dictabert‑large` (a Hebrew‑optimised BERT variant) with a linear token‑classification head.
- **Training data**: `DanielDDDS/recipe‑modifications‑v2` (6,004 threads, 24% containing modifications).
- **Class weights**: Computed from the training set to handle imbalance.
- **Performance on gold test set (n=496)**:
  - Exact Entity F1: **29.2%**
  - Relaxed Entity F1: **65.6%**
  - (Silver test set: Exact 30.1%, Relaxed 55.2%)

The model is hosted as a **free Gradio Space** on Hugging Face: