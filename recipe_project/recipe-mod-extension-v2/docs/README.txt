# Recipe Mod Detector — Hebrew (Chrome Extension v2.4)

Automatically extracts **recipe modifications** — ingredient substitutions, additions, quantity changes, and technique adjustments — from Hebrew YouTube cooking comments and displays them in a dark‑themed, RTL sidebar.

---

## Table of Contents

- [Screenshots](#screenshots)
- [How It Works](#how-it-works)
- [Model Information](#model-information)
- [HuggingFace Space API](#huggingface-space-api)
- [Confidence Scoring](#confidence-scoring)
- [Comment Loading Mechanism](#comment-loading-mechanism)
- [File Structure](#file-structure)
- [Installation (Developer Mode)](#installation-developer-mode)
- [Troubleshooting](#troubleshooting)
- [Future Production Upgrades](#future-production-upgrades)
- [Credits & Citation](#credits--citation)

---

## Screenshots

> The sidebar slides in automatically on Hebrew recipe videos and groups detected modifications by category.

![Sufganiyot recipe — Additions & Techniques detected](../docs/img1.png)
*Sufganiyot (doughnuts) video — 1 Addition (הזרקתי ריבה, 100%) and 1 Technique (אפיתי בתנור, 100%) detected across 20 comments.*

![Chocolate cake — Technique detected](../docs/img2.png)
*Chocolate Passover cake — 1 Technique (לחורר בשיפוד עץ את העוגה, 100%) detected across 19 comments.*

![Shakshuka — Quantity detected](../docs/img3.png)
*Original shakshuka recipe — 1 Quantity (הורדתי קצת בכמויות השמן, 96%) detected across 18 comments.*

---

## How It Works

1. **Page Detection**  
   The content script (`content.js`) is injected into every `youtube.com/watch` page. It scans the video title and description for Hebrew characters (`\u0590-\u05FF`) and a list of cooking keywords (e.g., מתכון, לבשל, מרכיבים, קמח). If both conditions are met, the sidebar is initialised.

2. **Comment Loading**  
   YouTube lazy‑loads comments as the user scrolls. The extension automatically scrolls the page in steps (800 px each, up to 15 iterations) until the number of visible comment threads stops increasing.  
   It then **clicks every "View replies" button** (`#more-replies`) to expand threaded conversations, waiting 800 ms between clicks to allow YouTube's rendering.

3. **Collection**  
   After expansion, the script grabs all `#content-text` elements inside both `ytd-comment-thread-renderer` (top‑level threads) and `ytd-comment-replies-renderer` (replies). Comments shorter than 10 characters or without Hebrew letters are ignored. Up to **200 comments** are collected per video to ensure timely inference.

4. **Inference**  
   The collected comments are sent via `chrome.runtime.sendMessage` to the background service worker (`background.js`).  
   The worker calls the **HuggingFace Space** API for each comment individually (see API details below) and returns the predictions.

5. **Display**  
   Results are grouped into four categories using the model's `entity_group` field:
   - ⇄ **תחליפים** (Substitutions)
   - + **תוספות** (Additions)
   - ⚖ **כמויות** (Quantities)
   - ✦ **טכניקות** (Techniques)

   Each item shows the extracted phrase, a **confidence score**, and a tooltip with the original comment text (on hover).

---

## Model Information

| Property | Value |
|----------|-------|
| **Architecture** | `dicta-il/dictabert-large` (Hebrew‑optimised BERT, 24 layers, 1024 hidden size) with a linear token‑classification head |
| **Labels (BIO)** | `O`, `B/I‑SUBSTITUTION`, `B/I‑QUANTITY`, `B/I‑TECHNIQUE`, `B/I‑ADDITION` |
| **Training data** | [DanielDDDS/recipe-modifications-v2](https://huggingface.co/datasets/DanielDDDS/recipe-modifications-v2) — 6,004 threads, 23.8% containing at least one modification |
| **Class weighting** | Computed from training set to handle imbalance (3.2:1 negative‑to‑positive ratio) |
| **Training hyperparams** | lr=2e‑5, dropout=0.1, batch_size=16, epochs=10 (best at epoch 9) |
| **Gold test (n=496)** | Exact Entity F1: **29.2%** \| Relaxed Entity F1: **65.6%** |
| **Silver test** | Exact: 30.1% \| Relaxed: 55.2% |

The model is the **DictaBERT‑large P1 variant** (class weights, no focal loss) that achieved the highest relaxed F1 among standard (non‑CRF) models — ideal for browser deployment where lightweight ONNX conversion was avoided.

---

## HuggingFace Space API

The model is served via a **free Gradio Space** at:  
🔗 [https://danielddds-hebrew-recipe-mod-api.hf.space](https://danielddds-hebrew-recipe-mod-api.hf.space)

### Endpoint (two‑step API)

#### Step 1 – Initiate Prediction

```http
POST /gradio_api/call/v2/predict
Content-Type: application/json

{"text": "אפשר להחליף חמאה בשמן קוקוס"}
```

Response:
```json
{"event_id":"0490a1a2738e43f7b42e8fc44d633208"}
```

#### Step 2 – Retrieve Results

```http
GET /gradio_api/call/predict/{event_id}
```

The response is a Server‑Sent Event (SSE) stream.  
The final `data:` line contains the predictions:

```json
data: [{"results": [{"entity_group": "SUBSTITUTION", "score": 0.9999651, "word": "להחליף חמאה בשמן קוקוס", "start": 5, "end": 27}]}]
```

#### Example with curl

```bash
# Step 1
EVENT_ID=$(curl -s -X POST "https://danielddds-hebrew-recipe-mod-api.hf.space/gradio_api/call/v2/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"אפשר להחליף חמאה בשמן קוקוס"}' | python -c "import sys,json; print(json.load(sys.stdin)['event_id'])")

# Step 2
curl -s "https://danielddds-hebrew-recipe-mod-api.hf.space/gradio_api/call/predict/$EVENT_ID" | grep "data:" | tail -1
```

No authentication is required — the Space is public and free. The first request may take a few seconds to wake the model (cold start); subsequent calls are faster.

---

## Confidence Scoring

The percentage displayed next to each extraction (e.g., 96%) is the softmax probability output by the model for that specific entity span.  
It reflects how certain the model is that the identified phrase truly belongs to the labelled category (e.g., SUBSTITUTION).

**Threshold:** Only predictions with confidence ≥ 45% are shown. Lower‑confidence spans are discarded to reduce noise.

**Interpretation:**
- **95‑100%** – very high confidence; almost certainly correct.
- **70‑95%** – high confidence; likely correct.
- **45‑70%** – moderate confidence; may still contain useful information, but worth manual verification.

The threshold (`0.45`) can be adjusted inside `content.js` → `parseEntities()`.

---

## Comment Loading Mechanism

YouTube's comment section loads dynamically. To capture all visible comments including replies, the extension performs the following steps:

1. **Scroll trigger** – The page is scrolled down incrementally (up to 15 times, 800 px each, 1.2 s apart) until the count of `ytd-comment-thread-renderer` elements stops growing.
2. **Reply expansion** – Every visible "View replies" button (`ytd-button-renderer#more-replies`) is clicked programmatically, with an 800 ms pause between clicks to allow YouTube to render the threaded replies.
3. **Collection** – All `#content-text` nodes inside both threads and replies are scanned. Comments must contain Hebrew text and be longer than 10 characters.

If you still see fewer comments than YouTube's displayed count:

- **Manual scroll** – After the sidebar appears, scroll further down manually. The extension listens for URL changes but does not re‑scan on manual scroll after the first scan.
- **Network lag** – Slow connections may delay comment loading. Increase the wait times in `scanComments()` (the `setTimeout` values) if needed.
- **Thread limit** – By default, a maximum of 200 comments are collected to prevent excessive API calls. This limit can be raised in `MAX_COMMENTS`.

---

## File Structure

```
ext2/
├── manifest.json          # Chrome Manifest V3 configuration
├── background.js          # Service worker – calls HuggingFace Space, returns predictions
├── content.js             # Content script – collects comments, drives UI, handles navigation
├── styles.css             # Sidebar styling (dark theme, RTL)
├── icons/
│   ├── icon16.png
│   ├── icon48.png
│   └── icon128.png
└── README.md              # This file
```

| File | Purpose |
|------|---------|
| `manifest.json` | Declares permissions, host permissions (youtube.com, hf.space), background worker, and content script injection. |
| `background.js` | Exchanges messages with content script; calls the two‑step Gradio API; returns predictions. |
| `content.js` | Detects Hebrew recipe videos, loads comments, expands replies, sends batches to background, parses results, renders sidebar. |
| `styles.css` | Dark YouTube‑matching theme, RTL text direction, smooth slide‑in animation. |
| `icons/` | Extension icons at 16×16, 48×48, and 128×128 pixels. |

---

## Installation (Developer Mode)

1. Download or clone this repository to your local machine.
2. Open Google Chrome and navigate to `chrome://extensions/`.
3. Toggle **Developer mode** (top‑right corner).
4. Click **Load unpacked** and select the `ext2/` folder.
5. Visit any Hebrew recipe video on YouTube, e.g.:  
   `https://www.youtube.com/watch?v=oJDC1SNDhvU`

The sidebar will slide in automatically after a few seconds once comments are loaded and scanned.

> **Note:** The first prediction after loading the extension may take 10‑20 seconds because the HuggingFace Space must download the 1.7 GB model on its first request (cold start). Later requests are significantly faster.

---

## Troubleshooting

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| "לא נמצאו תגובות בעברית" | No Hebrew comments found, or the video is not a recipe | Ensure the video has Hebrew comments. Try a known recipe video. |
| Sidebar shows fewer comments than YouTube's count | Replies not fully expanded, or YouTube throttled loading | Scroll manually further down and wait. Increase step count / timeouts in `scanComments()`. |
| "טיימאאוט — המודל עדיין טוען" | Space cold‑start or network delay | Refresh the page after 30 s. Subsequent scans should be fast. |
| No predictions appear (sidebar empty) | Confidence threshold too high or no actual modifications | Lower threshold in `parseEntities()` (change `0.45` to `0.35`). Try a video where viewers discuss changes. |
| Service worker console shows "Invalid regular expression" | Old `background.js` with transformers.js still cached | Remove the extension, clear service worker cache via `chrome://serviceworker-internals/`, and reload. |

---

## Future Production Upgrades

- **Faster Inference** – Replace the free Gradio Space with a dedicated HuggingFace Inference Endpoint (CPU starting at $0.03/hr) for sub‑second latency.
- **Local Caching** – Store predictions per video in `chrome.storage.local` to avoid re‑querying previously scanned videos.
- **Recipe Integration** – Parse the video description to extract the original ingredient list and automatically merge detected modifications into a "suggested recipe".
- **Data Export** – Add a button to download all detected modifications as JSON or CSV for further analysis.
- **Multi‑comment batch API** – Modify the Space to accept a list of texts, reducing network round‑trips.

---

## Credits & Citation

This project was developed in 2026 as part of the **Natural Language Processing (NLP) Course Project** at Tel Aviv University.

**Authors:** Daniel Simanovsky & Roei Ben Artzi  
**Paper:** Under submission.

**HuggingFace Repositories:**
- Model: [DanielDDDS/hebrew-recipe-modification-ner](https://huggingface.co/DanielDDDS/hebrew-recipe-modification-ner)
- Dataset: [DanielDDDS/recipe-modifications-v2](https://huggingface.co/datasets/DanielDDDS/recipe-modifications-v2)
