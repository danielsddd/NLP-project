# Recipe Modification Extraction

**Distilling LLMs for Crowd-Sourced Recipe Modification Extraction**

Group 11: Daniel Simanovsky & Roei Ben Artzi  
NLP 2026 — Tel Aviv University

---

## 📋 Project Overview

This project extracts recipe modifications (substitutions, additions, quantity changes, technique adjustments) from Hebrew YouTube cooking comments using a Teacher-Student distillation approach.

A **Gemini 1.5 Pro** teacher model generates silver labels from raw YouTube comments. Multiple Hebrew-optimised BERT-based student models are then fine-tuned for token-level NER (BIO/IO tagging) and evaluated on both silver and human-annotated gold sets.

The best-performing model — **DictaBERT-large with class weights (P1)** — is exported to HuggingFace and served via a Chrome Extension sidebar.

---

## 🗂️ Project Structure

```
recipe_project/
├── src/
│   ├── teacher_labeling/       # Gemini silver label generation
│   ├── preprocessing/          # Span → BIO/IO alignment (standard + thread-aware)
│   ├── models/                 # Training: standard, CRF, joint
│   ├── baselines/              # Keyword, majority, random baselines
│   ├── evaluation/             # Metrics, bootstrap, pipeline eval
│   └── utils/                  # Class weight computation
├── youtube_collector/          # YouTube Data API comment collection
├── scripts/
│   ├── local/                  # IAA, span validation, gold prep, final tables
│   └── slurm/                  # SLURM .sbatch scripts per model & ablation
├── models/
│   ├── checkpoints/            # Per-model training checkpoints
│   ├── checkpoints_sqrt_nodownsample/  # Sqrt-weighted, no-downsample variants
│   └── hf_export/              # HuggingFace-ready exports (safetensors + ONNX)
├── results/                    # Eval logs (gold & silver) per model/variant
├── raw_gold_labels/            # Human annotation CSV + annotation tool (HTML)
├── requirements.txt
└── README.md
```

---

## 🧠 Models Trained & Compared

All models were trained on the SLURM cluster (Kilonova, TAU) and evaluated on both **silver** (teacher-labeled) and **gold** (human-annotated, n=496) test sets.

| Model | Variants |
|-------|---------|
| `AlephBERT` | P0 baseline, P1 class weights, P2 focal loss |
| `DictaBERT` | P0–P2 + IO/thread-aware variants |
| `DictaBERT-large` | P0–P3, ablations A1–A8 **(best model)** |
| `DictaBERT-CRF` | P0–P10, thread-aware, focal, seed sweeps |
| `DictaBERT-large-CRF` | P2, P4, P10 variants |
| `HeBERT` | P0–P2 + IO/thread-aware |
| `mBERT` | P0–P2 + IO/thread-aware |
| `XLM-R` | P0–P2 |

**Best models:**
- `DictaBERT-large P1` (class weights, no CRF) → deployed in Chrome Extension: [`DanielDDDS/hebrew-recipe-modification-ner`](https://huggingface.co/DanielDDDS/hebrew-recipe-modification-ner)
- `DictaBERT-CRF P1` (class weights + CRF) → paper result: [`DanielDDDS/hebrew-recipe-modification-ner-crf`](https://huggingface.co/DanielDDDS/hebrew-recipe-modification-ner-crf)

---

## 📊 Key Results

> ⚠️ Numbers below are pending final verification from `gold_silver_table.txt`. To be updated.

| Model | Split | Exact Entity F1 | Relaxed Entity F1 |
|-------|-------|----------------|------------------|
| DictaBERT-large P1 (standard) | Gold (n=496) | 25.5% | 62.6% |
| DictaBERT-CRF P1 (paper) | Gold (n=496) | **29.2%** | **65.6%** |
| DictaBERT-CRF P1 (paper) | Silver | 30.1% | 55.2% |

---

## 🔬 Ablation Studies (DictaBERT-large)

| Ablation | Description |
|----------|-------------|
| A1a/b/c | No weights / uniform weights / weights only |
| A2 | Downsampling (2× and 4×) |
| A5 | Data size (25% / 50% / 75% of training set) |
| A6 | No enriched comments |
| A7 | IO scheme instead of BIO |
| A8 | Unanimous-only silver labels |

---

## 🚀 Pipeline

### 1. Data Collection

```bash
# Discover Hebrew cooking channels
python youtube_collector/discover_channels.py

# Collect comments
python youtube_collector/collect.py --api-key $YOUTUBE_API_KEY --collect

# Collect only positive (modification-containing) examples
python youtube_collector/collect_positives.py
```

### 2. Teacher Labeling (Gemini)

```bash
python -m src.teacher_labeling.generate_labels \
    --input data/raw_youtube/comments.jsonl \
    --provider gemini \
    --api-key $GOOGLE_API_KEY
```

### 3. Preprocessing (Span → BIO/IO)

Multiple preprocessing variants — run via SLURM:

```bash
# Default BIO
sbatch scripts/slurm/preprocessing/01_preprocess_default.sbatch

# Thread-aware BIO
sbatch scripts/slurm/preprocessing/07_preprocess_thread_aware.sbatch

# IO scheme
sbatch scripts/slurm/preprocessing/05_preprocess_io_scheme.sbatch
```

Or locally:

```bash
python -m src.preprocessing.prepare_data \
    --input data/silver_labels/teacher_output.jsonl \
    --output-dir data/processed \
    --model dicta-il/dictabert-large
```

### 4. Training (SLURM)

```bash
# Example: DictaBERT-large P1 (best model)
sbatch scripts/slurm/dictabert_large/P1_add_weights.sbatch

# Example: DictaBERT-CRF P1 with seed sweep
sbatch scripts/slurm/dictabert_crf/P1_crf_seed123.sbatch
sbatch scripts/slurm/dictabert_crf/P1_crf_seed2026.sbatch
sbatch scripts/slurm/dictabert_crf/P1_crf_seed7777.sbatch
```

### 5. Baselines

```bash
sbatch scripts/slurm/run_baselines.sbatch
# or locally:
python -m src.baselines.run_baselines --data-dir data/processed
```

### 6. Evaluation

```bash
# Evaluate a single model on gold + silver
python -m src.evaluation.evaluate \
    --model-path models/checkpoints/dictabert_large/P1_add_weights \
    --output-dir results/dictabert_large/P1_add_weights \
    --bootstrap

# Evaluate all models at once
sbatch scripts/slurm/eval_all_complete.sbatch
```

### 7. Gold Annotation

Human annotations are stored in `raw_gold_labels/`:
- `annotation_sample.csv` — raw annotated spans
- `annotation_tool.html` — browser-based annotation UI

IAA (Inter-Annotator Agreement) computed via:
```bash
python scripts/local/compute_iaa.py
python scripts/local/compute_kappa.py
```

### 8. HuggingFace Export

Final models exported to `models/hf_export/`:

| Export | Description |
|--------|-------------|
| `dictabert_large_P1` | Best model (safetensors) |
| `dictabert_large_P1_onnx` | ONNX version for browser/edge deployment |
| `crf_paper` | Best CRF variant for paper |
| `P3_no_crf` | P3 standard variant |

---

## ⚙️ Environment Setup

```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Add: YOUTUBE_API_KEY, GOOGLE_API_KEY
```

---

## 🔑 Getting API Keys

- **YouTube Data API v3** — [Google Cloud Console](https://console.cloud.google.com/) → Enable API → Create Key
- **Gemini** — [Google AI Studio](https://aistudio.google.com/) → Get API Key

---

## 📚 References

- Segal et al. (2020). "A Simple and Effective Model for Answering Multi-span Questions" — EMNLP 2020
- Gil et al. (2019). "White-to-Black: Efficient Distillation of Black-Box Adversarial Attacks" — NAACL 2019
- Shmidman et al. (2023). DictaBERT — Hebrew BERT models

---

## 🔌 Chrome Extension

A Chrome extension that runs this model live on Hebrew YouTube recipe videos is available in `recipe-mod-extension-v2/ext2/`. It automatically scans comments, calls the HuggingFace Gradio API, and displays detected modifications in a dark-themed RTL sidebar.

See the [dedicated README](../recipe-mod-extension-v2/README.md) for installation and usage instructions.

---



This project was developed in 2026 as part of the **Natural Language Processing (NLP) Course Project** at Tel Aviv University.

**Authors:** Daniel Simanovsky & Roei Ben Artzi  
**Paper:** Under submission.

**HuggingFace Repositories:**

* Best model (standard head, used in Chrome extension): **[DanielDDDS/hebrew-recipe-modification-ner](https://huggingface.co/DanielDDDS/hebrew-recipe-modification-ner)**
* Best CRF model (paper result): **[DanielDDDS/hebrew-recipe-modification-ner-crf](https://huggingface.co/DanielDDDS/hebrew-recipe-modification-ner-crf)**
* Dataset: **[DanielDDDS/recipe-modifications-v2](https://huggingface.co/datasets/DanielDDDS/recipe-modifications-v2)**
* Inference API (Gradio Space): **[danielddds-hebrew-recipe-mod-api.hf.space](https://danielddds-hebrew-recipe-mod-api.hf.space)**

---

## 📄 License

This project is for academic purposes (NLP 2026, Tel Aviv University).
