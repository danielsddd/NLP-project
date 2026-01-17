# Recipe Modification Extraction

**Distilling LLMs for Crowd-Sourced Recipe Modification Extraction**

Group 11: Daniel Simanovsky & Roei Ben Artzi  
NLP 2025a - Tel Aviv University

## 📋 Project Overview

This project extracts recipe modifications from Hebrew YouTube comments using a Teacher-Student distillation approach:

1. **Data Collection**: Collect Hebrew comments from cooking videos (YouTube API)
2. **Teacher Labeling**: Use Gemini 1.5 Pro to generate "silver labels"
3. **Preprocessing**: Convert spans to BIO tags for sequence tagging
4. **Student Training**: Fine-tune AlephBERT on the silver labels
5. **Evaluation**: Compare against baselines, analyze errors

## 🗂️ Project Structure

```
recipe_modification_extraction/
├── config/
│   ├── config.yaml          # Main configuration
│   └── channels.yaml        # YouTube channels to collect from
├── src/
│   ├── data_collection/     # YouTube comment collector (your existing code)
│   ├── teacher_labeling/    # Gemini/GPT-4o silver label generation
│   ├── preprocessing/       # Span → BIO alignment
│   ├── models/              # AlephBERT training
│   ├── baselines/           # Baseline implementations
│   └── evaluation/          # Metrics and error analysis
├── data/                    # Data storage (gitignored)
│   ├── raw_youtube/         # Collected comments
│   ├── silver_labels/       # Teacher outputs
│   ├── processed/           # Training-ready data
│   └── gold_validation/     # Human-annotated test set
├── models/                  # Model checkpoints (gitignored)
├── results/                 # Evaluation results
├── scripts/                 # SLURM scripts (for cluster)
├── .env.example             # Environment variables template
├── requirements.txt         # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/recipe-modification-extraction.git
cd recipe-modification-extraction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys:
# - YOUTUBE_API_KEY (from Google Cloud Console)
# - GOOGLE_API_KEY (from Google AI Studio for Gemini)
```

### 3. Run the Pipeline

```bash
# Step 1: Collect YouTube comments
# (Use your existing collector in youtube_collector/)
python youtube_collector/collect.py --api-key $YOUTUBE_API_KEY --collect

# Step 2: Generate silver labels
python -m src.teacher_labeling.generate_labels \
    --input data/raw_youtube/comments.jsonl \
    --provider gemini

# Step 3: Preprocess for training
python -m src.preprocessing.prepare_data \
    --input data/silver_labels/teacher_output.jsonl

# Step 4: Train student model
python -m src.models.train_student \
    --data-dir data/processed \
    --output-dir models/checkpoints/student

# Step 5: Run baselines
python -m src.baselines.run_baselines \
    --data-dir data/processed

# Step 6: Evaluate
python -m src.evaluation.evaluate \
    --model-path models/checkpoints/student/best_model \
    --output-dir results
```

## 📊 Pipeline Details

### Step 1: Data Collection (YouTube)

Your existing `youtube_collector/` handles this. Key files:
- `config.yaml` - Collection settings
- `channels.yaml` - Add Hebrew cooking channels here

**Finding Channels:**
```bash
python youtube_collector/collect.py --discover "מתכונים בישול"
```

### Step 2: Teacher Labeling (Gemini/GPT-4o)

Generates silver labels using LLM:
- Classifies modification aspects: SUBSTITUTION, QUANTITY, TECHNIQUE, ADDITION
- Extracts text spans containing modifications

```bash
# Using Gemini (free tier)
python -m src.teacher_labeling.generate_labels \
    --input data/raw_youtube/comments.jsonl \
    --provider gemini \
    --api-key YOUR_GOOGLE_API_KEY

# Using GPT-4o (backup)
python -m src.teacher_labeling.generate_labels \
    --input data/raw_youtube/comments.jsonl \
    --provider openai \
    --api-key YOUR_OPENAI_API_KEY
```

### Step 3: Preprocessing (Span → BIO)

Converts character-level spans to token-level BIO tags:

```bash
python -m src.preprocessing.prepare_data \
    --input data/silver_labels/teacher_output.jsonl \
    --output-dir data/processed \
    --model onlplab/alephbert-base \
    --scheme BIO
```

### Step 4: Student Training (AlephBERT)

Fine-tunes AlephBERT for token classification:

```bash
python -m src.models.train_student \
    --data-dir data/processed \
    --output-dir models/checkpoints/student \
    --epochs 5 \
    --batch-size 16 \
    --lr 2e-5 \
    --fp16
```

### Step 5: Baselines

Runs baseline comparisons:
- **Majority**: All "O" predictions
- **Random**: Random labels (weighted by distribution)
- **Keyword**: Hebrew keyword matching

```bash
python -m src.baselines.run_baselines --data-dir data/processed
```

### Step 6: Evaluation

Comprehensive evaluation with error analysis:

```bash
python -m src.evaluation.evaluate \
    --model-path models/checkpoints/student/best_model \
    --test-file data/processed/dataset_test.jsonl \
    --output-dir results \
    --bootstrap
```

## 🔧 Configuration

### Main Config (`config/config.yaml`)

```yaml
# Key settings
youtube:
  target_comments: 3000
  max_videos_per_channel: 50

teacher:
  provider: "gemini"  # or "openai"

student:
  model_name: "onlplab/alephbert-base"
  epochs: 5
  batch_size: 16
  learning_rate: 2.0e-5
```

### Channels (`config/channels.yaml`)

Add Hebrew cooking channels after discovering them:

```yaml
channels:
  - name: "השף הביתי"
    id: "UCxxxxxxxxxx"
    active: true
    category: "cooking"
```

## 📈 Expected Results

| Model | Token Acc. | Precision | Recall | F1 |
|-------|-----------|-----------|--------|-----|
| Majority | ~90% | 0.0% | 0.0% | 0.0% |
| Random | ~85% | ~2% | ~2% | ~2% |
| Keyword | ~88% | ~45% | ~22% | ~30% |
| **AlephBERT (Ours)** | ~93% | ~70% | ~65% | ~67% |

## 📝 For the Paper

Remember to:
1. Create a Gold validation set (200 human-annotated examples)
2. Run ablation studies (data size, BIO vs IO, mBERT vs AlephBERT)
3. Include error analysis in Results section
4. Cite Segal et al. (2020) and Gil et al. (2019)

## 🔑 Getting API Keys

### YouTube Data API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable "YouTube Data API v3"
4. Create credentials → API Key

### Google AI (Gemini)
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API Key"
3. Create new key

### OpenAI (Backup)
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create API key

## 📚 References

- Segal et al. (2020). "A Simple and Effective Model for Answering Multi-span Questions" - EMNLP 2020
- Gil et al. (2019). "White-to-Black: Efficient Distillation of Black-Box Adversarial Attacks" - NAACL 2019
- AlephBERT: Pre-trained Language Model for Hebrew

## 👥 Authors

- Daniel Simanovsky (212238174)
- Roei Ben Artzi (213498504)

## 📄 License

This project is for academic purposes (NLP 2025a course, Tel Aviv University).
