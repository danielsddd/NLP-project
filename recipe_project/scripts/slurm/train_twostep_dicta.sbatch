#!/bin/bash
#SBATCH --job-name=twostep
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_2080|geforce_rtx_3090|a5000|a6000|l40s"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/recipe_modification_extraction/logs/slurm_outputs/%j_twostep_dicta.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/recipe_modification_extraction/logs/slurm_outputs/%j_twostep_dicta.err

PROJ=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2
source $PROJ/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$PROJ/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd $PROJ/recipe_modification_extraction

MODEL="dicta-il/dictabert"
DATA_OUT="data/processed_dictabert"

echo "============================================================"
echo "TWO-STEP PIPELINE: DictaBERT"
echo "============================================================"
echo "Node: $(hostname), Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Preprocess (skip if already done)
if [ ! -f "$DATA_OUT/train.jsonl" ]; then
    echo ""
    echo "=== Step 0: Preprocessing ==="
    python -m src.preprocessing.prepare_data \
        --input data/silver_labels/teacher_output.jsonl \
        --output-dir $DATA_OUT --model "$MODEL" --scheme BIO
fi

echo ""
echo "============================================================"
echo "STEP 1: Binary Classifier (has modification?)"
echo "============================================================"
echo "Training on ALL examples to learn modification detection..."
echo ""

python -m src.models.train_classifier \
    --model "$MODEL" \
    --data-dir $DATA_OUT \
    --output-dir models/checkpoints/classifier_dictabert \
    --epochs 10 --batch-size 32 --lr 3e-5 --seed 42 --fp16

echo ""
echo "============================================================"
echo "STEP 2: Token Classifier (extract spans)"
echo "============================================================"
echo "Training with heavy downsampling (0.5:1 neg:pos ratio)"
echo "so the model focuses on EXTRACTING modifications,"
echo "not deciding if they exist."
echo ""

python -m src.models.train_student \
    --model "$MODEL" \
    --data-dir $DATA_OUT \
    --output-dir models/checkpoints/dictabert_twostep \
    --epochs 15 --batch-size 16 --lr 2e-5 --warmup 0.1 --seed 42 --fp16 \
    --class-weights balanced --neg-ratio 0.5 --patience 5

echo ""
echo "============================================================"
echo "TWO-STEP PIPELINE COMPLETE"
echo "============================================================"
echo "  Classifier: models/checkpoints/classifier_dictabert/best_model"
echo "  Extractor:  models/checkpoints/dictabert_twostep/best_model"
echo "  End: $(date)"
echo "============================================================"
