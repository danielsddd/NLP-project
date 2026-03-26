#!/bin/bash
# =============================================================================
# SLURM Script: Train DictaBERT + CRF (V7 — no intent conditioning)
# =============================================================================
# Model: dicta-il/dictabert + CRF layer
# Uses: train_joint.py (NOT train_student.py)
# Trains on train_merged.jsonl with class-weighted CRF emissions
# Differential LR: BERT 2e-5, CRF head 1e-4
# Submit: sbatch scripts/slurm/train_joint.sbatch
# =============================================================================

#SBATCH --job-name=joint_crf
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_joint.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_joint.err

# =============================================================================
# CONFIGURATION
# =============================================================================
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2"
PROJECT_DIR="simanovsky2_DIR/recipe_modification_extraction/recipe_project"

MODEL="dicta-il/dictabert"
TRAIN_FILE="train_merged.jsonl"
OUTPUT_DIR="models/checkpoints/joint_crf"
EPOCHS=10
BATCH_SIZE=8
LR=2e-5
SEED=42

echo "=========================================="
echo "TRAINING: DictaBERT + CRF (V7 — no intent)"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "Model:        $MODEL"
echo "Train File:   $TRAIN_FILE"
echo "Output:       $OUTPUT_DIR"
echo "Epochs:       $EPOCHS"
echo "Batch Size:   $BATCH_SIZE"
echo "LR (BERT):    $LR"
echo "LR (Head):    $(echo "$LR * 5" | bc)"
echo "Seed:         $SEED"
echo "=========================================="

# Activate environment
source simanovsky2_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp

# Set cache paths (avoid home quota)
export HF_HOME=simanovsky2_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false

cd $PROJECT_DIR

# Show GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=========================================="

# Verify data
if [ ! -f "data/processed/$TRAIN_FILE" ]; then
    echo "ERROR: data/processed/$TRAIN_FILE not found!"
    echo "Run preprocessing first:"
    echo "  python -m src.preprocessing.prepare_data_merged \\"
    echo "      --original data/silver_labels/teacher_output.jsonl \\"
    echo "      --enriched data/silver_labels/threads_positives_focus_labeled.jsonl \\"
    echo "      --output-dir data/processed --model $MODEL"
    exit 1
fi

if [ ! -f "data/processed/val.jsonl" ]; then
    echo "ERROR: data/processed/val.jsonl not found!"
    exit 1
fi

if [ ! -f "data/processed/stats_merged.json" ]; then
    echo "ERROR: data/processed/stats_merged.json not found!"
    echo "This file contains class weights. Re-run preprocessing."
    exit 1
fi

echo ""
echo "Data files:"
wc -l data/processed/$TRAIN_FILE data/processed/val.jsonl
echo ""

# Create output dir
mkdir -p $OUTPUT_DIR

# Train — uses train_joint.py with CRF + class-weighted emissions
python -m src.models.train_joint \
    --model "$MODEL" \
    --data-dir data/processed \
    --train-file "$TRAIN_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --neg-ratio 3.0 \

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "TRAINING COMPLETED SUCCESSFULLY"
    echo "Model saved to: $OUTPUT_DIR/best_model.pt"
    echo "=========================================="
    
    # Show result
    if [ -f "$OUTPUT_DIR/training_summary.json" ]; then
        echo "Training summary:"
        cat "$OUTPUT_DIR/training_summary.json"
    fi
else
    echo ""
    echo "=========================================="
    echo "TRAINING FAILED (exit code: $EXIT_CODE)"
    echo "=========================================="
    exit $EXIT_CODE
fi

echo ""
echo "End Time: $(date)"
echo "=========================================="
