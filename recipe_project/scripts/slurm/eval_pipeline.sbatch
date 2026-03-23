#!/bin/bash
#SBATCH --job-name=eval_pipe
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_2080|geforce_rtx_3090|a5000|a6000|l40s"
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/recipe_modification_extraction/logs/slurm_outputs/%j_eval_pipeline.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/recipe_modification_extraction/logs/slurm_outputs/%j_eval_pipeline.err

PROJ=/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2
source $PROJ/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp
export HF_HOME=$PROJ/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd $PROJ/recipe_modification_extraction

echo "============================================================"
echo "TWO-STEP PIPELINE EVALUATION"
echo "============================================================"
echo "Node: $(hostname), Start: $(date)"
echo ""

# ---- DictaBERT Pipeline ----
CLF="models/checkpoints/classifier_dictabert/best_model"
EXT="models/checkpoints/dictabert_twostep/best_model"
TEST="data/processed_dictabert/test.jsonl"

if [ -d "$CLF" ] && [ -d "$EXT" ] && [ -f "$TEST" ]; then
    echo "============================================================"
    echo "1. DictaBERT Pipeline"
    echo "============================================================"
    python -m src.evaluation.evaluate_pipeline \
        --classifier-path "$CLF" \
        --extractor-path "$EXT" \
        --test-file "$TEST" \
        --output-dir results/pipeline_dictabert \
        --bootstrap
else
    echo "SKIP DictaBERT: missing model or data"
    [ ! -d "$CLF" ] && echo "  Missing: $CLF"
    [ ! -d "$EXT" ] && echo "  Missing: $EXT"
    [ ! -f "$TEST" ] && echo "  Missing: $TEST"
fi

echo ""

# ---- AlephBertGimmel Pipeline ----
CLF="models/checkpoints/classifier_abg/best_model"
EXT="models/checkpoints/abg_twostep/best_model"
TEST="data/processed_abg/test.jsonl"

if [ -d "$CLF" ] && [ -d "$EXT" ] && [ -f "$TEST" ]; then
    echo "============================================================"
    echo "2. AlephBertGimmel Pipeline"
    echo "============================================================"
    python -m src.evaluation.evaluate_pipeline \
        --classifier-path "$CLF" \
        --extractor-path "$EXT" \
        --test-file "$TEST" \
        --output-dir results/pipeline_abg \
        --bootstrap
else
    echo "SKIP ABG: missing model or data"
fi

echo ""

# ---- DictaBERT-Large Pipeline ----
CLF="models/checkpoints/classifier_dicta_large/best_model"
EXT="models/checkpoints/dicta_large_twostep/best_model"
TEST="data/processed_dicta_large/test.jsonl"

if [ -d "$CLF" ] && [ -d "$EXT" ] && [ -f "$TEST" ]; then
    echo "============================================================"
    echo "3. DictaBERT-Large Pipeline"
    echo "============================================================"
    python -m src.evaluation.evaluate_pipeline \
        --classifier-path "$CLF" \
        --extractor-path "$EXT" \
        --test-file "$TEST" \
        --output-dir results/pipeline_dicta_large \
        --bootstrap
else
    echo "SKIP DictaBERT-Large: missing model or data"
fi

echo ""

# ---- NeoDictaBERT Pipeline ----
CLF="models/checkpoints/classifier_neodicta/best_model"
EXT="models/checkpoints/neodicta_twostep/best_model"
TEST="data/processed_neodicta/test.jsonl"

if [ -d "$CLF" ] && [ -d "$EXT" ] && [ -f "$TEST" ]; then
    echo "============================================================"
    echo "4. NeoDictaBERT Pipeline"
    echo "============================================================"
    python -m src.evaluation.evaluate_pipeline \
        --classifier-path "$CLF" \
        --extractor-path "$EXT" \
        --test-file "$TEST" \
        --output-dir results/pipeline_neodicta \
        --bootstrap
else
    echo "SKIP NeoDictaBERT: missing model or data"
fi

echo ""
echo "============================================================"
echo "ALL PIPELINE EVALUATIONS DONE: $(date)"
echo "============================================================"
