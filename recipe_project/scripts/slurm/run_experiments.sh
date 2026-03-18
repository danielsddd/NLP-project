#!/bin/bash
# =============================================================================
# Master Experiment Runner (FIXED)
# =============================================================================
# FIX: Changed all dataset_train.jsonl/dataset_val.jsonl/dataset_test.jsonl
#      references to train.jsonl/val.jsonl/test.jsonl to match prepare_data.py output
# =============================================================================

set -e

# Configuration
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction"
SCRIPTS_DIR="$PROJECT_DIR/scripts/slurm"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() { echo -e "\n${GREEN}=== $1 ===${NC}"; }
print_error()  { echo -e "${RED}ERROR: $1${NC}"; }
print_warning(){ echo -e "${YELLOW}WARNING: $1${NC}"; }

submit_job() {
    local script="$1"
    local name="$2"
    if [ ! -f "$script" ]; then
        print_error "Script not found: $script"
        return 1
    fi
    echo "Submitting: $name"
    sbatch "$script"
}

# =============================================================================
# PIPELINE STEPS
# =============================================================================

step_test() {
    print_header "Step: GPU Test"
    submit_job "$SCRIPTS_DIR/test_gpu.sbatch" "GPU Test"
}

step_preprocess() {
    print_header "Step: Preprocess Data"

    if [ ! -f "$PROJECT_DIR/data/silver_labels/teacher_output.jsonl" ]; then
        print_error "Silver labels not found!"
        echo "Run teacher labeling first (locally):"
        echo "  python -m src.teacher_labeling.generate_labels \\"
        echo "      --input data/raw_youtube/comments.jsonl \\"
        echo "      --provider gemini"
        return 1
    fi

    submit_job "$SCRIPTS_DIR/preprocess_data.sbatch" "Data Preprocessing"
}

step_train() {
    print_header "Step: Train AlephBERT Student"

    # FIX: was dataset_train.jsonl, now train.jsonl
    if [ ! -f "$PROJECT_DIR/data/processed/train.jsonl" ]; then
        print_error "Processed data not found!"
        echo "Run preprocessing first: bash $0 preprocess"
        return 1
    fi

    submit_job "$SCRIPTS_DIR/train_alephbert.sbatch" "AlephBERT Training"
}

step_baseline() {
    print_header "Step: Train mBERT Baseline"

    # FIX: was dataset_train.jsonl, now train.jsonl
    if [ ! -f "$PROJECT_DIR/data/processed/train.jsonl" ]; then
        print_error "Processed data not found!"
        return 1
    fi

    submit_job "$SCRIPTS_DIR/train_mbert.sbatch" "mBERT Baseline Training"
}

step_evaluate() {
    print_header "Step: Evaluate Models"
    submit_job "$SCRIPTS_DIR/evaluate_model.sbatch" "Model Evaluation"
}

step_ablation() {
    print_header "Step: Data Size Ablation Study"

    # FIX: was dataset_train.jsonl, now train.jsonl
    if [ ! -f "$PROJECT_DIR/data/processed/train.jsonl" ]; then
        print_error "Processed data not found!"
        return 1
    fi

    submit_job "$SCRIPTS_DIR/ablation_data_size.sbatch" "Ablation Study (6 jobs)"
}

step_all() {
    print_header "Running Complete Pipeline"

    echo ""
    echo "This will submit jobs in sequence:"
    echo "  1. Preprocess data"
    echo "  2. Train AlephBERT"
    echo "  3. Train mBERT baseline"
    echo "  4. Evaluate all models"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        return 1
    fi

    # Submit preprocessing
    step_preprocess
    echo ""
    print_warning "After preprocessing completes, submit training jobs manually:"
    echo "  bash $0 train"
    echo "  bash $0 baseline"
    echo ""
    print_warning "After training completes, run evaluation:"
    echo "  bash $0 evaluate"
}

# =============================================================================
# STATUS CHECK
# =============================================================================

show_status() {
    print_header "Project Status"

    echo ""
    echo "Current jobs:"
    squeue -u $USER

    echo ""
    echo "Data files:"
    # FIX: was dataset_train.jsonl etc., now train.jsonl etc.
    for f in "data/raw_youtube/comments.jsonl" \
             "data/silver_labels/teacher_output.jsonl" \
             "data/processed/train.jsonl" \
             "data/processed/val.jsonl" \
             "data/processed/test.jsonl"; do
        if [ -f "$PROJECT_DIR/$f" ]; then
            count=$(wc -l < "$PROJECT_DIR/$f")
            echo "  ✓ $f ($count lines)"
        else
            echo "  ✗ $f (not found)"
        fi
    done

    echo ""
    echo "Models:"
    for d in "models/checkpoints/student/best_model" \
             "models/baselines/mbert/best_model"; do
        if [ -d "$PROJECT_DIR/$d" ]; then
            echo "  ✓ $d"
        else
            echo "  ✗ $d (not trained)"
        fi
    done

    echo ""
    echo "Results:"
    for f in "results/alephbert/evaluation_results.json" \
             "results/mbert/evaluation_results.json" \
             "results/baselines/baseline_results.json"; do
        if [ -f "$PROJECT_DIR/$f" ]; then
            echo "  ✓ $f"
        else
            echo "  ✗ $f (not generated)"
        fi
    done
}

# =============================================================================
# MAIN
# =============================================================================

print_header "Recipe Modification Extraction - Experiment Runner"
echo "Project: $PROJECT_DIR"
echo ""

case "${1:-help}" in
    test)
        step_test
        ;;
    preprocess)
        step_preprocess
        ;;
    train)
        step_train
        ;;
    baseline)
        step_baseline
        ;;
    evaluate)
        step_evaluate
        ;;
    ablation)
        step_ablation
        ;;
    all)
        step_all
        ;;
    status)
        show_status
        ;;
    help|*)
        echo "Usage: bash $0 [command]"
        echo ""
        echo "Commands:"
        echo "  test       - Run GPU test"
        echo "  preprocess - Preprocess silver labels to BIO format"
        echo "  train      - Train AlephBERT student model"
        echo "  baseline   - Train mBERT baseline"
        echo "  evaluate   - Evaluate all models"
        echo "  ablation   - Run data size ablation (6 parallel jobs)"
        echo "  all        - Run complete pipeline"
        echo "  status     - Show project status"
        echo ""
        echo "Typical workflow:"
        echo "  1. Run teacher labeling locally (needs API key)"
        echo "  2. bash $0 preprocess"
        echo "  3. bash $0 train"
        echo "  4. bash $0 baseline"
        echo "  5. bash $0 evaluate"
        echo "  6. bash $0 ablation  (optional)"
        ;;
esac

echo ""