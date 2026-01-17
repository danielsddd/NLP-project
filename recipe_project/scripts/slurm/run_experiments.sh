#!/bin/bash
# =============================================================================
# Master Workflow Script: Recipe Modification Extraction
# =============================================================================
# This script shows the complete pipeline and can submit jobs in sequence
# 
# Usage:
#   bash run_experiments.sh [step]
#
# Steps:
#   test      - Run GPU test only
#   preprocess - Preprocess data (after teacher labeling)
#   train     - Train AlephBERT student
#   baseline  - Train mBERT baseline  
#   evaluate  - Evaluate all models
#   ablation  - Run data size ablation study
#   all       - Run everything (preprocess -> train -> baseline -> evaluate)
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION
# =============================================================================
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction"
SCRIPTS_DIR="$PROJECT_DIR/scripts/slurm"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_header() {
    echo ""
    echo -e "${GREEN}=========================================="
    echo "$1"
    echo -e "==========================================${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_file() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        return 1
    fi
    return 0
}

submit_job() {
    local script=$1
    local description=$2
    
    print_header "Submitting: $description"
    
    if [ ! -f "$script" ]; then
        print_error "Script not found: $script"
        return 1
    fi
    
    job_id=$(sbatch $script | awk '{print $4}')
    echo "Job submitted: $job_id"
    echo "Monitor with: squeue -j $job_id"
    echo "Logs at: $USER_DIR/logs/${job_id}_*.out"
    
    return 0
}

wait_for_job() {
    local job_id=$1
    echo "Waiting for job $job_id to complete..."
    
    while squeue -j $job_id 2>/dev/null | grep -q $job_id; do
        sleep 30
        echo "  Still running..."
    done
    
    echo "Job $job_id completed"
}

# =============================================================================
# PIPELINE STEPS
# =============================================================================

step_test() {
    print_header "Step: GPU Test"
    submit_job "$SCRIPTS_DIR/test_gpu.sbatch" "GPU Test"
}

step_preprocess() {
    print_header "Step: Data Preprocessing"
    
    # Check if silver labels exist
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
    
    # Check if processed data exists
    if [ ! -f "$PROJECT_DIR/data/processed/dataset_train.jsonl" ]; then
        print_error "Processed data not found!"
        echo "Run preprocessing first: bash $0 preprocess"
        return 1
    fi
    
    submit_job "$SCRIPTS_DIR/train_alephbert.sbatch" "AlephBERT Training"
}

step_baseline() {
    print_header "Step: Train mBERT Baseline"
    
    if [ ! -f "$PROJECT_DIR/data/processed/dataset_train.jsonl" ]; then
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
    
    if [ ! -f "$PROJECT_DIR/data/processed/dataset_train.jsonl" ]; then
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
    for f in "data/raw_youtube/comments.jsonl" \
             "data/silver_labels/teacher_output.jsonl" \
             "data/processed/dataset_train.jsonl" \
             "data/processed/dataset_val.jsonl" \
             "data/processed/dataset_test.jsonl"; do
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
             "results/baseline_results.json"; do
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
