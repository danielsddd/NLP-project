#!/bin/bash
# =============================================================================
# SLURM Script: GPU Test
# =============================================================================
# Quick test to verify GPU access and environment setup
# Submit with: sbatch test_gpu.sbatch
# =============================================================================

#SBATCH --job-name=gpu_test
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_gpu_test.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/%u/logs/%j_gpu_test.err

# =============================================================================
# CONFIGURATION - Update these paths for your setup
# =============================================================================
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
PROJECT_DIR="$USER_DIR/recipe_modification_extraction"

echo "=========================================="
echo "GPU TEST - Recipe Modification Extraction"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "=========================================="

# Activate conda environment
source $USER_DIR/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp

# Set HuggingFace cache paths
export HF_HOME=$USER_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false

# Navigate to project
cd $PROJECT_DIR

# Run GPU test
python << 'EOF'
import sys
print("=" * 50)
print("ENVIRONMENT TEST")
print("=" * 50)

# Python version
print(f"Python: {sys.version}")

# PyTorch
try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test GPU computation
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        print(f"GPU Computation Test: PASSED")
except ImportError as e:
    print(f"PyTorch: NOT INSTALLED - {e}")

# Transformers
try:
    import transformers
    print(f"Transformers Version: {transformers.__version__}")
    
    # Test AlephBERT loading
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("onlplab/alephbert-base")
    print(f"AlephBERT Tokenizer: LOADED")
    print(f"Vocab Size: {tokenizer.vocab_size}")
    
    # Test Hebrew tokenization
    test_text = "הוספתי עוד כפית סוכר"
    tokens = tokenizer.tokenize(test_text)
    print(f"Hebrew Test: '{test_text}' -> {tokens}")
except ImportError as e:
    print(f"Transformers: NOT INSTALLED - {e}")
except Exception as e:
    print(f"Transformers Error: {e}")

# Seqeval
try:
    import seqeval
    print(f"Seqeval: INSTALLED")
except ImportError:
    print(f"Seqeval: NOT INSTALLED")

print("=" * 50)
print("TEST COMPLETE")
print("=" * 50)
EOF

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
