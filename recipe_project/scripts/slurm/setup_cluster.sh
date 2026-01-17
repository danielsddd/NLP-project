#!/bin/bash
# =============================================================================
# Setup Script: Initialize Project on Cluster
# =============================================================================
# Run this ONCE when you first set up the project on the cluster
# This creates directories, symlinks cache, and verifies environment
# Run directly (not via sbatch): bash setup_cluster.sh
# =============================================================================

echo "=========================================="
echo "CLUSTER SETUP: Recipe Modification Extraction"
echo "=========================================="
echo "User: $USER"
echo "Date: $(date)"
echo "=========================================="

# =============================================================================
# CONFIGURATION - Update this for your setup
# =============================================================================
USER_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
PROJECT_NAME="recipe_modification_extraction"
PROJECT_DIR="$USER_DIR/$PROJECT_NAME"
CONDA_ENV_NAME="recipe_nlp"

# =============================================================================
# STEP 1: Create Directory Structure
# =============================================================================
echo ""
echo "Step 1: Creating directory structure..."

mkdir -p $USER_DIR
mkdir -p $PROJECT_DIR/{data,models,logs,results,scripts/slurm}
mkdir -p $PROJECT_DIR/data/{raw_youtube,silver_labels,processed,gold_validation}
mkdir -p $PROJECT_DIR/models/{checkpoints/student,baselines/mbert,ablations}
mkdir -p $PROJECT_DIR/results/{alephbert,mbert,ablations,figures,tables}
mkdir -p $USER_DIR/.cache/{huggingface,torch}
mkdir -p $USER_DIR/logs

echo "  Created: $PROJECT_DIR"
ls -la $PROJECT_DIR

# =============================================================================
# STEP 2: Symlink HuggingFace Cache (CRITICAL)
# =============================================================================
echo ""
echo "Step 2: Setting up HuggingFace cache symlink..."

# Remove existing cache if it's not a symlink
if [ -d "$HOME/.cache/huggingface" ] && [ ! -L "$HOME/.cache/huggingface" ]; then
    echo "  Moving existing cache to persistent storage..."
    mv $HOME/.cache/huggingface/* $USER_DIR/.cache/huggingface/ 2>/dev/null
    rm -rf $HOME/.cache/huggingface
fi

# Create symlink
if [ ! -L "$HOME/.cache/huggingface" ]; then
    mkdir -p $HOME/.cache
    ln -sf $USER_DIR/.cache/huggingface $HOME/.cache/huggingface
    echo "  Created symlink: ~/.cache/huggingface -> $USER_DIR/.cache/huggingface"
else
    echo "  Symlink already exists"
fi

# Same for torch cache
if [ ! -L "$HOME/.cache/torch" ]; then
    mkdir -p $HOME/.cache
    rm -rf $HOME/.cache/torch 2>/dev/null
    ln -sf $USER_DIR/.cache/torch $HOME/.cache/torch
    echo "  Created symlink: ~/.cache/torch -> $USER_DIR/.cache/torch"
fi

# Verify
ls -la $HOME/.cache/ | grep -E "huggingface|torch"

# =============================================================================
# STEP 3: Create/Verify Conda Environment
# =============================================================================
echo ""
echo "Step 3: Setting up Conda environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "  Conda not found. Installing Miniconda..."
    cd $USER_DIR
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $USER_DIR/anaconda3
    rm Miniconda3-latest-Linux-x86_64.sh
    
    # Add to path
    export PATH="$USER_DIR/anaconda3/bin:$PATH"
    echo "  Miniconda installed"
fi

# Initialize conda
source $USER_DIR/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null

# Check if environment exists
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "  Environment '$CONDA_ENV_NAME' already exists"
else
    echo "  Creating conda environment '$CONDA_ENV_NAME'..."
    conda create -n $CONDA_ENV_NAME python=3.10 -y
fi

# Activate and install packages
echo "  Installing packages..."
conda activate $CONDA_ENV_NAME

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.30.0 seqeval>=1.2.2 
pip install google-generativeai google-api-python-client
pip install pyyaml python-dotenv tqdm numpy scipy
pip install tensorboard

echo "  Packages installed"

# =============================================================================
# STEP 4: Create .env Template
# =============================================================================
echo ""
echo "Step 4: Creating .env template..."

if [ ! -f "$PROJECT_DIR/.env" ]; then
    cat > $PROJECT_DIR/.env << 'ENVFILE'
# API Keys - Fill these in!
YOUTUBE_API_KEY=your_youtube_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional backup

# Cluster paths (auto-configured)
USER_DIR=/vol/joberant_nobck/data/NLP_368307701_2526a/$USER
PROJECT_DIR=$USER_DIR/recipe_modification_extraction
HF_HOME=$USER_DIR/.cache/huggingface
ENVFILE
    echo "  Created .env template at $PROJECT_DIR/.env"
    echo "  ⚠️  IMPORTANT: Edit this file and add your API keys!"
else
    echo "  .env already exists"
fi

# =============================================================================
# STEP 5: Verify Setup
# =============================================================================
echo ""
echo "=========================================="
echo "VERIFICATION"
echo "=========================================="

echo ""
echo "Directory structure:"
du -sh $PROJECT_DIR/*

echo ""
echo "Cache symlinks:"
ls -la $HOME/.cache/ | grep -E "huggingface|torch"

echo ""
echo "Conda environment:"
conda list | head -20

echo ""
echo "Storage usage:"
du -sh $USER_DIR

echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit $PROJECT_DIR/.env with your API keys"
echo "2. Upload your code: rsync -avz ./src/ $USER@nova:$PROJECT_DIR/src/"
echo "3. Run GPU test: sbatch scripts/slurm/test_gpu.sbatch"
echo "4. Start data collection (locally with API keys)"
echo ""
