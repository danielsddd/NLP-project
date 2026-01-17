# SLURM Scripts for TAU Cluster

## Overview

These scripts are designed for the TAU CS SLURM cluster with the following partitions:
- `studentkillable` - Main partition for training (can be preempted)
- `studentbatch` - Limited slots, higher priority
- `studentrun` - Interactive sessions

## Quick Start

### 1. First-time Setup
```bash
# On the cluster, run once:
bash setup_cluster.sh
```

### 2. GPU Test
```bash
sbatch test_gpu.sbatch
```

### 3. Full Pipeline
```bash
# Use the master script:
bash run_experiments.sh status    # Check project status
bash run_experiments.sh train     # Train AlephBERT
bash run_experiments.sh baseline  # Train mBERT
bash run_experiments.sh evaluate  # Evaluate all
```

## Script Descriptions

| Script | Purpose | GPU | Time |
|--------|---------|-----|------|
| `setup_cluster.sh` | One-time environment setup | No | 5 min |
| `test_gpu.sbatch` | Verify GPU access | Yes | 15 min |
| `preprocess_data.sbatch` | Convert silver→BIO | No | 2 hours |
| `train_alephbert.sbatch` | Train main model | Yes | 6 hours |
| `train_mbert.sbatch` | Train baseline | Yes | 4 hours |
| `evaluate_model.sbatch` | Evaluate models | Yes | 1 hour |
| `ablation_data_size.sbatch` | Data size study | Yes | 4h × 6 jobs |
| `run_experiments.sh` | Master workflow | N/A | N/A |

## Common Commands

```bash
# Submit a job
sbatch script.sbatch

# Check your jobs
squeue -u $USER

# Cancel a job
scancel <job_id>

# View job output
cat /vol/joberant_nobck/data/NLP_368307701_2526a/$USER/logs/<job_id>_*.out

# Interactive GPU session (debugging)
srun --partition=studentrun --gres=gpu:1 --mem=16G --time=02:00:00 --pty bash

# Check available GPUs
sinfo -p studentkillable
```

## Directory Structure on Cluster

```
/vol/joberant_nobck/data/NLP_368307701_2526a/<username>/
├── recipe_modification_extraction/
│   ├── data/
│   │   ├── raw_youtube/          # YouTube comments
│   │   ├── silver_labels/        # Teacher outputs
│   │   ├── processed/            # BIO-format data
│   │   └── gold_validation/      # Human annotations
│   ├── models/
│   │   ├── checkpoints/student/  # AlephBERT model
│   │   ├── baselines/mbert/      # mBERT baseline
│   │   └── ablations/            # Ablation models
│   ├── results/                  # Evaluation results
│   ├── src/                      # Source code
│   └── scripts/slurm/            # These scripts
├── .cache/
│   └── huggingface/              # Model cache (symlinked)
├── logs/                         # SLURM logs
└── anaconda3/                    # Conda installation
```

## Troubleshooting

### "Disk quota exceeded"
```bash
# Check what's taking space
du -sh /vol/joberant_nobck/data/NLP_368307701_2526a/$USER/*

# Verify cache symlinks
ls -la ~/.cache/huggingface
# Should point to /vol/... not /home/...

# Clean old checkpoints
rm -rf models/checkpoints/checkpoint-*
```

### "CUDA not available"
```bash
# Check if you requested GPU
grep "gres=gpu" your_script.sbatch

# Check available GPUs
sinfo -p studentkillable -o "%N %G"
```

### "ModuleNotFoundError"
```bash
# Activate conda first
source /vol/joberant_nobck/data/NLP_368307701_2526a/$USER/anaconda3/etc/profile.d/conda.sh
conda activate recipe_nlp

# Verify packages
pip list | grep transformers
```

## Estimated Timeline

| Phase | Duration | Jobs |
|-------|----------|------|
| Data collection | 2-3 days | Local (API) |
| Teacher labeling | 2 days | Local (API) |
| Preprocessing | 2 hours | 1 job |
| AlephBERT training | 4-6 hours | 1 job |
| mBERT training | 3-4 hours | 1 job |
| Evaluation | 1 hour | 1 job |
| Ablation studies | 4 hours | 6 parallel jobs |

**Total cluster time: ~15-20 hours**
