# Experiment Scripts V2 — Complete Method Combinations

## 157+ SLURM scripts, organized by model

Every model folder contains:
- **P0→P5**: Progressive build-up (baseline → +weights → +focal → +downsample → +thread-aware)
- **A1→A8**: Isolated ablation toggles
- **combo_***: Multi-method combinations
- **evaluate_all**: Evaluate every trained variant

## Quick Start
```bash
bash setup_experiment_structure.sh              # Create dirs, copy files
sbatch scripts/slurm/preprocessing/01_preprocess_default.sbatch   # Default data
sbatch scripts/slurm/preprocessing/07_preprocess_thread_aware.sbatch  # Thread data
bash scripts/slurm/submit_all.sh main           # All 7 models, main config
bash scripts/slurm/submit_all.sh progressive    # Full P0-P5 for best model
bash scripts/slurm/submit_all.sh ablations      # All ablations for best model
bash scripts/slurm/submit_all.sh evaluate       # Evaluate everything
```

## Method Progression (per model)

```
P0: Plain CE (baseline)
 └─→ P1: + Class weights
      └─→ P2: + Focal loss  ★ MAIN CONFIG
           ├─→ P3a: + Downsample ratio 2:1
           ├─→ P3b: + Downsample ratio 4:1
           ├─→ P4: + Thread-aware context
           │    └─→ P5: + Thread-aware - enriched
           └─→ CRF (dictabert_crf only)
```

## All Methods Covered

| Method | How It's Tested | Scripts |
|--------|----------------|---------|
| Class weights | P0 vs P1 | `P0_baseline` vs `P1_add_weights` |
| Focal loss | P1 vs P2 | `P1_add_weights` vs `P2_add_focal` |
| CRF layer | P2 vs CRF/P2 | `dictabert/P2` vs `dictabert_crf/P2` |
| Downsample ratio | P2 vs P3 | `P2` vs `P3_downsample_2/4` |
| Thread-aware | P2 vs P4 | `P2` vs `P4_add_thread_aware` |
| Enriched data | P2 vs A6 | `P2` vs `A6_no_enriched` |
| IO vs BIO | P2 vs A7 | `P2` vs `A7_io_scheme` |
| Data size | P2 vs A5 | `P2` vs `A5_data_25/50/75pct` |
| Label quality | P2 vs A8 | `P2` vs `A8_unanimous` |
| Weight type | A1a vs A1b vs A1c | Three-way comparison |
| Combos | Various | `combo_*` scripts |
