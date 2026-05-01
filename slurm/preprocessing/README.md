# Preprocessing Scripts

Run these BEFORE the training scripts that need them.

| # | Script | Output Dir | Description |
|---|--------|-----------|-------------|
| 01 | `01_preprocess_default.sbatch` | `data/processed/` | Default: ratio 3.0, BIO, merged |
| 02 | `02_preprocess_ratio_2.sbatch` | `data/processed_ds2/` | A2: ratio 2:1 |
| 03 | `03_preprocess_ratio_4.sbatch` | `data/processed_ds4/` | A2: ratio 4:1 |
| 04 | `04_preprocess_no_enriched.sbatch` | `data/processed_no_enriched/` | A6: original only |
| 05 | `05_preprocess_io_scheme.sbatch` | `data/processed_io/` | A7: IO tags |
| 06 | `06_preprocess_unanimous.sbatch` | `data/processed_unanimous/` | A8: 3/3 only |
| 07 | `07_preprocess_thread_aware.sbatch` | `data/processed_thread_aware/` | Thread context |
| 08 | `08_preprocess_thread_aware_no_enriched.sbatch` | `data/processed_thread_aware_no_enriched/` | Thread + no enriched |

## Dependency: `07_preprocess_thread_aware` requires `src/preprocessing/prepare_thread_aware.py`
This file is included in the `helper_code/` folder of this zip. Copy it to your project.
