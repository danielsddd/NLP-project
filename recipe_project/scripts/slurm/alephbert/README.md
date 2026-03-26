# alephbert — All Experiment Scripts

**Model:** `onlplab/alephbert-base`  
**Script:** `src.models.train_student`  
**Batch:** 16 | **Mem:** 24G | **Time:** 08:00:00

## Progressive Build-Up (submit these first)
| Script | What's ON | What's New |
|--------|-----------|------------|
| `P0_baseline` | Plain CE | Nothing — pure baseline |
| `P1_add_weights` | + class weights | Inverse-frequency weighting |
| **`P2_add_focal`** | **+ weights + focal** | **Focal loss γ=2.0 (MAIN)** |
| `P3_add_downsample_2` | + weights + focal + ds2 | Ratio 2:1 data |
| `P3_add_downsample_4` | + weights + focal + ds4 | Ratio 4:1 data |
| `P4_add_thread_aware` | + weights + focal + thread ctx | Question→answer context |
| `P5_thread_aware_no_enriched` | + weights + focal + thread - enriched | Isolation test |

## Ablation Toggles (remove one thing from MAIN)
| Script | Ablation | What Changes |
|--------|----------|-------------|
| `A1a_no_weights` | A1 | Remove all weights |
| `A1b_uniform_weights` | A1 | Uniform weights (sanity) |
| `A1c_weights_only` | A1/A3 | Weights but no focal |
| `A2_downsample_2` | A2 | Data ratio 2:1 |
| `A2_downsample_4` | A2 | Data ratio 4:1 |
| `A5_data_25pct` | A5 | 25% training data |
| `A5_data_50pct` | A5 | 50% training data |
| `A5_data_75pct` | A5 | 75% training data |
| `A6_no_enriched` | A6 | Without enriched data |
| `A7_io_scheme` | A7 | IO tags (not BIO) |
| `A8_unanimous` | A8 | 3/3 agreement only |

## Combo Scripts (stack multiple changes)
| Script | Description |
|--------|-------------|
| `combo_thread_aware_io` | Thread ctx + IO scheme |
| `combo_thread_aware_ds2` | Thread ctx + downsample 2:1 |
| `combo_no_enriched_no_weights` | Worst realistic config |

## A3 (focal loss) and A4 (CRF) Notes
- **A3**: Compare `P1_add_weights` vs `P2_add_focal` — same model, focal on/off
- **A4**: Compare `alephbert/P2_add_focal` vs `dictabert_crf/P2_crf_weighted` — CRF on/off
