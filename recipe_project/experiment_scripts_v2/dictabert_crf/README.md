# dictabert_crf — All Experiment Scripts

**Model:** `dicta-il/dictabert` + CRF layer  
**Script:** `src.models.train_joint` (NOT train_student)

## Progressive Build-Up
| Script | What's ON |
|--------|-----------|
| `P0_crf_baseline` | CRF only, no weights |
| `P1_crf_weighted` | CRF + class weights |
| **`P2_crf_weighted`** | **CRF + weights (MAIN — compare with dictabert/P2)** |
| `P3_crf_downsample_2` | CRF + weights + ratio 2:1 |
| `P3_crf_downsample_4` | CRF + weights + ratio 4:1 |
| `P4_crf_thread_aware` | CRF + weights + thread context |
| `P5_crf_thread_no_enriched` | CRF + weights + thread - enriched |

## Key Comparison (A4: CRF contribution)
Compare `dictabert/P2_add_focal` vs `dictabert_crf/P2_crf_weighted`
Same encoder, same data — only difference is CRF vs softmax+focal.

## Ablation + Combo Scripts
Same A2-A8 and combo scripts as softmax models.
