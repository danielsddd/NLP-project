#!/usr/bin/env python3
"""
upload_to_hf.py
================
Uploads all project artifacts to HuggingFace.

Repos created:
  Dataset: DanielDDDs/recipe-modifications-v2
  Model:   DanielDDDs/hebrew-recipe-modification-ner

Run from recipe_project root inside tmux:
  tmux new -s hf_upload
  python3 scripts/local/upload_to_hf.py
"""

from huggingface_hub import HfApi
import os

api = HfApi()

DATASET_REPO = "DanielDDDS/recipe-modifications-v2"
MODEL_REPO   = "DanielDDDS/hebrew-recipe-modification-ner"
# ── Create repos ──────────────────────────────────────────────
print("Creating repos...")
api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True, private=False)
api.create_repo(repo_id=MODEL_REPO,   repo_type="model",   exist_ok=True, private=False)
print("  ✓ repos ready\n")

# ================================================================
# DATASET FILES
# ================================================================
dataset_files = {

    # ── Raw YouTube data ────────────────────────────────────────
    "data/raw_youtube/threads.jsonl":
        "raw/threads.jsonl",
    "data/raw_youtube/threads_positives_focus.jsonl":
        "raw/threads_positives_focus.jsonl",

    # ── V2 Silver labels (3-pass LLM annotation) ───────────────
    "data/silver_labels/teacher_output_v2.jsonl":
        "silver_labels/teacher_output_v2.jsonl",
    "data/silver_labels/threads_positives_focus_labeled_v2.jsonl":
        "silver_labels/threads_positives_focus_labeled_v2.jsonl",

    # ── Gold human annotations (corrected) ─────────────────────
    "data/gold_validation/gold_final_corrected.jsonl":
        "gold/gold_final_corrected.jsonl",

    # ── Final preprocessed data (IO+thread-aware) ───────────────
    "data/processed_thread_aware_io/train_merged.jsonl":
        "processed/train_merged.jsonl",
    "data/processed_thread_aware_io/val.jsonl":
        "processed/val.jsonl",
    "data/processed_thread_aware_io/test.jsonl":
        "processed/test.jsonl",
    "data/processed_thread_aware_io/id2label.json":
        "processed/id2label.json",
    "data/processed_thread_aware_io/label2id.json":
        "processed/label2id.json",
    "data/processed_thread_aware_io/stats_merged.json":
        "processed/stats_merged.json",

    # ── Teacher upper bound evaluation ──────────────────────────
    "results/teacher_upper_bound_corrected.json":
        "evaluation/teacher_upper_bound.json",

    # ── Best model gold + silver evaluation results ─────────────
    "results/dictabert_crf/P10_crf_thread_aware_io/gold/evaluation_results.json":
        "evaluation/best_model_gold_results.json",
    "results/dictabert_crf/P10_crf_thread_aware_io/silver/evaluation_results.json":
        "evaluation/best_model_silver_results.json",

    # ── Best model training summary ─────────────────────────────
    "models/checkpoints/dictabert_crf/P10_crf_thread_aware_io/training_summary.json":
        "evaluation/best_model_training_summary.json",

    # ── Ablation training summaries (proof of Table 6 numbers) ──
    "results/dictabert_crf/A1b_crf_uniform_weights/training_summary.json":
        "evaluation/ablations/A1b_uniform_weights.json",
    "results/dictabert_crf/A2_crf_downsample_2/training_summary.json":
        "evaluation/ablations/A2_downsample_2.json",
    "results/dictabert_crf/A2_crf_downsample_4/training_summary.json":
        "evaluation/ablations/A2_downsample_4.json",
    "results/dictabert_crf/A5_crf_data_25pct/training_summary.json":
        "evaluation/ablations/A5_data_25pct.json",
    "results/dictabert_crf/A5_crf_data_50pct/training_summary.json":
        "evaluation/ablations/A5_data_50pct.json",
    "results/dictabert_crf/A5_crf_data_75pct/training_summary.json":
        "evaluation/ablations/A5_data_75pct.json",
    "results/dictabert_crf/A6_crf_no_enriched/training_summary.json":
        "evaluation/ablations/A6_no_enriched.json",
    "results/dictabert_crf/A7_crf_io_scheme/training_summary.json":
        "evaluation/ablations/A7_io_scheme.json",
    "results/dictabert_crf/A8_crf_unanimous/training_summary.json":
        "evaluation/ablations/A8_unanimous.json",

    # ── P-series training summaries ─────────────────────────────
    "models/checkpoints/dictabert_crf/P0_crf_baseline/training_summary.json":
        "evaluation/p_series/P0_baseline.json",
    "models/checkpoints/dictabert_crf/P1_crf_weighted/training_summary.json":
        "evaluation/p_series/P1_weighted.json",
    "models/checkpoints/dictabert_crf/P2_crf_weighted/training_summary.json":
        "evaluation/p_series/P2_weighted.json",
    "models/checkpoints/dictabert_crf/P3_crf_downsample_2/training_summary.json":
        "evaluation/p_series/P3_downsample_2.json",
    "models/checkpoints/dictabert_crf/P3_crf_downsample_4/training_summary.json":
        "evaluation/p_series/P3_downsample_4.json",
    "models/checkpoints/dictabert_crf/P4_crf_thread_aware/training_summary.json":
        "evaluation/p_series/P4_thread_aware.json",
    "models/checkpoints/dictabert_crf/P10_crf_thread_aware_io/training_summary.json":
        "evaluation/p_series/P10_thread_aware_io.json",
    "models/checkpoints/dictabert_crf/P10v3_fixed_weights/training_summary.json":
        "evaluation/p_series/P10v3_fixed_weights.json",
    "models/checkpoints/dictabert_crf/P10v3_seed123/training_summary.json":
        "evaluation/p_series/P10v3_seed123.json",
    "models/checkpoints/dictabert_crf/P10v3_seed2026/training_summary.json":
        "evaluation/p_series/P10v3_seed2026.json",
    "models/checkpoints/dictabert_crf/P10v3_seed7777/training_summary.json":
        "evaluation/p_series/P10v3_seed7777.json",
    "models/checkpoints/dictabert_crf/P10v3_lr1e5/training_summary.json":
        "evaluation/p_series/P10v3_lr1e5.json",
    "models/checkpoints/dictabert_crf/P10v4_focal/training_summary.json":
        "evaluation/p_series/P10v4_focal.json",
}

print(f"=== UPLOADING DATASET ({len(dataset_files)} files) ===")
failed = []
for i, (local, remote) in enumerate(dataset_files.items(), 1):
    if not os.path.exists(local):
        print(f"  [{i:02d}/{len(dataset_files)}] SKIP (not found): {local}")
        continue
    try:
        print(f"  [{i:02d}/{len(dataset_files)}] {local} → {remote}")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )
        print(f"         ✓")
    except Exception as e:
        print(f"         ✗ FAILED: {e}")
        failed.append((local, str(e)))

print(f"\nDataset done. Failed: {len(failed)}")
if failed:
    for f, e in failed:
        print(f"  ✗ {f}: {e}")

# ================================================================
# MODEL FILES
# ================================================================
model_files = {
    "models/checkpoints/dictabert_crf/P10_crf_thread_aware_io/best_model.pt":
        "best_model.pt",
    "models/checkpoints/dictabert_crf/P10_crf_thread_aware_io/id2label.json":
        "id2label.json",
    "models/checkpoints/dictabert_crf/P10_crf_thread_aware_io/label2id.json":
        "label2id.json",
    "models/checkpoints/dictabert_crf/P10_crf_thread_aware_io/training_summary.json":
        "training_summary.json",
    "results/dictabert_crf/P10_crf_thread_aware_io/gold/evaluation_results.json":
        "evaluation/gold_results.json",
    "results/dictabert_crf/P10_crf_thread_aware_io/silver/evaluation_results.json":
        "evaluation/silver_results.json",
}

print(f"\n=== UPLOADING MODEL ({len(model_files)} files) ===")
print("NOTE: best_model.pt is 2.1GB — this will take ~15 minutes")
failed_model = []
for i, (local, remote) in enumerate(model_files.items(), 1):
    if not os.path.exists(local):
        print(f"  [{i:02d}/{len(model_files)}] SKIP (not found): {local}")
        continue
    try:
        print(f"  [{i:02d}/{len(model_files)}] {local} → {remote}")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=MODEL_REPO,
            repo_type="model",
        )
        print(f"         ✓")
    except Exception as e:
        print(f"         ✗ FAILED: {e}")
        failed_model.append((local, str(e)))

print(f"\nModel done. Failed: {len(failed_model)}")
if failed_model:
    for f, e in failed_model:
        print(f"  ✗ {f}: {e}")

print("\n" + "="*55)
print("ALL UPLOADS COMPLETE")
print("="*55)
print(f"Dataset: https://huggingface.co/datasets/{DATASET_REPO}")
print(f"Model:   https://huggingface.co/{MODEL_REPO}")
