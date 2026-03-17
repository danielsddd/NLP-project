#!/usr/bin/env python3
"""
Pipeline Runner — v4
Run the complete recipe modification extraction pipeline.

Usage:
    python run_pipeline.py --step collect
    python run_pipeline.py --step label
    python run_pipeline.py --step preprocess
    python run_pipeline.py --step train
    python run_pipeline.py --step baselines
    python run_pipeline.py --step evaluate
    python run_pipeline.py --step all
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def run(cmd, name):
    """Run a command, return success bool."""
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    print(f"  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    ok = result.returncode == 0
    print(f"\n  {'✅' if ok else '❌'} {name} {'completed' if ok else 'FAILED'}")
    return ok

def step_collect(args):
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        print("❌ YOUTUBE_API_KEY not set in .env")
        return False
    return run([sys.executable, "youtube_collector/collect.py",
                "--api-key", key, "--collect", "--target", str(args.target)],
               "YouTube Thread Collection")

def step_label(args):
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("❌ GOOGLE_API_KEY not set in .env")
        return False
    input_file = "data/raw_youtube/threads_cooking_only.jsonl"
    if not Path(input_file).exists():
        input_file = "data/raw_youtube/threads.jsonl"
    if not Path(input_file).exists():
        print(f"❌ No threads file found. Run --step collect first.")
        return False
    cmd = [sys.executable, "-m", "src.teacher_labeling.generate_labels",
           "--input", input_file, "--api-key", key]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    return run(cmd, "Teacher Labeling (Gemini)")

def step_preprocess(args):
    input_file = "data/silver_labels/teacher_output.jsonl"
    if not Path(input_file).exists():
        print(f"❌ {input_file} not found. Run --step label first.")
        return False
    return run([sys.executable, "-m", "src.preprocessing.prepare_data",
                "--input", input_file, "--output-dir", "data/processed",
                "--scheme", args.scheme],
               "Preprocessing (Span → BIO)")

def step_train(args):
    if not Path("data/processed/train.jsonl").exists():
        print("❌ Processed data not found. Run --step preprocess first.")
        return False
    cmd = [sys.executable, "-m", "src.models.train_student",
           "--data-dir", "data/processed",
           "--output-dir", args.output_dir or "models/checkpoints/student",
           "--model", args.model,
           "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
           "--lr", str(args.lr), "--seed", str(args.seed)]
    if args.fp16:
        cmd.append("--fp16")
    if args.max_examples:
        cmd.extend(["--max-examples", str(args.max_examples)])
    return run(cmd, f"Student Training ({args.model})")

def step_baselines(args):
    if not Path("data/processed/test.jsonl").exists():
        print("❌ Processed data not found. Run --step preprocess first.")
        return False
    return run([sys.executable, "-m", "src.baselines.run_baselines",
                "--data-dir", "data/processed"],
               "Baselines")

def step_evaluate(args):
    model_path = args.model_path or "models/checkpoints/student/best_model"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}. Run --step train first.")
        return False
    cmd = [sys.executable, "-m", "src.evaluation.evaluate",
           "--model-path", model_path,
           "--test-file", "data/processed/test.jsonl",
           "--output-dir", "results"]
    if args.bootstrap:
        cmd.append("--bootstrap")
    return run(cmd, "Evaluation")

# =============================================================================
# MAIN
# =============================================================================

STEPS = {
    "collect": step_collect,
    "label": step_label,
    "preprocess": step_preprocess,
    "train": step_train,
    "baselines": step_baselines,
    "evaluate": step_evaluate,
}
STEP_ORDER = ["collect", "label", "preprocess", "train", "baselines", "evaluate"]

def main():
    parser = argparse.ArgumentParser(description="Recipe Modification Extraction Pipeline v4")
    parser.add_argument("--step", required=True,
                        choices=list(STEPS.keys()) + ["all"],
                        help="Pipeline step to run")
    # Collection
    parser.add_argument("--target", type=int, default=5000)
    # Labeling
    parser.add_argument("--limit", type=int, help="Limit threads to label")
    # Preprocessing
    parser.add_argument("--scheme", choices=["BIO", "IO"], default="BIO")
    # Training
    parser.add_argument("--model", default="onlplab/alephbert-base")
    parser.add_argument("--output-dir", help="Model output directory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-examples", type=int, help="Limit training examples")
    # Evaluation
    parser.add_argument("--model-path", help="Path to trained model")
    parser.add_argument("--bootstrap", action="store_true")

    args = parser.parse_args()

    # Create directories
    for d in ["data/raw_youtube", "data/silver_labels", "data/processed",
              "models/checkpoints", "results", "logs/slurm_outputs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if args.step == "all":
        for name in STEP_ORDER:
            if not STEPS[name](args):
                print(f"\n❌ Pipeline stopped at step: {name}")
                return
        print("\n✅ Full pipeline completed!")
    else:
        STEPS[args.step](args)

if __name__ == "__main__":
    main()