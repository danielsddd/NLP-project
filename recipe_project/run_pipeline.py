#!/usr/bin/env python3
"""
Main Pipeline Runner
====================
Run the complete recipe modification extraction pipeline.

Usage:
    python run_pipeline.py --step all
    python run_pipeline.py --step collect
    python run_pipeline.py --step label
    python run_pipeline.py --step preprocess
    python run_pipeline.py --step train
    python run_pipeline.py --step evaluate
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def run_command(cmd: list, description: str = ""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {cmd[0]}")
        return False


def check_prerequisites():
    """Check that required files and keys exist."""
    print("\n🔍 Checking prerequisites...")
    
    issues = []
    
    # Check API keys
    if not os.environ.get("YOUTUBE_API_KEY"):
        issues.append("YOUTUBE_API_KEY not set in .env")
    
    if not os.environ.get("GOOGLE_API_KEY"):
        issues.append("GOOGLE_API_KEY not set in .env (needed for Gemini)")
    
    # Check for .env file
    if not Path(".env").exists():
        issues.append(".env file not found. Copy from .env.example")
    
    if issues:
        print("\n⚠️  Prerequisites issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("   All prerequisites OK!")
    return True


def step_collect(args):
    """Step 1: Collect YouTube comments."""
    youtube_api_key = os.environ.get("YOUTUBE_API_KEY")
    
    if not youtube_api_key:
        print("❌ YOUTUBE_API_KEY not set!")
        return False
    
    # Check if collector exists
    collector_paths = [
        "youtube_collector/collect.py",
        "src/data_collection/collector.py",
    ]
    
    collector_path = None
    for path in collector_paths:
        if Path(path).exists():
            collector_path = path
            break
    
    if not collector_path:
        print("❌ YouTube collector not found!")
        print("   Make sure youtube_collector/collect.py exists")
        return False
    
    cmd = [
        sys.executable, collector_path,
        "--api-key", youtube_api_key,
        "--collect"
    ]
    
    return run_command(cmd, "YouTube Comment Collection")


def step_label(args):
    """Step 2: Generate silver labels with Teacher model."""
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not google_api_key:
        print("❌ GOOGLE_API_KEY not set!")
        return False
    
    input_file = args.input or "data/raw_youtube/comments.jsonl"
    
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        print("   Run --step collect first")
        return False
    
    cmd = [
        sys.executable, "-m", "src.teacher_labeling.generate_labels",
        "--input", input_file,
        "--provider", args.teacher or "gemini",
        "--api-key", google_api_key,
    ]
    
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    
    return run_command(cmd, "Silver Label Generation (Teacher)")


def step_preprocess(args):
    """Step 3: Preprocess data for training."""
    input_file = "data/silver_labels/teacher_output.jsonl"
    
    if not Path(input_file).exists():
        print(f"❌ Silver labels not found: {input_file}")
        print("   Run --step label first")
        return False
    
    cmd = [
        sys.executable, "-m", "src.preprocessing.prepare_data",
        "--input", input_file,
        "--output-dir", "data/processed",
        "--scheme", "BIO",
    ]
    
    return run_command(cmd, "Data Preprocessing (Span → BIO)")


def step_train(args):
    """Step 4: Train student model."""
    data_dir = "data/processed"
    
    if not Path(data_dir).exists() or not Path(f"{data_dir}/dataset_train.jsonl").exists():
        print(f"❌ Processed data not found: {data_dir}")
        print("   Run --step preprocess first")
        return False
    
    cmd = [
        sys.executable, "-m", "src.models.train_student",
        "--data-dir", data_dir,
        "--output-dir", args.output or "models/checkpoints/student",
        "--epochs", str(args.epochs or 5),
        "--batch-size", str(args.batch_size or 16),
        "--lr", str(args.lr or 2e-5),
    ]
    
    if args.fp16:
        cmd.append("--fp16")
    
    if args.max_examples:
        cmd.extend(["--max-examples", str(args.max_examples)])
    
    return run_command(cmd, "Student Model Training (AlephBERT)")


def step_baselines(args):
    """Step 5: Run baselines."""
    cmd = [
        sys.executable, "-m", "src.baselines.run_baselines",
        "--data-dir", "data/processed",
    ]
    
    return run_command(cmd, "Baseline Evaluation")


def step_evaluate(args):
    """Step 6: Evaluate trained model."""
    model_path = args.model_path or "models/checkpoints/student/best_model"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run --step train first")
        return False
    
    cmd = [
        sys.executable, "-m", "src.evaluation.evaluate",
        "--model-path", model_path,
        "--test-file", "data/processed/dataset_test.jsonl",
        "--output-dir", "results",
    ]
    
    if args.bootstrap:
        cmd.append("--bootstrap")
    
    return run_command(cmd, "Model Evaluation")


def main():
    parser = argparse.ArgumentParser(
        description="Run Recipe Modification Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --step all           # Run entire pipeline
  python run_pipeline.py --step collect       # Only collect data
  python run_pipeline.py --step label         # Only generate labels
  python run_pipeline.py --step train         # Only train model
  python run_pipeline.py --step evaluate      # Only evaluate
        """
    )
    
    # Step selection
    parser.add_argument(
        "--step",
        choices=["all", "collect", "label", "preprocess", "train", "baselines", "evaluate"],
        required=True,
        help="Pipeline step to run"
    )
    
    # Data collection options
    parser.add_argument("--input", help="Input file path")
    
    # Teacher labeling options
    parser.add_argument("--teacher", choices=["gemini", "openai"], default="gemini",
                       help="Teacher model provider")
    parser.add_argument("--limit", type=int, help="Limit examples to process")
    
    # Training options
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    parser.add_argument("--max-examples", type=int, help="Limit training examples")
    
    # Evaluation options
    parser.add_argument("--model-path", help="Path to trained model")
    parser.add_argument("--bootstrap", action="store_true", help="Compute bootstrap CI")
    
    args = parser.parse_args()
    
    # Create directories
    for dir_path in ["data/raw_youtube", "data/silver_labels", "data/processed", 
                     "models/checkpoints", "results", "logs"]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n⚠️  Fix the issues above and try again.")
        return 1
    
    # Run requested step(s)
    steps = {
        "collect": step_collect,
        "label": step_label,
        "preprocess": step_preprocess,
        "train": step_train,
        "baselines": step_baselines,
        "evaluate": step_evaluate,
    }
    
    if args.step == "all":
        # Run all steps in order
        for step_name, step_func in steps.items():
            print(f"\n{'#'*60}")
            print(f"# Running: {step_name}")
            print(f"{'#'*60}")
            
            success = step_func(args)
            if not success:
                print(f"\n❌ Pipeline stopped at step: {step_name}")
                return 1
        
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
    else:
        # Run single step
        success = steps[args.step](args)
        if not success:
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
