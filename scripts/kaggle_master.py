#!/usr/bin/env python
import argparse
import sys
import os
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Master Kaggle Experiment Launcher.")
    parser.add_argument(
        "stage",
        choices=["architecture", "loss", "augmentation", "tta", "cv"],
        help="Pipeline stage to run."
    )
    
    # Selected winners from prior stages (passed as inputs to subsequent stages)
    parser.add_argument("--architecture", help="Winning model architecture (required for loss, augmentation, tta, cv)")
    parser.add_argument("--loss-function", help="Winning loss function (required for augmentation, tta, cv)")
    parser.add_argument("--augmentation-profile", help="Winning augmentation profile (required for tta, cv)")
    parser.add_argument("--checkpoint-path", help="Path to best_model.pt (required for tta)")
    parser.add_argument("--tta", action="store_true", help="Enable TTA (for cv stage)")
    
    # Resource configurations
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for quick tests")
    parser.add_argument("--device", default="auto", help="Execution device")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    # Path configuration overrides (with defaults pointing to Kaggle environment)
    parser.add_argument("--data-root", default="/kaggle/input/neuroframework-data", help="Raw dataset root directory")
    parser.add_argument("--index-file", default="data/abide_index.json", help="Path to index JSON")
    parser.add_argument("--split-file", default="data/abide_splits.json", help="Path to split JSON")
    parser.add_argument("--kfold-file", default="data/abide_kfold.json", help="Path to KFold split JSON")
    parser.add_argument("--preprocessed-dir", default="/kaggle/temp/cache/abide", help="Preprocessed cache folder")
    parser.add_argument("--experiment-dir", default="/kaggle/working/experiments", help="Base directory for experiment outputs")
    
    args = parser.parse_args()
    python_exe = sys.executable
    
    # 1. Setup PYTHONPATH env
    env = os.environ.copy()
    root_dir = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root_dir + (os.pathsep + env.get("PYTHONPATH", "")) if env.get("PYTHONPATH") else root_dir
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = []
    
    # 2. Stage Execution Routing
    if args.stage == "architecture":
        print(">>> Starting Stage 1: Architecture Benchmark Sweep...")
        cmd = [
            python_exe, str(Path(root_dir) / "scripts/run_architecture_benchmark.py"),
            "--data-root", args.data_root,
            "--index-file", args.index_file,
            "--split-file", args.split_file,
            "--preprocessed-dir", args.preprocessed_dir,
            "--experiment-dir", str(Path(args.experiment_dir) / "architecture"),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", args.device
        ]
        
    elif args.stage == "loss":
        if not args.architecture:
            print("Error: --architecture argument is required for loss benchmark stage.")
            sys.exit(1)
        print(f">>> Starting Stage 2: Loss Function Sweep for {args.architecture}...")
        cmd = [
            python_exe, str(Path(root_dir) / "scripts/run_loss_benchmark.py"),
            "--architecture", args.architecture,
            "--data-root", args.data_root,
            "--index-file", args.index_file,
            "--split-file", args.split_file,
            "--preprocessed-dir", args.preprocessed_dir,
            "--experiment-dir", str(Path(args.experiment_dir) / "loss"),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", args.device
        ]
        
    elif args.stage == "augmentation":
        if not args.architecture or not args.loss_function:
            print("Error: --architecture and --loss-function arguments are required for augmentation stage.")
            sys.exit(1)
        print(f">>> Starting Stage 3: Augmentation Sweep for {args.architecture} with {args.loss_function}...")
        cmd = [
            python_exe, str(Path(root_dir) / "scripts/run_augmentation_benchmark.py"),
            "--architecture", args.architecture,
            "--loss-function", args.loss_function,
            "--data-root", args.data_root,
            "--index-file", args.index_file,
            "--split-file", args.split_file,
            "--preprocessed-dir", args.preprocessed_dir,
            "--experiment-dir", str(Path(args.experiment_dir) / "augmentation"),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", args.device
        ]
        
    elif args.stage == "tta":
        if not args.architecture or not args.checkpoint_path:
            print("Error: --architecture and --checkpoint-path arguments are required for TTA evaluation stage.")
            sys.exit(1)
        print(f">>> Starting Stage 4: Inference TTA evaluation for {args.architecture}...")
        cmd = [
            python_exe, str(Path(root_dir) / "scripts/run_tta.py"),
            "--architecture", args.architecture,
            "--checkpoint-path", args.checkpoint_path,
            "--data-root", args.data_root,
            "--index-file", args.index_file,
            "--split-file", args.split_file,
            "--preprocessed-dir", args.preprocessed_dir,
            "--experiment-dir", str(Path(args.experiment_dir) / "tta"),
            "--batch-size", str(args.batch_size),
            "--device", args.device
        ]
        
    elif args.stage == "cv":
        if not args.architecture or not args.loss_function:
            print("Error: --architecture and --loss-function arguments are required for cross-validation stage.")
            sys.exit(1)
        print(f">>> Starting Stage 5: 5-Fold Cross-Validation for {args.architecture}...")
        cmd = [
            python_exe, str(Path(root_dir) / "scripts/run_final_cv.py"),
            "--architecture", args.architecture,
            "--loss-function", args.loss_function,
            "--augmentation-profile", args.augmentation_profile or "none",
            "--data-root", args.data_root,
            "--index-file", args.index_file,
            "--split-file", args.split_file,
            "--kfold-file", args.kfold_file,
            "--preprocessed-dir", args.preprocessed_dir,
            "--experiment-dir", str(Path(args.experiment_dir) / "final_cv"),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", args.device
        ]
        if args.tta:
            cmd.append("--tta")

    # Forward common arguments
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.lr is not None and args.stage != "tta":
        cmd += ["--lr", str(args.lr)]
    if args.seed is not None and args.stage != "tta":
        cmd += ["--seed", str(args.seed)]

    # Execute
    print(f"Executing subprocess command:\n{' '.join(cmd)}\n")
    res = subprocess.run(cmd, env=env)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
