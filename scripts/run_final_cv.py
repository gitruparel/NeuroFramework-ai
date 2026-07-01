#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

# Add project root to path if needed for local imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_single_experiment import run_experiment

def main():
    parser = argparse.ArgumentParser(description="Run 5-Fold Cross-Validation on final optimized configuration.")
    parser.add_argument("--architecture", required=True, help="Backbone model architecture")
    parser.add_argument("--loss-function", required=True, help="Loss function to apply (e.g. focal)")
    parser.add_argument("--augmentation-profile", default="none", help="MONAI augmentation profile (minimal, moderate, strong, research, or none)")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation during fold evaluations")
    parser.add_argument("--data-root", default="data/raw", help="Path to raw dataset")
    parser.add_argument("--index-file", default="data/abide_index.json", help="Path to index JSON")
    parser.add_argument("--split-file", default="data/abide_splits.json", help="Path to train/val split JSON")
    parser.add_argument("--kfold-file", default="data/abide_kfold.json", help="Path to JSON containing stratified fold splits")
    parser.add_argument("--preprocessed-dir", default="data/processed", help="Path to preprocessed cache directory")
    parser.add_argument("--config-yaml", default="configs/preprocessing.yaml", help="Path to preprocessing config YAML")
    parser.add_argument("--experiment-dir", default="experiments/final_cv", help="Directory to save CV outputs")
    parser.add_argument("--device", default="auto", help="Execution device")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train per fold")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit subject count for testing")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    # Construct arguments list for CV run
    run_args = [
        "--data-root", args.data_root,
        "--index-file", args.index_file,
        "--split-file", args.split_file,
        "--kfold-file", args.kfold_file,
        "--preprocessed-dir", args.preprocessed_dir,
        "--config-yaml", args.config_yaml,
        "--experiment-dir", args.experiment_dir,
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--architecture", args.architecture,
        "--loss-function", args.loss_function,
        "--skip-preprocess",  # CV benchmark always reuses cache
        "--cv",
        "--experiment-name", f"final_cv_{args.architecture}"
    ]

    # Handle augmentation profiles
    if args.augmentation_profile != "none":
        run_args += ["--augment", "--augmentation-profile", args.augmentation_profile]
        
    # Handle Test-Time Augmentation
    if args.tta:
        run_args.append("--tta")

    # Limit dataset sizes if requested
    if args.limit is not None:
        run_args += ["--limit", str(args.limit)]
        
    if args.lr is not None:
        run_args += ["--lr", str(args.lr)]
        
    if args.seed is not None:
        run_args += ["--seed", str(args.seed)]

    print(f"\n==================================================")
    print(f"LAUNCHING 5-FOLD CROSS-VALIDATION PIPELINE")
    print(f"Config: Arch={args.architecture} | Loss={args.loss_function} | Aug={args.augmentation_profile} | TTA={args.tta}")
    print(f"==================================================")

    result = run_experiment(run_args)
    if result.returncode == 0:
        print(f"\n5-Fold CV completed successfully! Check outputs in: {args.experiment_dir}")
    else:
        print(f"\nError: Cross-validation pipeline failed with exit code: {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
