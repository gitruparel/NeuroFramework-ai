#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

# Add project root to path if needed for local imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_single_experiment import run_experiment
from training.augmentations import aggregate_augmentations_benchmark

def main():
    parser = argparse.ArgumentParser(description="Run data augmentation profile benchmarking.")
    parser.add_argument("--architecture", required=True, help="Backbone architecture to evaluate (e.g. resnet18)")
    parser.add_argument("--loss-function", required=True, help="Loss function to apply (e.g. focal)")
    parser.add_argument("--data-root", default="data/raw", help="Path to raw dataset")
    parser.add_argument("--index-file", default="data/abide_index.json", help="Path to index JSON")
    parser.add_argument("--split-file", default="data/abide_splits.json", help="Path to train/val split JSON")
    parser.add_argument("--preprocessed-dir", default="data/processed", help="Path to preprocessed cache directory")
    parser.add_argument("--config-yaml", default="configs/preprocessing.yaml", help="Path to preprocessing config YAML")
    parser.add_argument("--experiment-dir", default="experiments/augmentation_benchmark", help="Directory to save outputs")
    parser.add_argument("--device", default="auto", help="Execution device")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    profiles = ["none", "minimal", "moderate", "strong", "research"]
    
    for idx, profile in enumerate(profiles):
        print(f"\n==================================================")
        print(f"BENCHMARKING AUGMENTATION PROFILE: {profile} ({idx + 1}/{len(profiles)})")
        print(f"==================================================")
        
        run_args = [
            "--data-root", args.data_root,
            "--index-file", args.index_file,
            "--split-file", args.split_file,
            "--preprocessed-dir", args.preprocessed_dir,
            "--config-yaml", args.config_yaml,
            "--experiment-dir", str(Path(args.experiment_dir) / profile),
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--architecture", args.architecture,
            "--loss-function", args.loss_function,
            "--skip-preprocess",  # Always reuse cache
            "--experiment-name", f"aug_bench_{args.architecture}_{profile}"
        ]
        
        if profile != "none":
            run_args += ["--augment", "--augmentation-profile", profile]
            
        if args.limit is not None:
            run_args += ["--limit", str(args.limit)]
            
        if args.lr is not None:
            run_args += ["--lr", str(args.lr)]
            
        if args.seed is not None:
            run_args += ["--seed", str(args.seed)]
            
        result = run_experiment(run_args)
        if result.returncode != 0:
            print(f"Warning: Augmentation profile {profile} failed with code {result.returncode}")

    print("\nAggregating augmentation comparison metrics...")
    aggregate_augmentations_benchmark(Path(args.experiment_dir), profiles)
    print(f"Augmentation benchmark run finished. Comparison saved to: {Path(args.experiment_dir) / 'augmentation_comparison.csv'}")

if __name__ == "__main__":
    main()
