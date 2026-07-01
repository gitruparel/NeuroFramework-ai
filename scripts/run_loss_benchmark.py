#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

# Add project root to path if needed for local imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_single_experiment import run_experiment
from training.losses import aggregate_losses_benchmark

def main():
    parser = argparse.ArgumentParser(description="Run loss function benchmarking on selected architecture.")
    parser.add_argument("--architecture", required=True, help="Backbone architecture to evaluate (e.g. resnet18)")
    parser.add_argument("--data-root", default="data/raw", help="Path to raw dataset")
    parser.add_argument("--index-file", default="data/abide_index.json", help="Path to index JSON")
    parser.add_argument("--split-file", default="data/abide_splits.json", help="Path to train/val split JSON")
    parser.add_argument("--preprocessed-dir", default="data/processed", help="Path to preprocessed cache directory")
    parser.add_argument("--config-yaml", default="configs/preprocessing.yaml", help="Path to preprocessing config YAML")
    parser.add_argument("--experiment-dir", default="experiments/loss_benchmark", help="Directory to save outputs")
    parser.add_argument("--device", default="auto", help="Execution device")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()

    losses = ["ce", "weighted_ce", "focal", "ce_ls", "focal_ls"]
    
    for idx, loss in enumerate(losses):
        print(f"\n==================================================")
        print(f"BENCHMARKING LOSS FUNCTION: {loss} ({idx + 1}/{len(losses)})")
        print(f"==================================================")
        
        run_args = [
            "--data-root", args.data_root,
            "--index-file", args.index_file,
            "--split-file", args.split_file,
            "--preprocessed-dir", args.preprocessed_dir,
            "--config-yaml", args.config_yaml,
            "--experiment-dir", str(Path(args.experiment_dir) / loss),
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--architecture", args.architecture,
            "--loss-function", loss,
            "--skip-preprocess",  # Loss sweep always reuses the preprocessed cache
            "--experiment-name", f"loss_bench_{args.architecture}_{loss}"
        ]
        
        if args.limit is not None:
            run_args += ["--limit", str(args.limit)]
            
        result = run_experiment(run_args)
        if result.returncode != 0:
            print(f"Warning: Loss {loss} failed with code {result.returncode}")

    print("\nAggregating loss comparison metrics...")
    aggregate_losses_benchmark(Path(args.experiment_dir), losses)
    print(f"Loss benchmark run finished. Comparison saved to: {Path(args.experiment_dir) / 'loss_comparison.csv'}")

if __name__ == "__main__":
    main()
