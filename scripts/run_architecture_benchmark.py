#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

# Add project root to path if needed for local imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_single_experiment import run_experiment
from training.benchmark import aggregate_architecture_benchmark

def main():
    parser = argparse.ArgumentParser(description="Run baseline architecture benchmarking.")
    parser.add_argument("--data-root", default="data/raw", help="Path to raw dataset")
    parser.add_argument("--index-file", default="data/abide_index.json", help="Path to index JSON")
    parser.add_argument("--split-file", default="data/abide_splits.json", help="Path to train/val split JSON")
    parser.add_argument("--preprocessed-dir", default="data/processed", help="Path to preprocessed cache directory")
    parser.add_argument("--config-yaml", default="configs/preprocessing.yaml", help="Path to preprocessing config YAML")
    parser.add_argument("--experiment-dir", default="experiments/architecture_benchmark", help="Directory to save outputs")
    parser.add_argument("--device", default="auto", help="Execution device")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip dataset preprocessing")
    args = parser.parse_args()

    architectures = ["densenet121", "resnet10", "resnet18"]
    
    for idx, arch in enumerate(architectures):
        print(f"\n==================================================")
        print(f"BENCHMARKING ARCHITECTURE: {arch} ({idx + 1}/{len(architectures)})")
        print(f"==================================================")
        
        run_args = [
            "--data-root", args.data_root,
            "--index-file", args.index_file,
            "--split-file", args.split_file,
            "--preprocessed-dir", args.preprocessed_dir,
            "--config-yaml", args.config_yaml,
            "--experiment-dir", str(Path(args.experiment_dir) / arch),
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--architecture", arch,
            "--loss-function", "ce",
            "--experiment-name", f"arch_bench_{arch}"
        ]
        
        if args.limit is not None:
            run_args += ["--limit", str(args.limit)]
            
        if args.lr is not None:
            run_args += ["--lr", str(args.lr)]
            
        if args.seed is not None:
            run_args += ["--seed", str(args.seed)]
            
        # Re-use cache for runs index > 0 or if explicitly requested
        if idx > 0 or args.skip_preprocess:
            run_args.append("--skip-preprocess")
            
        result = run_experiment(run_args)
        if result.returncode != 0:
            print(f"Warning: Architecture {arch} failed with code {result.returncode}")

    print("\nAggregating architecture comparison metrics...")
    aggregate_architecture_benchmark(Path(args.experiment_dir), architectures)
    print(f"Architecture benchmark run finished. Comparison saved to: {Path(args.experiment_dir) / 'architecture_comparison.csv'}")

if __name__ == "__main__":
    main()
