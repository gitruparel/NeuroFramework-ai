import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

def run_benchmark():
    parser = argparse.ArgumentParser(description="Benchmark a preprocessing cache version on 10 epochs.")
    parser.add_argument("--cache-dir", required=True, help="Path to preprocessed cache version directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--experiment-name", default="benchmark_run", help="Name of experiment")
    
    args = parser.parse_args()
    
    cache_path = Path(args.cache_dir)
    if not cache_path.exists():
        print(f"Error: Cache directory {cache_path} does not exist.")
        sys.exit(1)
        
    pipeline_json = cache_path / "pipeline.json"
    pipeline_meta = {}
    if pipeline_json.exists():
        with open(pipeline_json, encoding="utf-8") as f:
            pipeline_meta = json.load(f)
            
    print(f"\n==================================================")
    print(f"BENCHMARKING CACHE VERSION: {cache_path.name.upper()}")
    print(f"==================================================")
    print(f"Pipeline Fingerprint: {json.dumps(pipeline_meta, indent=2)}")
    
    python_exe = sys.executable
    root_dir = Path(__file__).resolve().parent.parent
    
    exp_dir = root_dir / "experiments" / "benchmarks" / f"run_{cache_path.name}"
    
    cmd = [
        python_exe, str(root_dir / "training" / "train_autism.py"),
        "--architecture", "densenet121",
        "--loss-function", "focal",
        "--augmentation-profile", "minimal",
        "--epochs", str(args.epochs),

        "--batch-size", "16",
        "--lr", "2e-4",
        "--seed", "42",
        "--device", args.device,
        "--preprocessed-dir", str(cache_path.parent),  # Pass the parent processed directory
        "--skip-preprocess",
        "--experiment-dir", str(exp_dir),
        "--experiment-name", args.experiment_name

    ]
    
    # We must modify train_autism parameters if cache_version config doesn't match expected.
    # To bypass hash validation for benchmarking, we override validate_cache_data check or bypass.
    # Note: validate_cache_data returns a report. train_autism reads config_yaml.
    
    print("\nLaunching training baseline run...")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    res = subprocess.run(cmd, capture_output=False)
    elapsed_time = time.time() - start_time
    
    if res.returncode != 0:
        print("\nError: Training run failed. Benchmark aborted.")
        sys.exit(1)
        
    print("\nTraining completed successfully! Parsing metrics...")
    
    report_file = exp_dir / "classification_report.json"
    meta_file = exp_dir / "experiment_meta.json"
    
    metrics = {
        "cache_version": cache_path.name,
        "pipeline": pipeline_meta,
        "val_accuracy": 0.0,
        "val_roc_auc": 0.0,
        "val_pr_auc": 0.0,
        "val_f1": 0.0,
        "balanced_accuracy": 0.0,
        "training_time_sec": elapsed_time
    }
    
    if report_file.exists():
        with open(report_file, encoding="utf-8") as f:
            rep = json.load(f)
            
        # Parse metrics
        metrics["val_accuracy"] = rep.get("accuracy", 0.0)
        metrics["val_roc_auc"] = rep.get("roc_auc", 0.0)
        metrics["val_pr_auc"] = rep.get("pr_auc", 0.0)
        metrics["val_f1"] = rep.get("f1_score", 0.0)
        metrics["balanced_accuracy"] = rep.get("balanced_accuracy", 0.0)
        
    print("\n==================================================")
    print(f"BENCHMARK RESULTS FOR: {cache_path.name}")
    print(f"==================================================")
    print(f"Accuracy:          {metrics['val_accuracy']:.4f}")
    print(f"ROC-AUC:           {metrics['val_roc_auc']:.4f}")
    print(f"PR-AUC:            {metrics['val_pr_auc']:.4f}")
    print(f"F1-Score:          {metrics['val_f1']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Execution Time:    {elapsed_time:.1f} seconds")
    print(f"==================================================")
    
    # Save to benchmark.json
    bench_file = root_dir / "reports" / "benchmark.json"
    all_benchmarks = {}
    if bench_file.exists():
        try:
            with open(bench_file, encoding="utf-8") as f:
                all_benchmarks = json.load(f)
        except Exception:
            pass
            
    all_benchmarks[cache_path.name] = metrics
    with open(bench_file, "w", encoding="utf-8") as f:
        json.dump(all_benchmarks, f, indent=4)
        
    print(f"Saved benchmark data entry to: {bench_file}")

if __name__ == "__main__":
    run_benchmark()
