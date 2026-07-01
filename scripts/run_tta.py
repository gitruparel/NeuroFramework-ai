#!/usr/bin/env python
import argparse
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score, balanced_accuracy_score

# Add project root to path if needed for local imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.abide import ABIDEDataset
from schemas.dataset import collate_dataset_samples
from models.factory import ModelFactory
from utils.device import resolve_backend, load_state_dict_flexible
from training.inference import TestTimeAugmentor, PredictionAggregator, aggregate_tta_comparison

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained checkpoint with and without Test-Time Augmentation.")
    parser.add_argument("--architecture", required=True, help="Backbone model architecture")
    parser.add_argument("--checkpoint-path", required=True, help="Path to best_model.pt checkpoint")
    parser.add_argument("--data-root", default="data/raw", help="Path to raw dataset")
    parser.add_argument("--index-file", default="data/abide_index.json", help="Path to index JSON")
    parser.add_argument("--split-file", default="data/abide_splits.json", help="Path to train/val split JSON")
    parser.add_argument("--preprocessed-dir", default="data/processed", help="Path to preprocessed cache directory")
    parser.add_argument("--config-yaml", default="configs/preprocessing.yaml", help="Path to preprocessing config YAML")
    parser.add_argument("--experiment-dir", default="experiments/tta_evaluation", help="Directory to save outputs")
    parser.add_argument("--device", default="auto", help="Execution device")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for validation inference")
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Classification decision threshold")
    parser.add_argument("--tta-runs", type=int, default=5, help="Number of TTA runs")
    parser.add_argument("--tta-method", default="mean", choices=["mean", "median", "majority"], help="TTA aggregation method")
    parser.add_argument("--limit", type=int, default=None, help="Limit validation subject count for smoke test")
    args = parser.parse_args()

    # Create output directory
    exp_dir = Path(args.experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve Device Backend
    backend = resolve_backend(args.device)
    device = backend.device
    print(f"Using device backend: {device}")

    # 2. Build Dataset
    label_map = {"CONTROL": 0, "ASD": 1}
    val_dataset = ABIDEDataset(
        index_file=Path(args.index_file),
        split_file=Path(args.split_file),
        split_name="val",
        preprocessed_dir=Path(args.preprocessed_dir),
        raw_dir=Path(args.data_root),
        label_map=label_map
    )
    if args.limit is not None:
        val_dataset.items = val_dataset.items[:args.limit]
    print(f"Loaded validation samples: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=backend.capabilities.pin_memory,
        collate_fn=collate_dataset_samples
    )

    # 3. Instantiate Model
    print(f"Constructing model: {args.architecture}")
    model = ModelFactory.create_model(args.architecture, dropout_prob=0.0)
    
    # 4. Load weights
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
        
    print(f"Loading best weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_state_dict_flexible(model, checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Wrap in DataParallel if multiple GPUs are available
    eval_model = model
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs for fast evaluation.")
        eval_model = torch.nn.DataParallel(model)

    # 5. Define Inference Loop
    def run_inference_loop(tta_augmentor=None, runs=1):
        y_true_list = []
        y_pred_list = []
        y_prob_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["image"].to(device, non_blocking=backend.capabilities.non_blocking)
                targets = batch["label"].to(device, non_blocking=backend.capabilities.non_blocking)
                
                if tta_augmentor is not None and runs > 1:
                    batch_probs = []
                    for r in range(runs):
                        augmented_inputs = []
                        for i in range(inputs.size(0)):
                            augmented_inputs.append(tta_augmentor.get_augmentations(inputs[i], r))
                        augmented_batch = torch.stack(augmented_inputs, dim=0)
                        outputs = eval_model(augmented_batch)
                        probs = torch.softmax(outputs, dim=1)
                        batch_probs.append(probs)
                        
                    aggregated_probs = PredictionAggregator.aggregate(batch_probs, method=args.tta_method)
                    preds = (aggregated_probs[:, 1] >= args.decision_threshold).long()
                    
                    y_true_list.extend(targets.cpu().numpy())
                    y_pred_list.extend(preds.cpu().numpy())
                    y_prob_list.extend(aggregated_probs.cpu().numpy())
                else:
                    outputs = eval_model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    preds = (probs[:, 1] >= args.decision_threshold).long()
                    
                    y_true_list.extend(targets.cpu().numpy())
                    y_pred_list.extend(preds.cpu().numpy())
                    y_prob_list.extend(probs.cpu().numpy())
                    
        return np.array(y_true_list), np.array(y_pred_list), np.array(y_prob_list)

    # Helper function to compile performance dict
    def compute_metrics(y_true_arr, y_pred_arr, y_prob_arr):
        accuracy = float(np.mean(y_true_arr == y_pred_arr))
        
        sensitivity = 0.0
        specificity = 0.0
        if len(np.unique(y_true_arr)) > 1:
            cm = confusion_matrix(y_true_arr, y_pred_arr)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            
        roc_auc = 0.5
        pr_auc = 0.5
        if len(np.unique(y_true_arr)) > 1:
            roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr[:, 1]))
            pr_auc = float(average_precision_score(y_true_arr, y_prob_arr[:, 1]))
            
        bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
        report = classification_report(y_true_arr, y_pred_arr, target_names=["Control", "Autism"] if len(np.unique(y_true_arr)) > 1 else None, output_dict=True, zero_division=0)
        macro_f1 = float(report.get('macro avg', {}).get('f1-score', 0.0))
        
        return {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "macro_f1": macro_f1,
            "balanced_accuracy": bal_acc,
            "sensitivity": sensitivity,
            "specificity": specificity
        }

    # 6. Run Baseline Evaluation
    print("Running baseline evaluation...")
    start = time.time()
    y_true_base, y_pred_base, y_prob_base = run_inference_loop(tta_augmentor=None, runs=1)
    baseline_latency = time.time() - start
    baseline_metrics = compute_metrics(y_true_base, y_pred_base, y_prob_base)
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f} | ROC-AUC: {baseline_metrics['roc_auc']:.4f} | Latency: {baseline_latency:.2f}s")

    # 7. Run TTA Evaluation
    print(f"Running Test-Time Augmentation evaluation ({args.tta_runs} runs)...")
    start = time.time()
    tta_augmentor = TestTimeAugmentor(seed=42)
    y_true_tta, y_pred_tta, y_prob_tta = run_inference_loop(tta_augmentor=tta_augmentor, runs=args.tta_runs)
    tta_latency = time.time() - start
    tta_metrics = compute_metrics(y_true_tta, y_pred_tta, y_prob_tta)
    print(f"TTA Accuracy: {tta_metrics['accuracy']:.4f} | ROC-AUC: {tta_metrics['roc_auc']:.4f} | Latency: {tta_latency:.2f}s")

    # 8. Compile comparison reports
    aggregate_tta_comparison(
        baseline_metrics=baseline_metrics,
        tta_metrics=tta_metrics,
        baseline_latency=baseline_latency,
        tta_latency=tta_latency,
        output_dir=exp_dir
    )
    print(f"TTA evaluation finished successfully! Results saved to {exp_dir / 'tta_comparison.csv'}")

if __name__ == "__main__":
    main()
