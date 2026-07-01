"""Out-of-Fold prediction assembly, threshold tuning, and CV report generation."""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from core.logging import setup_logger

logger = setup_logger("training.evaluation", "training/evaluation.log")


def assemble_oof_predictions(experiment_dir: Path, n_folds: int = 5) -> pd.DataFrame:
    """Assembles individual fold predictions into a central oof_predictions.csv file."""
    all_dfs = []
    
    for f_idx in range(n_folds):
        pred_csv = experiment_dir / f"fold_{f_idx}" / "predictions.csv"
        if pred_csv.exists():
            try:
                df = pd.read_csv(pred_csv)
                df["Fold"] = f_idx
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to read predictions for fold {f_idx} from {pred_csv}: {e}")
        else:
            logger.warning(f"Prediction file for fold {f_idx} not found at {pred_csv}")
            
    if not all_dfs:
        raise FileNotFoundError("No individual fold predictions.csv files found to assemble OOF predictions.")
        
    oof_df = pd.concat(all_dfs, ignore_index=True)
    oof_csv_path = experiment_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_csv_path, index=False)
    logger.info(f"Successfully compiled OOF predictions to: {oof_csv_path}")
    return oof_df


def tune_optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path) -> Dict[str, Any]:
    """Tunes optimal classification thresholds based on Out-of-Fold predictions."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    best_f1_th = 0.5
    best_f1_val = 0.0
    best_bal_acc_th = 0.5
    best_bal_acc_val = 0.0
    best_youden_th = 0.5
    best_youden_val = -1.0
    
    thresholds = np.linspace(0.01, 0.99, 99)
    
    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        
        # F1
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1_val:
            best_f1_val = float(f1)
            best_f1_th = float(th)
            
        # Balanced Accuracy
        bal_acc = balanced_accuracy_score(y_true, preds)
        if bal_acc > best_bal_acc_val:
            best_bal_acc_val = float(bal_acc)
            best_bal_acc_th = float(th)
            
        # Youden's J
        cm = confusion_matrix(y_true, preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            youden = sens + spec - 1.0
            if youden > best_youden_val:
                best_youden_val = float(youden)
                best_youden_th = float(th)
                
    optimal_thresholds = {
        "best_f1_threshold": best_f1_th,
        "best_f1_value": best_f1_val,
        "best_balanced_accuracy_threshold": best_bal_acc_th,
        "best_balanced_accuracy_value": best_bal_acc_val,
        "best_youden_threshold": best_youden_th,
        "best_youden_value": best_youden_val
    }
    
    th_path = output_dir / "optimal_thresholds.json"
    with open(th_path, "w", encoding="utf-8") as f:
        json.dump(optimal_thresholds, f, indent=4)
        
    logger.info(f"Saved optimal classification thresholds to: {th_path}")
    return optimal_thresholds


def generate_cv_report(experiment_dir: Path, n_folds: int = 5) -> None:
    """Aggregates metrics from individual fold metadata files, generates cv_summary.csv and cv_summary.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fold_metrics = []
    
    metric_keys = {
        "best_val_accuracy": "Accuracy",
        "best_val_roc_auc": "ROC_AUC",
        "best_val_pr_auc": "PR_AUC",
        "best_val_macro_f1": "Macro_F1",
        "best_val_balanced_accuracy": "Balanced_Accuracy",
        "best_val_sensitivity": "Sensitivity",
        "best_val_specificity": "Specificity"
    }
    
    for f_idx in range(n_folds):
        meta_path = experiment_dir / f"fold_{f_idx}" / "experiment_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                fold_data = {"Fold": f_idx}
                for json_k, csv_k in metric_keys.items():
                    fold_data[csv_k] = meta.get(json_k, 0.0)
                fold_metrics.append(fold_data)
            except Exception as e:
                logger.error(f"Failed to read metadata for fold {f_idx}: {e}")
                
    if not fold_metrics:
        logger.warning("No fold metadata found to aggregate for CV report.")
        return
        
    df = pd.DataFrame(fold_metrics)
    
    # Calculate Mean and Std for each metric
    summary_data = []
    cols_to_summarize = list(metric_keys.values())
    
    mean_row = {"Fold": "Mean"}
    std_row = {"Fold": "Std"}
    
    for col in cols_to_summarize:
        mean_row[col] = df[col].mean()
        std_row[col] = df[col].std()
        
    summary_df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    csv_path = experiment_dir / "cv_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved CV summary statistics to: {csv_path}")
    
    # Generate grouped comparative metrics plot
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = [mean_row[col] for col in cols_to_summarize]
    stds = [std_row[col] for col in cols_to_summarize]
    
    # Replace NaN standard deviations with 0.0 (if only 1 fold present)
    stds = [s if not np.isnan(s) else 0.0 for s in stds]
    
    x_pos = np.arange(len(cols_to_summarize))
    ax.bar(
        x_pos, means, yerr=stds, align='center', alpha=0.8, ecolor='black', capsize=10,
        color=['royalblue', 'seagreen', 'orange', 'purple', 'crimson', 'teal', 'darkviolet'],
        edgecolor='black', width=0.5
    )
    
    ax.set_ylabel('Score / Value')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cols_to_summarize, rotation=15)
    ax.set_title('5-Fold Cross Validation Evaluation Metrics (Mean ± Std Dev)', fontsize=14, fontweight="semibold")
    ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Add values on top of bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f"{m:.4f}\n±{s:.4f}", ha='center', va='bottom', fontsize=9)
        
    plot_path = experiment_dir / "cv_summary.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved CV metrics comparison plot to: {plot_path}")
