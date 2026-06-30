"""Hyperparameter optimization runner using Optuna for systematic parameter search."""

import csv
import json
import logging
from pathlib import Path
from typing import Callable, Any, Dict
import numpy as np
import optuna

from core.logging import setup_logger

logger = setup_logger("training.hyperopt", "training/hyperopt.log")


def save_optimization_plots(study: optuna.Study, output_dir: Path) -> None:
    """Generates and saves optimization history and parameter importance plots.
    
    Uses standard optuna.visualization.matplotlib if available, falling back
    gracefully to basic matplotlib plots to prevent headless/missing dependency crashes.
    """
    import matplotlib.pyplot as plt
    
    # 1. Optimization History
    try:
        import optuna.visualization.matplotlib as vis
        # Clean figure environment
        plt.close("all")
        vis.plot_optimization_history(study)
        history_path = output_dir / "optimization_history.png"
        plt.savefig(history_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info(f"Saved optimization history plot to: {history_path}")
    except Exception as e:
        logger.warning(f"Could not generate Optuna visualization plots using standard modules: {e}. Generating fallback plot.")
        try:
            plt.close("all")
            plt.figure(figsize=(10, 6))
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
                trial_nums = [t.number for t in completed_trials]
                values = [t.value for t in completed_trials]
                running_max = np.maximum.accumulate(values)
                
                plt.plot(trial_nums, values, "o", label="Trial Value", color="royalblue")
                plt.plot(trial_nums, running_max, "-", label="Best Value", color="crimson", linewidth=2)
                plt.xlabel("Trial Number")
                plt.ylabel("Validation ROC-AUC")
                plt.title("Optimization History (Fallback)")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.7)
            else:
                plt.text(0.5, 0.5, "No completed trials yet", ha="center", va="center")
                
            history_path = output_dir / "optimization_history.png"
            plt.savefig(history_path, dpi=300, bbox_inches="tight")
            plt.close("all")
            logger.info(f"Saved fallback optimization history plot to: {history_path}")
        except Exception as ex:
            logger.error(f"Failed to generate fallback optimization history plot: {ex}")

    # 2. Parameter Importance
    try:
        import optuna.visualization.matplotlib as vis
        plt.close("all")
        # Only plot importance if there are completed trials with parameter variations
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) > 1:
            vis.plot_param_importances(study)
            importance_path = output_dir / "parameter_importance.png"
            plt.savefig(importance_path, dpi=300, bbox_inches="tight")
            plt.close("all")
            logger.info(f"Saved parameter importance plot to: {importance_path}")
        else:
            logger.info("Skipping parameter importance plot: need at least 2 completed trials.")
    except Exception as e:
        logger.warning(f"Could not generate Optuna parameter importance plot using standard modules: {e}. Generating fallback plot.")
        try:
            plt.close("all")
            plt.figure(figsize=(10, 6))
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed) > 1:
                plt.text(0.5, 0.5, "Parameter importance requires optuna.visualization or scikit-learn", ha="center", va="center")
            else:
                plt.text(0.5, 0.5, "Need at least 2 completed trials for parameter importance", ha="center", va="center")
                
            importance_path = output_dir / "parameter_importance.png"
            plt.savefig(importance_path, dpi=300, bbox_inches="tight")
            plt.close("all")
            logger.info(f"Saved fallback parameter importance plot to: {importance_path}")
        except Exception as ex:
            logger.error(f"Failed to generate fallback parameter importance plot: {ex}")


def optimize_hyperparameters(
    objective_fn: Callable[[optuna.Trial], float],
    n_trials: int,
    seed: int,
    output_dir: Path
) -> Dict[str, Any]:
    """Runs hyperparameter search utilizing Optuna, returning the best parameters dict.
    
    Generates trials logs in CSV and JSON formats, and creates diagnostic plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Enable study logging redirection to Optuna logger
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # Fix TPE sampler seed for complete reproducibility
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    logger.info(f"Starting Optuna hyperparameter study ({n_trials} trials, seed={seed}) to maximize ROC-AUC...")
    
    study.optimize(objective_fn, n_trials=n_trials)
    
    logger.info("Optuna hyperparameter study completed successfully.")
    
    # 1. Log best trial details
    best_trial = study.best_trial
    logger.info(f"Best Trial Number: {best_trial.number}")
    logger.info(f"Best Trial Value (Validation ROC-AUC): {best_trial.value:.4f}")
    logger.info(f"Best Parameters: {best_trial.params}")
    
    # 2. Save best configuration to JSON
    best_json_path = output_dir / "optuna_best.json"
    best_config = {
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "best_params": best_trial.params
    }
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=4)
    logger.info(f"Saved best optuna configuration details to: {best_json_path}")
    
    # 3. Save all trials details to CSV
    csv_path = output_dir / "optuna_trials.csv"
    try:
        trials = study.trials
        param_keys = set()
        for t in trials:
            param_keys.update(t.params.keys())
        param_keys = sorted(list(param_keys))
        
        fieldnames = ["trial_number", "state", "value"] + param_keys
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in trials:
                row = {
                    "trial_number": t.number,
                    "state": t.state.name,
                    "value": f"{t.value:.4f}" if t.value is not None else "N/A"
                }
                for k in param_keys:
                    row[k] = t.params.get(k, "N/A")
                writer.writerow(row)
        logger.info(f"Saved complete Optuna study trials log to: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to write optuna_trials.csv: {e}")
        
    # 4. Generate optimization history and importance plots
    save_optimization_plots(study, output_dir)
    
    return best_config
