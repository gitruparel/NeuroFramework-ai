"""Metrics manager computing standard evaluation metrics for classification tasks."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any


class MetricsManager:
    """Computes standard model evaluation metrics, keeping execution loops decoupled from metric libraries."""

    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> Dict[str, float]:
        """Calculates accuracy, precision, recall, f1, and roc_auc.

        Args:
            y_true: Ground-truth integer labels of shape (N,).
            y_pred: Predicted class integer labels of shape (N,).
            y_prob: Class probabilities of shape (N,) or (N, C). If None, defaults roc_auc to 0.5.
        """
        metrics = {}
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

        unique_classes = np.unique(y_true)
        is_binary = len(unique_classes) <= 2
        average = "binary" if is_binary else "macro"

        metrics["precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))

        if y_prob is not None:
            try:
                # If binary class, check if probabilities are 2D or 1D
                if is_binary:
                    if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
                        prob_for_auc = y_prob[:, 1]
                    else:
                        prob_for_auc = y_prob
                    metrics["roc_auc"] = float(roc_auc_score(y_true, prob_for_auc))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
            except Exception:
                # Fail-safe (e.g. if only 1 class is present in y_true)
                metrics["roc_auc"] = 0.5
        else:
            metrics["roc_auc"] = 0.5

        return metrics
