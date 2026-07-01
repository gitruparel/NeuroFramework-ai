"""Expected Calibration Error (ECE) and reliability plotting functions."""

from pathlib import Path
import numpy as np


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculates the Expected Calibration Error (ECE) for binary classification."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    if len(y_true) == 0:
        return 0.0
        
    # In binary classification, the confidence is the probability of the predicted class
    pred_labels = (y_prob >= 0.5).astype(int)
    confidences = np.where(pred_labels == 1, y_prob, 1.0 - y_prob)
    accuracies = (pred_labels == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find indices of samples falling in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if i == 0:
            in_bin = in_bin | (confidences == bin_lower)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path, n_bins: int = 10) -> float:
    """Generates reliability diagram and ECE metric annotation, saving to output_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    pred_labels = (y_prob >= 0.5).astype(int)
    confidences = np.where(pred_labels == 1, y_prob, 1.0 - y_prob)
    accuracies = (pred_labels == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if i == 0:
            in_bin = in_bin | (confidences == bin_lower)
            
        bin_size = np.sum(in_bin)
        bin_counts.append(bin_size)
        if bin_size > 0:
            bin_accuracies.append(np.mean(accuracies[in_bin]))
            bin_confidences.append(np.mean(confidences[in_bin]))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((bin_lower + bin_upper) / 2.0)
            
    ece = calculate_ece(y_true, y_prob, n_bins)
    
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 1. Reliability Plot
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    
    # Draw bars for bins with samples
    bin_midpoints = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0
    widths = 1.0 / n_bins
    
    ax1.bar(
        bin_midpoints, bin_accuracies, width=widths * 0.9, color="blue", alpha=0.7,
        edgecolor="black", label="Accuracy"
    )
    
    # Highlight gaps
    gaps = np.abs(np.array(bin_confidences) - np.array(bin_accuracies))
    ax1.bar(
        bin_midpoints, gaps, bottom=bin_accuracies, width=widths * 0.9, color="red", alpha=0.3,
        edgecolor="red", linestyle="--", label="Gap"
    )
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Reliability Diagram (ECE = {ece:.4f})", fontsize=14, fontweight="semibold")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # 2. Sample count histogram
    ax2.bar(bin_midpoints, bin_counts, width=widths * 0.9, color="dimgray", edgecolor="black", alpha=0.8)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Samples Count")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    
    return ece
