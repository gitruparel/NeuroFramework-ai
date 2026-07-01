import pytest
import numpy as np
from pathlib import Path
from training.calibration import calculate_ece, plot_reliability_diagram


def test_calculate_ece_perfect():
    """Verify ECE calculation on perfectly predictable deterministic distribution."""
    # 4 samples: 2 controls, 2 cases
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    
    # Expected ECE = 0.15 as calculated manually
    ece = calculate_ece(y_true, y_prob, n_bins=10)
    assert pytest.approx(ece, 1e-5) == 0.15


def test_calculate_ece_empty():
    """Verify ECE calculation on empty array returns 0.0."""
    ece = calculate_ece(np.array([]), np.array([]))
    assert ece == 0.0


def test_reliability_diagram_plot(tmp_path):
    """Verify that reliability diagram generates valid output image files."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.15, 0.25, 0.85, 0.95, 0.45, 0.75])
    
    output_png = tmp_path / "diagram.png"
    ece = plot_reliability_diagram(y_true, y_prob, output_png, n_bins=5)
    
    assert output_png.exists()
    assert ece > 0.0
