# Kaggle Experimental Workflow & Execution Guide

This document outlines the exact sequential workflow to execute your MRI ASD classification benchmarks on Kaggle.

Since you have uploaded the **preprocessed cache** to Kaggle, you do not need raw NIfTI scans at all! All commands have been updated to point to your preprocessed cache directory `/kaggle/input/datasets/swayamruparel/abide-cache` and use the `--skip-preprocess` flag.

---

## 🚀 Environment Setup

Copy and run this cell first to clone the repo, change directory, install dependencies, and initialize folder structures:

```python
import os

# Clone the repository
REPO_URL = "https://github.com/gitruparel/NeuroFramework-ai.git"
if not os.path.exists("NeuroFramework-ai"):
    print("Cloning repository...")
    !git clone {REPO_URL}
else:
    print("Repository folder exists. Pulling latest updates...")
    !cd NeuroFramework-ai && git pull

# Change into the repository
%cd NeuroFramework-ai

# Install required packages (MONAI, TorchIO, SimpleITK, Optuna, etc.)
print("\nInstalling dependencies...")
!pip install --quiet -r requirements.txt

# Initialize folders
print("\nInitializing folders...")
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/external", exist_ok=True)
os.makedirs("data/interim", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("experiments", exist_ok=True)

print("\nSetup complete! You are ready to start training.")
```

---

## 📊 Summary of Benchmark Progression

| Stage | Command Stage Identifier | Epochs | Batch Size | Primary Purpose |
|---|---|---|---|---|
| **1. Architecture Benchmark** | `architecture` | 15 | 24 | Find the best CNN backbone |
| **2. Loss Benchmark** | `loss` | 15 | 24 | Find the best loss function |
| **3. Augmentation Benchmark** | `augmentation` | 15 | 24 | Find the best data augmentations |
| **4. Final Training** | (Direct invocation) | 40 | 24 | Train the final production model |
| **5. Test Time Augmentation (TTA)** | `tta` | 0 (inference) | 24 | Evaluate prediction calibration and boost metrics |
| **6. 5-Fold Cross Validation** | `cv` | 30 / fold | 24 | Compute final scientific CV numbers |

---

## 🛠️ Step-by-Step Command Playbook

### Stage 1: Architecture Benchmark Sweep
Compares baseline `densenet121`, `resnet10`, and `resnet18`. We pass `--skip-preprocess` to load directly from your preprocessed `.pt` cache.
```bash
!python scripts/kaggle_master.py architecture \
    --epochs 15 \
    --batch-size 24 \
    --device cuda \
    --lr 2e-4 \
    --seed 42 \
    --preprocessed-dir /kaggle/input/datasets/swayamruparel/abide-cache \
    --skip-preprocess
```
* **Inspect Results:** Check `/kaggle/working/experiments/architecture/architecture_comparison.csv` and comparative plots. Select the winner backbone based on ROC-AUC, Macro F1, Sensitivity, and Specificity (e.g., `resnet18`).

---

### Stage 2: Loss Function Sweep
Evaluates the winning architecture under CE, Weighted CE, Focal, CE + Label Smoothing, and Focal + Label Smoothing.
```bash
# Replace resnet18 with your winning architecture
!python scripts/kaggle_master.py loss \
    --architecture resnet18 \
    --epochs 15 \
    --batch-size 24 \
    --device cuda \
    --lr 2e-4 \
    --seed 42 \
    --preprocessed-dir /kaggle/input/datasets/swayamruparel/abide-cache
```
* **Inspect Results:** Check `/kaggle/working/experiments/loss/loss_comparison.csv` and select the winning loss function (e.g., `focal`).

---

### Stage 3: Augmentation Profile Sweep
Evaluates MONAI augmentation profiles (`none`, `minimal`, `moderate`, `strong`, `research`) under your optimized configuration.
```bash
!python scripts/kaggle_master.py augmentation \
    --architecture resnet18 \
    --loss-function focal \
    --epochs 15 \
    --batch-size 24 \
    --device cuda \
    --lr 2e-4 \
    --seed 42 \
    --preprocessed-dir /kaggle/input/datasets/swayamruparel/abide-cache
```
* **Inspect Results:** Check `/kaggle/working/experiments/augmentation/augmentation_comparison.csv` and select the winning profile (e.g., `moderate`).

---

### Stage 4: Final Training Run
Train the final production model using the selected optimal setup over 40 epochs.
```bash
!python -u -m training.train_autism \
    --architecture resnet18 \
    --loss-function focal \
    --augmentation-profile moderate \
    --augment \
    --epochs 40 \
    --batch-size 24 \
    --device cuda \
    --lr 2e-4 \
    --optimizer adam \
    --weight-decay 1e-4 \
    --scheduler-patience 6 \
    --scheduler-factor 0.5 \
    --early-stopping-patience 8 \
    --seed 42 \
    --resume-from none \
    --preprocessed-dir /kaggle/input/datasets/swayamruparel/abide-cache \
    --skip-preprocess \
    --experiment-dir /kaggle/working/experiments/final_model
```
* **Output:** This generates `/kaggle/working/experiments/final_model/best_model.pt` which represents your primary production model.

---

### Stage 5: Test-Time Augmentation (TTA)
Runs inference only (no training) comparing baseline metrics/latencies vs. TTA predictions.
```bash
!python scripts/kaggle_master.py tta \
    --architecture resnet18 \
    --checkpoint-path /kaggle/working/experiments/final_model/best_model.pt \
    --batch-size 24 \
    --device cuda \
    --preprocessed-dir /kaggle/input/datasets/swayamruparel/abide-cache
```
* **Decision:** Review `tta_comparison.csv`. If TTA improves performance metrics (ROC-AUC / F1-score) without unacceptable latency, enable it in the final CV evaluate stage.

---

### Stage 6: 5-Fold Cross Validation
Trains five independent folds from scratch to generate final metrics, OOF predictions calibration diagrams, and Youden's threshold optimizations.
```bash
# Append --tta to the end of this command if TTA improved metrics in Stage 5
!python scripts/kaggle_master.py cv \
    --architecture resnet18 \
    --loss-function focal \
    --augmentation-profile moderate \
    --epochs 30 \
    --batch-size 24 \
    --device cuda \
    --lr 2e-4 \
    --seed 42 \
    --preprocessed-dir /kaggle/input/datasets/swayamruparel/abide-cache
```
* **Output:** Review the final compiled `cv_summary.csv` and `calibration_curve.png`.

---

## 💡 Key Execution & Resource Tips

* **VRAM Constraints (T4 Dual GPUs):**
  * The scripts automatically wrap the model in `nn.DataParallel` when multiple GPUs are available.
  * Start execution using **`batch-size = 24`** (highly optimized for Dual T4 GPUs).
  * If an Out-Of-Memory (OOM) error occurs, stop the cell and immediately drop down to **`batch-size = 16`**.
* **Preprocessed Cache Sharing:**
  * Since the preprocessed cache directory is pointing directly to your Kaggle datasets mount path `/kaggle/input/datasets/swayamruparel/abide-cache`, the training starts immediately without executing any raw file loading or transform computation.
