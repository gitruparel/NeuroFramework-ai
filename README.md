# AI-Powered Structural MRI Analysis Platform

A modular, production-grade research repository for processing, analyzing, training, and deploying deep learning models on structural MRI scans.

## Repository Layout

- `configs/`: YAML configurations for dataset, model, training, and deployment.
- `core/`: Config loaders, custom exceptions, abstract interfaces, and loggers.
- `engine/`: Processing pipeline core (readers, validators, converters, caching).
- `preprocessing/`: Modular preprocessing steps (bias field correction, registration, skull stripping, normalization).
- `data/`: Local storage structure for raw, interim, and processed scans.
- `datasets/`: Pytorch & MONAI compatible datasets.
- `models/`: PyTorch networks (e.g. DenseNet, MedicalNet, nnU-Net).
- `training/`: Standardized Trainer, loss functions, metrics, callbacks, and optimizer schedules.
- `explainability/`: GradCAM, Integrated Gradients, and other attribution algorithms.
- `reports/`: PDF report generation modules and templates.
- `deployment/`: Backend APIs (FastAPI) and Frontend client interfaces.
- `schemas/`: Pydantic data schemas/types for rigid structural and prediction validation.
- `utils/`: Common helpers, metrics, visualization, and seeding.
- `experiments/`: Tracking metrics, weights, and configurations per run.
- `logs/`: Isolated training, preprocessing, API, and error logfiles.
- `docs/`: Extensive design, api, and pipeline documentation.

## Setup & installation

1. Install Python 3.11+
2. Install virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```
3. Initialize pre-commit:
   ```bash
   pre-commit install
   ```

## Local AMD GPU Training (DirectML)

For Windows systems with AMD Radeon or Intel GPUs, you can train models with hardware acceleration using the DirectML backend.

### Setup Steps
1. Activate your virtual environment.
2. Install the DirectML PyTorch extension:
   ```bash
   pip install -r requirements-directml.txt
   ```
3. Run the device check script to verify status and capabilities:
   ```bash
   python -m utils.device
   ```

### Execution Example
Start your training run by specifying `directml` as the target device:
```bash
python -m training.train_autism --device directml --skip-preprocess
```

> [!NOTE]
> CUDA remains the recommended backend for NVIDIA GPUs, while DirectML provides hardware acceleration on Windows for AMD and Intel GPUs. Checkpoint files (`latest_model.pt`) are compatible across backends when using matching PyTorch versions, allowing you to train locally on DirectML and resume on CUDA (e.g. Google Colab) seamlessly.
