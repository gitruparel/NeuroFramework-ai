# AI-Powered Structural MRI Analysis Platform: Source of Truth (context.md)

This document is the master configuration, architecture blueprint, and source of truth for the project. Any AI agent modifying this repository must read and update this document to reflect new design decisions and code changes.

---

## Core Philosophy

1. **Framework, Not Single-Model App:** 
   We are building a highly modular medical imaging analysis framework. Individual diseases (Autism, Alzheimer's, Brain Tumors, etc.) are treated as pluggable models. No disease-specific logic should exist inside the core engine.
2. **Unified Data Structures:** 
   All imaging data formats (NIfTI, DICOM, images) must be ingested, checked, and normalized through the Universal MRI Processing Engine into a standardized `MRIData` Pydantic object.
3. **No Jupyter Notebooks in Core:** 
   All components must be written as structured, modular, and testable Python files.
4. **Codebase is Product:** 
   The local repository is the source of truth. GPU training is performed externally (Colab, RunPod, Kaggle) using git-cloned checkpoints; trained weights are brought back to `models/weights/` as static assets.
5. **Token Conservation Rule (CRITICAL FOR AI AGENTS):** 
   Agents must conserve tokens during reasoning and output generation. Keep responses concise, avoid unnecessary pleasantries, and focus on delivering high-quality code edits and exact architectural updates.

---

## Final Repository Structure

```text
brain-ai/
├── README.md               # Quickstart & setup
├── context.md              # THIS FILE (AI Source of Truth)
├── pyproject.toml          # Package metadata & dependencies
├── pytest.ini              # Pytest configs (pythonpath = ["."])
├── configs/                # YAML configs for dataset, model, training, paths
├── core/                   # Loggers, custom exceptions, abstract interfaces
├── engine/                 # Ingestion engine (readers, validation, metadata, previews, QA)
│   └── readers/            # DicomReader, NiftiReader, ImageReader
├── preprocessing/          # MRI normalization, skull-stripping, bias correction pipelines
├── datasets/               # Custom PyTorch and MONAI dataset wrappers
├── models/                 # Model backbones (DenseNet3D, MedicalNet3D, nnUNet3D)
├── explainability/         # Attribution algorithms (GradCAM, SmoothGrad)
├── reports/                # PDF report generators
├── deployment/             # Backend API (FastAPI) and Frontend Dashboard (Streamlit)
├── schemas/                # Pydantic schemas (MRIData, AuditReport, etc.)
├── utils/                  # Metrics, visualization, seeds, helpers
├── tests/                  # Pytest unit testing suite
├── experiments/            # Tracked hyperparameters, metrics, and models
├── data/                   # Data folders (raw, interim, processed, cache, sample)
└── logs/                   # Preprocessing, API, and training logs
```

---

## Platform Core Components

### 1. Universal MRI Processing Engine (UMPE)
- **Entrance point:** `MRIEngine.load(path)`
- **Workflow:**
  1. `MRICache` checks if the path's hash (based on size, mtime, and first 1MB) exists.
  2. `FormatDetector` inspects file signatures (magic bytes) first, falling back to file extensions, to detect format (`nifti`, `dicom`, or `image`).
  3. `ReaderFactory` instantiates the corresponding `BaseReader` implementation:
     - `NiftiReader`: Reads `.nii`/`.nii.gz` volumes using `nibabel`.
     - `DicomReader`: Reads `.dcm` files/directories using `SimpleITK`.
     - `ImageReader`: Reads conventional formats (JPG, PNG) using `opencv-python`.
  4. `MetadataExtractor` constructs normalized `ImageMetadata`, `ScannerMetadata`, and `PatientMetadata` from file headers.
  5. `ScanStatistics` computes basic intensity metrics (min, max, mean, std) from the voxel tensor.
  6. `ValidationEngine` executes double-layered checks:
     - *File validation:* verifies existence, read permissions, and checks for corruption.
     - *MRI validation:* verifies spacing, orientations, dimensions, non-flat intensity variance, and flags blank slices.
  7. `PreviewGenerator` extracts orthogonal slices (Axial, Coronal, Sagittal) and maps values to grayscale `[0-255]` for visual feedback.
  8. `QualityAnalyzer` runs QA heuristics measuring signal-to-noise ratio (SNR), blur index via Laplacian variance, motion/ringing artifacts, and missing slices.
  9. `MRIData` packages all raw metrics, validation outputs, statistics, and previews.

### 2. Dataset Audit Engine
- **Class:** `DatasetAuditor`
- **Purpose:** Traverses local directories (e.g. ABIDE, ADNI, BraTS) to analyze shapes, corruptions, class distributions, and metadata completeness prior to executing preprocessors or training.

### 3. Preprocessing Framework (Next Stage)
- Designed to run sequence-based preprocessors sequentially. 
- *Transforms include:* N4 Bias Correction, Registration, Skull Stripping, Resampling, Cropping, and Normalization.

### 4. Training Engine
- Completely disease-agnostic training loops configured via abstract interfaces (`BaseTrainer`, `BaseDataset`, `BaseCallback`).

---

## Key Design Decisions & Bug Fixes History

### 1. Nested Schema Instantiation (test_core.py)
- **Problem:** `test_mri_metadata_schema` was instantiating `MRIMetadata` using flat properties like `patient_id` or `voxel_dims`.
- **Decision:** Updated tests to instantiate correct nested models (`ImageMetadata` and `PatientMetadata`) and access properties nestedly.

### 2. Pipeline Step History Decorator (decorators.py)
- **Problem:** The decorator `@log_pipeline_step` appended log strings to `args[0].history`. Since methods are class instance methods, `args[0]` is the class instance (`self`), which has no `history` list attribute.
- **Decision:** Updated the decorator to look up the `history` parameter inside `kwargs` or search positional `args[1:]` for a list object. Fallback to `args[0].history` is preserved.

### 3. Dataset Auditor Subject Grouping (audit.py)
- **Problem:** `_identify_subjects` was grouping files using `parts[0]` relative to the dataset root. For folders styled as `root/Class/Subject/scan.nii.gz`, the class folder (`AD`, `CN`, etc.) was incorrectly evaluated as the subject.
- **Decision:** If the first part matches standard pathology class abbreviations, the auditor skips to `parts[1]` for subject identification.

### 4. PyTorch Dataset Loader (datasets/base.py)
- **Problem:** `MRIDataset.__getitem__` attempted to access `item.file_path`, causing an `AttributeError`.
- **Decision:** Changed access pattern to `item.raw.source_path` matching `MRIData` schemas.

### 5. Windows SimpleITK DICOM Compatibilities (test_engine.py)
- **Problem:** SimpleITK fails to write 3D `int16` volumes directly to a single DICOM file on Windows.
- **Decision:** Updated mock DICOM generation inside tests to use `np.uint16` which writes successfully.
- **Problem:** `FormatDetector` magic bytes check fails on gzipped `.nii.gz` because NIfTI header bytes are compressed.
- **Decision:** Mock test setup saves an uncompressed `brain.nii` file in addition to the compressed file, which is used for magic bytes verification. Added `pytest.approx` for floating-point voxel spacing list checks.

### 6. NiftiReader Extension Fallback (nifti.py)
- **Problem:** NiBabel's `load` method determines format solely from filename extensions. If a file is signature-detected as NIfTI but renamed to a non-standard extension (e.g. `.jpg`), `nib.load` raises an error.
- **Decision:** Added a fallback mechanism inside `NiftiReader.read` using `nib.FileHolder` and `.from_file_map`, checking for Gzip magic bytes dynamically to decompress standard gzip-wrapped streams.

## Key Design Decisions & Bug Fixes History (Continued)

### 7. Custom Collation Fallback
- **Problem:** When batching `DatasetSample` structures, PyTorch's default collator threw exceptions on non-tensor/nullable fields.
- **Decision:** Implemented `collate_dataset_samples` to stack only tensor fields, package metadata dictionaries, and preserve list strings, with a dictionary fallback for compatibility.

### 8. Trainer Scheduler Ordering
- **Problem:** For learning rate schedulers like `ReduceLROnPlateau`, the scheduler step must be computed after evaluating validation loss, not before.
- **Decision:** Ordered validation metric computation first before executing the scheduler step in the `Trainer.fit` loop.

### 9. Dataset Compiler Framework (Stage 5B)
- **Problem:** Need a clean, reusable dataset parsing layer to convert BIDS-layout raw files and phenotypic CSV records into splits and index files without duplicating dataset management.
- **Decision:** Implemented `DatasetCompiler` as an abstract class, creating `ABIDECompiler` to parse BIDS structures and phenotypic variables (clearing null placeholding integers, mapping diagnosis, extracting demographic metadata), generating stratified splits and stratified 5-fold cross-validation configs, while leaving placeholders for `ADNICompiler` and `BraTSCompiler`.

### 10. Training Performance & Preprocessing Telemetry Optimizations
- **Problem:** Need to maximize training throughput, reduce CPU-GPU data copy latency, support fast local SSD caching/output backups, and identify preprocessing performance bottlenecks.
- **Decision:**
  - **AMP & cudnn.benchmark:** Enabled PyTorch Automatic Mixed Precision (AMP) dynamically on CUDA devices to accelerate 3D model convolutions, and activated `torch.backends.cudnn.benchmark = True` for optimized fixed-shape input pipelines.
  - **Multi-worker DataLoader:** Parameterized trainer DataLoaders with `num_workers=2`, `pin_memory=True`, and `persistent_workers=True` on CUDA.
  - **SSD Local Caching Support:** Parameterized the pipeline with a custom `--raw-dir` path resolver (`resolve_raw_path`) and training output copy helper (`--copy-outputs-to`) to support local SSD staging and training loop execution decoupling on cloud environments (like Google Colab).
  - **Transform Telemetry Profiling:** Collected execution duration of each pipeline step and logged progress breakdowns (e.g. `Reorient`, `BiasFieldCorrector`) to monitor CPU bottlenecks.

### 11. Spatial Resize & Cache Validation
- **Problem:** Dynamic size variation across subject MRIs causes shape mismatch collation failures during batch training. Caches need to be cleared and segmented to avoid stale configuration incompatibilities.
- **Decision:**
  - **Resize Transform:** Implemented a new `Resize` transform utilizing SimpleITK's linear/nearest/bspline resampling to map voxel sizes to uniform coordinate dimensions. Added exact reverse resampling reconstruction mapping inside the inversion engine.
  - **Cache Versioning (`cache_version: "v2"`):** Introduced config-driven routing (writing and reading from a subdirectory named after the cache version) to prevent mixing incompatible cached tensors.
  - **Dynamic Shape Validator:** Enforces dynamic shape assertions (e.g., checking that the preprocessed volume matches the final spatial transform target size `(1, 128, 128, 128)`) before saving `.pt` caches.
  - **Cache Clearing:** Implemented a `--clear-cache` CLI flag in the preprocessor executable script to safely wipe active version folders.

### 12. Stage 6 Clinical Robustness & Regularization Tuning
- **Problem:** Overfitting regularization is needed on the deep 3D DenseNet with a limited dataset, raw validation outputs must be backed up to conserve compute, and multiple experimental runs need to be tracked and compared automatically.
- **Decision:**
  - **Dropout and Label Smoothing:** Exposed `--dropout-prob` (passed to DenseNet blocks) and `--label-smoothing` (passed to CrossEntropyLoss) as parameters default to off (0.0) for controlled experiments.
  - **Smart Class Weighting:** Logs training Control/ASD distribution and exact imbalance ratio on startup. Applies inverse frequency weights only when `--use-class-weights` is requested.
  - **Diagnostic Output & Numpy Backups:** Saves validation raw logits (`val_logits.npy`) and probabilities (`val_probabilities.npy`) next to `predictions.csv`, `roc_points.csv`, and `pr_points.csv` for post-inference analysis and publication plotting.
  - **Threshold Analysis:** Searches validation probabilities for boundaries maximizing Youden's J, F1, and Balanced Accuracy, logging them to `experiment_meta.json`.
  - **Central Comparison Logging:** Automates adding or updating rows in a parent `comparison.csv` file, providing an aggregated matrix of metrics across all experiments.

### 13. Stage 6.5A Automated Hyperparameter Optimization (Optuna)
- **Problem:** Manual tuning of learning rate, batch size, weight decay, dropout, and scheduler patient settings is slow and suboptimal. We need a modular, reusable hyperparameter search framework to maximize model generalization.
- **Decision:**
  - **Decoupled Engine (`training/hyperopt.py`):** Implemented an architecture-agnostic Optuna runner with seed-fixed TPESampler reproducibility.
  - **Diagnostic Sweep Reports:** Logs complete studies to `optuna_trials.csv` and optimal configurations to `optuna_best.json`. Renders optimization history and parameter importance plots with basic matplotlib visual fallbacks.
  - **CLI Search & Sandbox Isolation:** Added `--optuna-trials` CLI flag to `train_autism.py`. Suppresses WandB, plotting, and comparisons during intermediate trials, writing checkpoints to temporary trial-specific subdirectories that are immediately deleted upon completion to conserve gigabytes of disk space. Automatically kicks off a final, fully package-compiled baseline run using the best hyperparameters found.

### 14. Stage 6.5B Research-Grade MRI Augmentation Framework (MONAI)
- **Problem:** Basic spatial translations/flips do not expose model training to realistic intensity deviations and scanning noise. We need a modular, seed-reproducible augmentation suite that preserves clinical anatomy while improving generalization.
- **Decision:**
  - **Configurable MONAI Pipeline (`training/augmentations.py`):** Implemented predefined augmentation profiles (`minimal`, `moderate` (default), `strong`, and `research` with elastic deformations) leveraging advanced 3D spatial and intensity transforms (Gaussian noise/smoothing, contrast, scaling, and affine shifts).
  - **Determinism Seeding:** Coupled MONAI's random state setting (`monai.utils.set_determinism`) with global reproducibility seeds.
  - **Slice Validation Previews:** Exports `augmentation_preview.png` comparing the middle axial slice of the original scan against 5 random augmented outputs for quick visual validation.
  - **CLI Profiling & Logging:** Added `--augmentation-profile` flag. Logs active profiles, enabled transforms lists, and exact transform probabilities inside `experiment_meta.json`.

---

## Pipeline Development Status

- **Stage 0 (Architecture Foundation):** Completed
- **Stage 1 (Universal Ingestion Engine):** Completed & Verified (NIfTI, DICOM, images)
- **Stage 1.5 (Dataset Audit Engine):** Completed & Verified (Distribution stats, corruptions)
- **Stage 2 (Dataset Management Engine):** Completed & Verified (Indexing, parsing, patient splits for ABIDE, ADNI, BraTS)
- **Stage 3 (Research Preprocessing Engine):** Completed & Verified (Orientation, Normalization, Resampling, Spatial cropping/padding, N4 Bias correction, Skull-stripping, and Inverse reconstruction framework)
- **Stage 4 (Generic Training Framework & 4.5 Smoke Tests):** Completed & Verified (Trainer, Callbacks, Checkpointer, MetricsManager, EarlyStopping, Resumption, and ONNX model export validation)
- **Stage 5 (Autism Model & Disease Modules):** Completed & Verified (3D DenseNet121 model training, preprocessed cache, 5-fold stratification, and generic compiler framework)
- **Stage 6 (Clinical Robustness & Advanced Regularization):** Completed & Verified (Dropout, Label Smoothing, Smart Class Weighting, threshold analysis, validation backups, and central comparison logging)
- **Stage 6.5A (Automated Hyperparameter Optimization):** Completed & Verified (Optuna integration, search spaces, trial plotting, and validation tests)
- **Stage 6.5B (Research-Grade MRI Augmentation Framework):** Completed & Verified (Configurable profiles, slice visualizations, deterministic seeding, and unit tests)
- **Stage 6.5C-F (Future Sub-Stages):** Future (Architecture modifications, Focal Loss, TTA, explainability, etc.)
- **Stage 7 (Optimal Thresholds, Calibration, & 5-Fold Evaluation):** Future (5-Fold cross-validation, ECE calibration plots, threshold calibrations)
- **Stage 8 (Attribution & Explainability):** Future (Grad-CAM heatmaps, localized visualization)
- **Stage 9 (Clinical Dashboard & Deployment):** Future (FastAPI, Streamlit, local offline deployment setup)

---

## Proposed Experiment Training Progression (After Stage 6.5F)

1. **Experiment 1:** DenseNet | No Augmentations | Baseline
2. **Experiment 2:** DenseNet | Moderate Augmentations
3. **Experiment 3:** DenseNet | Best Optuna Hyperparameters
4. **Experiment 4:** DenseNet | Best Optuna Hyperparameters + Best Augmentation Profile
5. **Experiment 5:** ResNet10 | Same Settings as Experiment 4 (Architecture Comparison)
6. **Experiment 6:** ResNet18 | Same Settings as Experiment 4 (Architecture Comparison)
7. **Experiment 7:** Winner Architecture | Focal Loss Regularization
8. **Experiment 8:** Winner Architecture | Test-Time Augmentation (TTA)
9. *Proceed to Stage 7 (Calibration and 5-Fold Evaluation).*
