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

## Pipeline Development Status

- **Stage 0 (Architecture Foundation):** Completed
- **Stage 1 (Universal Ingestion Engine):** Completed & Verified (NIfTI, DICOM, images)
- **Stage 1.5 (Dataset Audit Engine):** Completed & Verified (Distribution stats, corruptions)
- **Stage 2 (Dataset Management Engine):** Completed & Verified (Indexing, parsing, patient splits for ABIDE, ADNI, BraTS)
- **Stage 3 (Research Preprocessing Engine):** Completed & Verified (Orientation, Normalization, Resampling, Spatial cropping/padding, N4 Bias correction, Skull-stripping, and Inverse reconstruction framework)
- **Stage 4 (Generic Training Framework & 4.5 Smoke Tests):** Completed & Verified (Trainer, Callbacks, Checkpointer, MetricsManager, EarlyStopping, Resumption, and ONNX model export validation)
- **Stage 5 (Autism Model & Disease Modules):** Completed & Verified (3D DenseNet121 model training, preprocessed cache, 5-fold stratification, and generic compiler framework)
- **Stage 6 (Inference Engine):** Future (Standardized Prediction packaging)
- **Stage 7 (Clinical PDF Reporter):** Future (PDF layouts, previews, GradCAM, summaries)
- **Stage 8 (Dashboard UI & Deployment):** Future (Streamlit, FastAPI Backend)
