# 🧠 3D Multi-Planar Structural MRI Deep Learning Framework for ASD Classification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![ABIDE-I Benchmark](https://img.shields.io/badge/ABIDE--I-Multi--Site-orange.svg)](http://fcon_1000.projects.nitrc.org/indi/abide/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end 3D Volumetric Deep Learning research platform for classifying **Autism Spectrum Disorder (ASD)** from T1-weighted structural Magnetic Resonance Imaging (sMRI) scans on the **ABIDE-I Dataset**.

Replicating and adapting the methodology of **Hammash & Younis (2026)**, this repository implements genuine **3D Convolutions (`Conv3D`)** across three orthogonal anatomical views (Axial, Coronal, Sagittal) with 3D CBAM (Convolutional Block Attention Module) and Adaptive Focal Loss.

---

## 🌟 Key Highlights

- **Multi-Site Generalization (Peak Fold Accuracy: 75.95% 🔥):** Trained across three distinct academic medical centers (NYU Langone, University of Michigan, and University of Utah; $N=395$ subjects) at Full HD $224 \times 224 \times 50$ volumetric resolution.
- **Tri-Planar 3D ConvNet Architecture:** Three independent volumetric streams process Axial, Coronal, and Sagittal anatomical views in parallel, fused via dense multi-layer representations.
- **Interpretability & Biomarker Localization:** 3D CBAM spatial attention weights reveal discriminative anatomical regions (e.g., corpus callosum, fronto-temporal cortices, ventricles).
- **Interactive Research Workstation (Web UI):** Browser-based clinical workstation with tri-planar 60fps slice scrubbers, dynamic CBAM heatmap toggles, direct `.nii`/`.nii.gz` upload, and single-click clinical PDF dossier export.

---

## 💻 How to Run Locally on Any Device

This project is fully cross-platform and can be run on **Windows**, **macOS** (Intel & Apple Silicon M-series), and **Linux**.

### System Prerequisites
- **Python:** Version `3.10`, `3.11`, `3.12`, or `3.13`
- **Git:** Installed on your system
- **RAM:** Minimum 8 GB (16 GB recommended for deep volumetric inference)
- **Hardware Acceleration (Optional):** NVIDIA GPU with CUDA, Apple Silicon GPU (MPS), or standard multi-core CPU.

---

### Step 1: Clone the Repository

Open your terminal or command prompt and clone the repository:

```bash
git clone https://github.com/gitruparel/NeuroFramework-ai.git
cd NeuroFramework-ai
```

---

### Step 2: Create & Activate a Virtual Environment

It is strongly recommended to use an isolated Python virtual environment.

#### 🪟 Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> *Tip for Windows:* If you receive an execution policy restriction error, run:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

#### 🪟 Windows (Command Prompt - CMD)
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### 🍎 macOS & 🐧 Linux (Bash / Zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install Required Dependencies

Install the core dependencies using `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### (Optional) Hardware Acceleration Configuration

- **NVIDIA GPU (CUDA) on Windows / Linux:**
  To enable high-speed GPU acceleration for PyTorch:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- **Apple Silicon (M1 / M2 / M3 / M4) on macOS:**
  Standard PyTorch (`pip install torch`) automatically detects and utilizes Apple Metal Performance Shaders (`mps`). No extra configuration needed.
- **CPU-Only Devices:**
  Standard PyTorch will seamlessly run inference on your CPU.

---

### Step 4: Launch the Interactive Web Workstation

Launch the local FastAPI server:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
*Or alternatively:*
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Once started, open your web browser and navigate to:
👉 **`http://127.0.0.1:8000`**

#### Using the Workstation:
1. **Load Preloaded Benchmark:** Click **"Select Preloaded Benchmark Scan"** in the top navigation bar or select any sample (e.g., *NYU Subject 0050952 [ASD Peak: 75.95%]* or *UM_1 Subject 0050327 [Control: 24.2%]*).
2. **Upload Custom NIfTI:** Drag and drop any raw `.nii` or `.nii.gz` anatomical brain scan into the upload zone.
3. **Interactive Inspection:**
   - Use the **Slice Scrubbers** to slice through Axial, Coronal, and Sagittal planes at 60 fps.
   - Toggle **CBAM Attention Heatmap** to reveal regional spatial importance.
   - Inspect the **Tri-Planar Synchronized Crosshair** and volumetric stream activations.
   - Click **"Generate Clinical Dossier"** to export a clean, printable medical research PDF.

---

## 📊 Benchmark Model Performance

| Pipeline | Target Cohort | Scanners | Resolution | Attention Mechanism | Peak Fold Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`models/abide_3d_hierarchical_cnn_pytorch.py`** | **Multi-Site (NYU + UM_1 + USM)** <br> $N=395$ | Siemens Allegra, Trio & GE Signa | $50 \times 224 \times 224$ (Full HD) | **3D CBAM** (Dual Channel + Spatial) | **75.95%** 🔥 |
| **`models/abide_3d_hierarchical_cnn_128_nyu.py`** | **Single-Site (NYU Alone)** <br> $N=184$ | Siemens Allegra 3T | $50 \times 128 \times 128$ (Standard) | Spatial Squeeze-Excitation | **75.00%** |

---

## 📂 Repository Directory Structure

```text
NeuroFramework-ai/
├── app.py                                        # FastAPI Web Workstation Backend & API
├── requirements.txt                              # Python environment dependency manifest
├── README.md                                     # System documentation & reproduction manual
├── context.md                                    # Architectural blueprint & engineering specification
├── .gitignore                                    # Git exclusion rules
│
├── data/
│   └── ABIDE_Phenotypic.csv                      # Official ABIDE-I phenotypic metadata table
│
├── inference/
│   └── engine.py                                 # Multi-planar preprocessor, 3D ConvNet inference, & CBAM attention extractor
│
├── web/
│   ├── index.html                                # Clinical research workstation layout
│   ├── style.css                                 # Clinical high-contrast theme styling
│   └── app.js                                    # Dual-layer canvas renderer, 60fps scrubbers, & PDF exporter
│
├── preprocessing/
│   ├── abide_3d_preprocessing_224_multisite.py   # Multi-site AWS S3 fetcher + Otsu + N4 bias + 224x224 tensor pipeline
│   └── abide_3d_preprocessing_128_nyu.py         # Single-site NYU 128x128 tensor pipeline
│
├── models/
│   ├── abide_3d_hierarchical_cnn_pytorch.py      # SOTA Multi-Site 3-Stream 3D ConvNet + CBAM PyTorch model
│   └── abide_3d_hierarchical_cnn_128_nyu.py      # Single-site NYU 128x128 model
│
└── report/
    └── 3D_MultiPlanar_ASD_Research_Report.md     # Full academic research report & methodology validation
```

---

## 📄 Academic Reference & Citation

If you use or adapt this framework in your research, please cite the foundational publication:

```bibtex
@article{hammash2026hierarchical,
  title     = {A Hierarchical Multi-View Deep Learning Framework for Autism Classification Using Structural and Functional MRI},
  author    = {Hammash, N. M. and Younis, M. C.},
  journal   = {MDPI Journal of Imaging},
  volume    = {12},
  number    = {3},
  pages     = {109},
  year      = {2026},
  doi       = {10.3390/jimaging12030109}
}
```
