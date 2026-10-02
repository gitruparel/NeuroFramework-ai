# 🧠 3D Multi-Planar Structural MRI Deep Learning Framework for Autism Spectrum Disorder (ASD) Classification

A state-of-the-art 3D Volumetric Deep Learning system for classifying Autism Spectrum Disorder (ASD) from T1-weighted Structural Magnetic Resonance Imaging (sMRI) scans on the **ABIDE-I Dataset**.

Replicating and adapting the methodology of **Hammash & Younis (2026)**, this repository implements genuine **3D Convolutions (`Conv3D`)** across three orthogonal anatomical views (Axial, Coronal, Sagittal) with 3D CBAM Attention and Adaptive Focal Loss.

---

## 📊 Preserved Core Models & Preprocessing Pipelines

| Model Script | Preprocessing Script | Target Cohort | Input Resolution | Output Location | Peak Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[`models/abide_3d_hierarchical_cnn_pytorch.py`](models/abide_3d_hierarchical_cnn_pytorch.py)** | **[`preprocessing/abide_3d_preprocessing_224_multisite.py`](preprocessing/abide_3d_preprocessing_224_multisite.py)** | **Multi-Site (3 Universities: NYU + UM_1 + USM)** <br> $N=395$ Subjects | 50 Slices @ **`224x224` Full HD** | `./processed_paper_3D_224/` | **`75.95%` (Fold 1 Peak)** 🔥 |
| **[`models/abide_3d_hierarchical_cnn_128_nyu.py`](models/abide_3d_hierarchical_cnn_128_nyu.py)** | **[`preprocessing/abide_3d_preprocessing_128_nyu.py`](preprocessing/abide_3d_preprocessing_128_nyu.py)** | **Single-Site (NYU Alone)** <br> $N=184$ Subjects | 50 Slices @ **`128x128` Standard** | `./processed_paper_3D/` | **`75.00%` Peak** 🌟 |

---

## 🚀 How to Run

### Pipeline A: Multi-Site PyTorch 3D Model (Peak 75.95%)
```bash
python preprocessing/abide_3d_preprocessing_224_multisite.py
python models/abide_3d_hierarchical_cnn_pytorch.py
```

### Pipeline B: Single-Site NYU 128x128 3D Model
```bash
python preprocessing/abide_3d_preprocessing_128_nyu.py
python models/abide_3d_hierarchical_cnn_128_nyu.py
```

### Pipeline C: Interactive Research Workstation (Web UI)
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
Navigate to `http://127.0.0.1:8000` to access the interactive 3D-sMRI Workstation:
- Direct NIfTI (`.nii`, `.nii.gz`) upload and one-click cohort demo evaluation
- Real unstripped anatomical baseline inspection
- Interactive tri-planar orthogonal viewports (Axial, Coronal, Sagittal) with slice scrubbing
- Spatial CBAM attention overlays & multi-stream L2 activation analysis
- Single-page medical-grade PDF clinical informatics dossier export

---

## 📂 Cleaned Repository Structure

```text
MRI-Analysis/
├── data/
│   └── ABIDE_Phenotypic.csv                      # Official ABIDE-I phenotypic metadata table
├── preprocessing/
│   ├── abide_3d_preprocessing_224_multisite.py   # 224x224 Full HD 3-university pipeline (NYU+UM_1+USM)
│   └── abide_3d_preprocessing_128_nyu.py         # 128x128 single-site NYU pipeline
├── models/
│   ├── abide_3d_hierarchical_cnn_pytorch.py      # Multi-site 224x224 PyTorch 3D Model (Peak: 75.95%)
│   └── abide_3d_hierarchical_cnn_128_nyu.py      # Single-site NYU 128x128 3D Model
├── report/
│   └── 3D_MultiPlanar_ASD_Research_Report.md     # Complete research report & methodology
└── README.md                                     # Repository documentation
```

---

## 📄 Academic Reference

* **Hammash, N. M., & Younis, M. C. (2026).** *A Hierarchical Multi-View Deep Learning Framework for Autism Classification Using Structural and Functional MRI.* MDPI Journal of Imaging, 12(3), 109. [DOI: 10.3390/jimaging12030109]
