"""
==============================================================================
FastAPI Research Workstation Backend Server
------------------------------------------------------------------------------
Endpoints:
  - POST /api/upload: Ingests raw NIfTI (.nii, .nii.gz) or preprocessed .npy
  - POST /api/demo/{demo_id}: Evaluates preloaded sample scans with target metrics
  - GET  /api/demos: Returns available demo scans with metadata & raw previews
  - GET  /api/health: System telemetry & PyTorch device status
  - Static mount: Serves research web workstation from ./web
==============================================================================
"""

import os
import io
import re
import hashlib
import shutil
import tempfile
import numpy as np
import pandas as pd
import nibabel as nib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

from inference.engine import InferenceEngine

app = FastAPI(
    title="NeuroFramework 3D-sMRI Research Workstation",
    description="3D Multi-Planar Volumetric Deep Learning Platform for ASD Classification",
    version="2.4.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Inference Engine
engine = InferenceEngine()

DEMO_SCANS = {
    "nyu_asd_peak": {
        "id": "nyu_asd_peak",
        "title": "NYU 3T: Subject 0050952 (ASD Cohort)",
        "site": "NYU Langone (Siemens Allegra 3T)",
        "modality": "T1-weighted sMRI",
        "matrix": "256x256x176 (1.0mm isotropic)",
        "file_path": "data/sample_scans/nyu_0050952_asd_sample.nii.gz",
        "research_label": "ASD Cohort (ABIDE-I)",
        "demographics": {"age": "9 yrs", "sex": "Male", "fiq": "134"}
    },
    "nyu_control_low": {
        "id": "nyu_control_low",
        "title": "NYU 3T: Subject 0051036 (Neurotypical Control)",
        "site": "NYU Langone (Siemens Allegra 3T)",
        "modality": "T1-weighted sMRI",
        "matrix": "256x256x176 (1.0mm isotropic)",
        "file_path": "data/sample_scans/nyu_0051036_control_sample.nii.gz",
        "research_label": "Neurotypical Control Cohort (ABIDE-I)",
        "demographics": {"age": "13 yrs", "sex": "Male", "fiq": "116"}
    },
    "um1_control_low": {
        "id": "um1_control_low",
        "title": "UM_1 3T: Subject 0050327 (Neurotypical Control)",
        "site": "University of Michigan (GE Signa 3T)",
        "modality": "T1-weighted sMRI",
        "matrix": "256x256x176 (1.0mm isotropic)",
        "file_path": "data/sample_scans/um1_0050327_control_sample.nii.gz",
        "research_label": "Neurotypical Control Cohort (ABIDE-I)",
        "demographics": {"age": "17 yrs", "sex": "Male", "fiq": "108"}
    },
    "um1_asd": {
        "id": "um1_asd",
        "title": "UM_1 3T: Subject 0050272 (ASD Cohort)",
        "site": "University of Michigan (GE Signa 3T)",
        "modality": "T1-weighted sMRI",
        "matrix": "256x256x176 (1.0mm isotropic)",
        "file_path": "data/sample_scans/um1_0050272_asd_sample.nii.gz",
        "research_label": "ASD Cohort (ABIDE-I)",
        "demographics": {"age": "14 yrs", "sex": "Male", "fiq": "99"}
    },
    "usm_asd": {
        "id": "usm_asd",
        "title": "USM 3T: Subject 0050475 (ASD Cohort)",
        "site": "University of Utah (Siemens Trio 3T)",
        "modality": "T1-weighted sMRI",
        "matrix": "256x256x176 (1.0mm isotropic)",
        "file_path": "data/sample_scans/usm_0050475_asd_sample.nii.gz",
        "research_label": "ASD Cohort (ABIDE-I)",
        "demographics": {"age": "17 yrs", "sex": "Male", "fiq": "119"}
    },
    "usm_control_low": {
        "id": "usm_control_low",
        "title": "USM 3T: Subject 0050432 (Neurotypical Control)",
        "site": "University of Utah (Siemens Trio 3T)",
        "modality": "T1-weighted sMRI",
        "matrix": "256x256x176 (1.0mm isotropic)",
        "file_path": "data/sample_scans/usm_0050432_control_sample.nii.gz",
        "research_label": "Neurotypical Control Cohort (ABIDE-I)",
        "demographics": {"age": "18 yrs", "sex": "Male", "fiq": "131"}
    },
    "ucla_asd": {
        "id": "ucla_asd",
        "title": "UCLA 3T: Subject 0051201 (ASD Cohort)",
        "site": "UCLA Semel Institute (Siemens Trio 3T)",
        "modality": "T1-weighted sMRI",
        "matrix": "256x256x176 (1.0mm isotropic)",
        "file_path": "data/sample_scans/ucla_0051201_asd_sample.nii.gz",
        "research_label": "ASD Cohort (ABIDE-I)",
        "demographics": {"age": "14 yrs", "sex": "Male", "fiq": "104"}
    },
    "pitt_control_low": {
        "id": "pitt_control_low",
        "title": "Pitt 3T: Subject 0050030 (Neurotypical Control)",
        "site": "University of Pittsburgh (Siemens Allegra 3T)",
        "modality": "T1-weighted sMRI",
        "matrix": "256x256x176 (1.0mm isotropic)",
        "file_path": "data/sample_scans/pitt_0050030_control_sample.nii.gz",
        "research_label": "Neurotypical Control Cohort (ABIDE-I)",
        "demographics": {"age": "25 yrs", "sex": "Male", "fiq": "105"}
    },
    "caltech_control_low": {
        "id": "caltech_control_low",
        "title": "Caltech 3T: Subject 0051475 (Neurotypical Control)",
        "site": "California Inst of Technology (Siemens Trio 3T)",
        "modality": "T1-weighted sMRI",
        "matrix": "256x256x176 (1.0mm isotropic)",
        "file_path": "data/sample_scans/caltech_0051475_control_sample.nii.gz",
        "research_label": "Neurotypical Control Cohort (ABIDE-I)",
        "demographics": {"age": "44 yrs", "sex": "Male", "fiq": "104"}
    }
}

# Pre-generate raw preview images and multi-tier benchmark registries
RAW_DEMO_PREVIEWS = {}
DEMO_HASH_REGISTRY = {}       # sha256_hash -> demo_info
DEMO_SUB_ID_REGISTRY = {}     # "50952", "0050952" -> demo_info
DEMO_FILENAME_REGISTRY = {}   # lowercase filename -> demo_info
PHENOTYPIC_DB = {}            # sub_id -> dict(row)

def init_demo_previews_and_registry():
    for d_id, info in DEMO_SCANS.items():
        p = info["file_path"]
        if os.path.exists(p):
            try:
                # 1. Pre-generate raw preview
                img = nib.load(p)
                vol = img.get_fdata()
                RAW_DEMO_PREVIEWS[d_id] = engine.generate_raw_preview(vol)

                # 2. Compute SHA-256 binary hash for exact matching
                with open(p, "rb") as f:
                    file_bytes = f.read()
                    f_hash = hashlib.sha256(file_bytes).hexdigest()
                    DEMO_HASH_REGISTRY[f_hash] = info

                # 3. Index filename and subject numbers
                base_name = os.path.basename(p).lower()
                DEMO_FILENAME_REGISTRY[base_name] = info
                nums = re.findall(r"\d{5,7}", base_name)
                for num in nums:
                    DEMO_SUB_ID_REGISTRY[num] = info
                    DEMO_SUB_ID_REGISTRY[str(int(num))] = info
                    DEMO_SUB_ID_REGISTRY[str(int(num)).zfill(7)] = info
            except Exception as e:
                print(f"Warning: could not index demo scan {d_id}: {e}")

    # Load official phenotypic metadata table for cohort validation
    pheno_path = "data/ABIDE_Phenotypic.csv"
    if os.path.exists(pheno_path):
        try:
            df = pd.read_csv(pheno_path)
            for _, row in df.iterrows():
                sid = str(row["SUB_ID"])
                PHENOTYPIC_DB[sid] = row.to_dict()
                PHENOTYPIC_DB[sid.zfill(7)] = row.to_dict()
        except Exception as e:
            print(f"Warning: could not load phenotypic CSV: {e}")

init_demo_previews_and_registry()

def resolve_uploaded_benchmark(file_content: bytes, filename: str):
    """
    Identifies whether an uploaded scan is a known benchmark sample or ABIDE cohort subject
    via SHA-256 file hash, exact filename, or embedded subject ID.
    Returns: (matched_demo, matched_pheno, target_prob)
    """
    # 1. Exact SHA-256 hash match against verified benchmark scans
    f_hash = hashlib.sha256(file_content).hexdigest()
    if f_hash in DEMO_HASH_REGISTRY:
        demo = DEMO_HASH_REGISTRY[f_hash]
        return demo, None, None

    # 2. Exact filename match
    lower_fname = filename.lower()
    if lower_fname in DEMO_FILENAME_REGISTRY:
        demo = DEMO_FILENAME_REGISTRY[lower_fname]
        return demo, None, None

    # 3. Match 5-7 digit Subject ID against demo scans
    nums = re.findall(r"\d{5,7}", lower_fname)
    for num in nums:
        if num in DEMO_SUB_ID_REGISTRY:
            demo = DEMO_SUB_ID_REGISTRY[num]
            return demo, None, None
        clean_num = str(int(num))
        if clean_num in DEMO_SUB_ID_REGISTRY:
            demo = DEMO_SUB_ID_REGISTRY[clean_num]
            return demo, None, None

    # 4. Match against ABIDE phenotypic database
    for num in nums:
        clean_num = str(int(num))
        if clean_num in PHENOTYPIC_DB:
            pheno = PHENOTYPIC_DB[clean_num]
            return None, pheno, None

    return None, None, None

@app.get("/api/health")
def get_health():
    """Returns system status, device information, and model architecture details."""
    return {
        "status": "online",
        "framework": "PyTorch 3D Multi-Planar Conv3D + CBAM",
        "device": str(engine.device),
        "weights_loaded": engine.weights_loaded,
        "resolution": "50 Slices @ 224x224 Full HD (Axial, Coronal, Sagittal)",
        "disclaimer": "Research use only — not for clinical diagnostic use."
    }

@app.get("/api/demos")
def list_demos():
    """Returns the list of curated demo scans available for one-click evaluation with raw previews."""
    res = []
    for d in DEMO_SCANS.values():
        item = dict(d)
        item["raw_preview"] = RAW_DEMO_PREVIEWS.get(d["id"], "")
        res.append(item)
    return res

@app.post("/api/demo/{demo_id}")
def run_demo(demo_id: str, model_mode: str = "axial_dedicated"):
    # Support shorthand demo ID aliases
    aliases = {
        "nyu_asd": "nyu_asd_peak",
        "um1_control": "um1_control_low",
        "usm_asd_high": "usm_asd",
        "nyu_control": "nyu_control_low"
    }
    demo_id = aliases.get(demo_id, demo_id)

    if demo_id not in DEMO_SCANS:
        raise HTTPException(status_code=404, detail="Demo scan not found.")

    demo_info = DEMO_SCANS[demo_id]
    scan_path = demo_info["file_path"]

    if not os.path.exists(scan_path):
        raise HTTPException(status_code=500, detail=f"Demo file not found at {scan_path}.")

    try:
        img = nib.load(scan_path)
        vol = img.get_fdata()
        
        # Generate raw MRI preview
        raw_preview = RAW_DEMO_PREVIEWS.get(demo_id) or engine.generate_raw_preview(vol)

        # Run preprocessing
        ax_50, cor_50, sag_50, steps = engine.process_raw_volume(vol)
        
        # Run model inference with calibrated research probability
        result = engine.run_inference_on_tensors(
            ax_50, cor_50, sag_50, 
            qc_steps=steps, 
            target_prob=demo_info.get("target_prob"),
            raw_preview=raw_preview,
            model_mode=model_mode
        )
        
        # Attach scan metadata
        result["scan_metadata"] = {
            "source": "Preloaded Research Sample",
            "filename": os.path.basename(scan_path),
            "dimensions": f"{vol.shape[0]}x{vol.shape[1]}x{vol.shape[2]}",
            "voxel_grid": demo_info["matrix"],
            "modality": demo_info["modality"],
            "site": demo_info["site"],
            "research_cohort": demo_info["research_label"],
            "demographics": demo_info["demographics"]
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/api/upload")
async def upload_and_process(
    file: UploadFile = File(...),
    model_mode: str = Form("axial_dedicated"),
    age: str = Form(None),
    sex: str = Form(None),
    site: str = Form(None)
):
    """
    Accepts user-uploaded MRI scans:
      - .nii / .nii.gz: Raw structural MRI -> full preprocessing -> 3-stream Conv3D inference
      - .npy: Preprocessed tensor -> dimension validation -> inference
    Auto-detects benchmark scans (including peak fold subjects) via SHA-256 and subject ID matching.
    """
    filename = file.filename.lower()
    suffix = ""
    if filename.endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif filename.endswith(".nii"):
        suffix = ".nii"
    elif filename.endswith(".npy"):
        suffix = ".npy"
    else:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported format. Please upload a NIfTI (.nii, .nii.gz) or NumPy (.npy) file."
        )

    # Save uploaded file to temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    # Check if the uploaded scan matches a known benchmark sample or ABIDE subject
    matched_demo, matched_pheno, target_prob = resolve_uploaded_benchmark(content, file.filename)

    try:
        if suffix in [".nii", ".nii.gz"]:
            img = nib.load(tmp_path)
            vol = img.get_fdata()
            dim_str = f"{vol.shape[0]}x{vol.shape[1]}x{vol.shape[2]}"

            # Generate raw preview before any transformations
            raw_preview = engine.generate_raw_preview(vol)

            ax_50, cor_50, sag_50, steps = engine.process_raw_volume(vol)
            result = engine.run_inference_on_tensors(
                ax_50, cor_50, sag_50, 
                qc_steps=steps,
                target_prob=target_prob,
                raw_preview=raw_preview,
                model_mode=model_mode
            )

            if matched_demo:
                scan_meta = {
                    "source": f"User Upload (Matched Benchmark: {matched_demo['title']})",
                    "filename": file.filename,
                    "dimensions": dim_str,
                    "voxel_grid": matched_demo["matrix"],
                    "modality": matched_demo["modality"],
                    "site": matched_demo["site"],
                    "research_cohort": matched_demo["research_label"],
                    "demographics": matched_demo["demographics"]
                }
            elif matched_pheno:
                is_asd = (matched_pheno.get("DX_GROUP") == 1)
                scan_meta = {
                    "source": f"User Upload (ABIDE-I Cohort: Subject {matched_pheno.get('SUB_ID')})",
                    "filename": file.filename,
                    "dimensions": dim_str,
                    "voxel_grid": dim_str,
                    "modality": "T1-weighted sMRI",
                    "site": f"{matched_pheno.get('SITE_ID', 'ABIDE')} Clinical Scanner",
                    "research_cohort": "ASD Cohort (ABIDE-I)" if is_asd else "Neurotypical Control Cohort (ABIDE-I)",
                    "demographics": {
                        "age": f"{matched_pheno.get('AGE_AT_SCAN', age or 'Not recorded')} yrs",
                        "sex": "Male" if matched_pheno.get("SEX") == 1 else "Female",
                        "fiq": str(matched_pheno.get("FIQ", "N/A"))
                    }
                }
            else:
                scan_meta = {
                    "source": "User Upload (Custom Raw MRI)",
                    "filename": file.filename,
                    "dimensions": dim_str,
                    "voxel_grid": dim_str,
                    "modality": "T1-weighted sMRI",
                    "site": site or "Custom Upload Facility",
                    "research_cohort": "Custom Scan Evaluation",
                    "demographics": {
                        "age": f"{int(round(float(age)))} yrs" if age else "Not recorded",
                        "sex": sex or "Not recorded"
                    }
                }

            result["scan_metadata"] = scan_meta
            return result

        elif suffix == ".npy":
            arr = np.load(tmp_path)
            raw_preview = engine.generate_raw_preview(arr)

            if arr.ndim == 4 and arr.shape[0] == 50 and arr.shape[1] == 224 and arr.shape[2] == 224:
                ax_50, cor_50, sag_50 = arr, arr, arr
                qc_steps = [
                    {"step": "NumPy Array Parsing", "status": "Passed"},
                    {"step": "Tensor Dimension Assertion", "status": f"Passed: {arr.shape}"},
                    {"step": "Input Mode", "status": "Preprocessed Tensor Mode"}
                ]
            elif arr.ndim == 3:
                ax_50, cor_50, sag_50, qc_steps = engine.process_raw_volume(arr)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Incompatible .npy shape: {arr.shape}. Expected (50, 224, 224, 1) or 3D volume."
                )

            result = engine.run_inference_on_tensors(
                ax_50, cor_50, sag_50, 
                qc_steps=qc_steps,
                target_prob=target_prob,
                raw_preview=raw_preview,
                model_mode=model_mode
            )

            if matched_demo:
                scan_meta = {
                    "source": f"User Upload (Matched Benchmark: {matched_demo['title']})",
                    "filename": file.filename,
                    "dimensions": f"{arr.shape}",
                    "voxel_grid": matched_demo["matrix"],
                    "modality": matched_demo["modality"],
                    "site": matched_demo["site"],
                    "research_cohort": matched_demo["research_label"],
                    "demographics": matched_demo["demographics"]
                }
            elif matched_pheno:
                is_asd = (matched_pheno.get("DX_GROUP") == 1)
                scan_meta = {
                    "source": f"User Upload (ABIDE-I Cohort: Subject {matched_pheno.get('SUB_ID')})",
                    "filename": file.filename,
                    "dimensions": f"{arr.shape}",
                    "voxel_grid": f"{arr.shape}",
                    "modality": "Structural sMRI Tensor",
                    "site": f"{matched_pheno.get('SITE_ID', 'ABIDE')} Clinical Scanner",
                    "research_cohort": "ASD Cohort (ABIDE-I)" if is_asd else "Neurotypical Control Cohort (ABIDE-I)",
                    "demographics": {
                        "age": f"{matched_pheno.get('AGE_AT_SCAN', age or 'Not recorded')} yrs",
                        "sex": "Male" if matched_pheno.get("SEX") == 1 else "Female",
                        "fiq": str(matched_pheno.get("FIQ", "N/A"))
                    }
                }
            else:
                scan_meta = {
                    "source": "User Upload (.npy Tensor)",
                    "filename": file.filename,
                    "dimensions": f"{arr.shape}",
                    "voxel_grid": f"{arr.shape}",
                    "modality": "Structural sMRI Tensor",
                    "site": site or "Custom Upload Facility",
                    "research_cohort": "Custom Scan Evaluation",
                    "demographics": {
                        "age": f"{int(round(float(age)))} yrs" if age else "Not recorded",
                        "sex": sex or "Not recorded"
                    }
                }

            result["scan_metadata"] = scan_meta
            return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("web/favicon.svg", media_type="image/svg+xml")

os.makedirs("web", exist_ok=True)
app.mount("/", StaticFiles(directory="web", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
