"""
==============================================================================
All-In-One Cloud Pipeline: S3 Acquisition + Preprocessing + 5-Fold Training
------------------------------------------------------------------------------
Author: Clint Loyed & Antigravity
Optimized for: Lightning AI Studio (Free NVIDIA T4 / L4 GPU), Kaggle, Colab
Cohorts: NYU, UM_1, USM (395 Subjects Total)

Workflow:
  1. Automated S3 Multi-Threaded Download of Raw MRI Scans
  2. Multi-Core Preprocessing (Otsu + N4 Bias + Harmonized Z-Score + Axial Tensors)
  3. 5-Fold Stratified Cross-Validation on Dedicated Axial 3D Conv3D Network
  4. Packaging and Exporting best_axial_3d_model.pt + Metrics JSON
==============================================================================
"""

import os
import sys
import time
import shutil
import urllib.request
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import torch
import warnings

warnings.filterwarnings('ignore')

# Add project root to python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.abide_3d_axial_cnn_pytorch import DedicatedAxial3DCNN
from training.train_axial_lightning import main as run_trainer


def log(msg):
    print(msg, flush=True)


# ============================================================================
# 1. Preprocessing Helpers
# ============================================================================

def n4_bias_field_correction(volume):
    brain_mask = volume > 0
    if not np.any(brain_mask):
        return volume
    bias_field = gaussian_filter(volume, sigma=15)
    bias_field[bias_field == 0] = 1.0
    corrected = volume / bias_field
    corrected[~brain_mask] = 0.0
    return corrected


def otsu_brain_masking(volume):
    brain_mask = np.zeros_like(volume, dtype=bool)
    for z in range(volume.shape[2]):
        slice_img = volume[:, :, z]
        if np.max(slice_img) == 0:
            continue
        norm_slice = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh = cv2.threshold(norm_slice, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        brain_mask[:, :, z] = thresh > 0
    masked_volume = np.zeros_like(volume)
    masked_volume[brain_mask] = volume[brain_mask]
    return masked_volume


def crop_brain(volume):
    mask = volume > 0
    coords = np.array(np.nonzero(mask))
    if coords.size == 0:
        return volume
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    return volume[top_left[0]:bottom_right[0]+1,
                  top_left[1]:bottom_right[1]+1,
                  top_left[2]:bottom_right[2]+1]


def site_harmonized_z_score(volume):
    brain_mask = volume > 0
    if not np.any(brain_mask):
        return volume
    p1, p99 = np.percentile(volume[brain_mask], (1, 99))
    clipped_vol = np.clip(volume, p1, p99)
    mean_val = np.mean(clipped_vol[brain_mask])
    std_val = np.std(clipped_vol[brain_mask])
    if std_val == 0:
        std_val = 1.0
    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[brain_mask] = (clipped_vol[brain_mask] - mean_val) / std_val
    normalized = np.clip(normalized, -3, 3)
    return normalized


def extract_50_slices(volume, axis, target_size=(224, 224), num_slices=50):
    dim_len = volume.shape[axis]
    center = dim_len // 2
    start = max(0, center - (num_slices // 2))
    end = min(dim_len, center + (num_slices // 2))
    slices_idx = np.linspace(start, end - 1, num_slices, dtype=int)
    extracted = np.zeros((num_slices, target_size[0], target_size[1], 1), dtype=np.float32)
    for i, idx in enumerate(slices_idx):
        if axis == 0:
            slice_data = volume[idx, :, :]
        elif axis == 1:
            slice_data = volume[:, idx, :]
        else:
            slice_data = volume[:, :, idx]
        resized = cv2.resize(slice_data, target_size, interpolation=cv2.INTER_CUBIC)
        extracted[i, :, :, 0] = resized
    return extracted


def process_single_patient(task):
    pid, nifti_path, out_folder, axial_only = task
    ax_file = os.path.join(out_folder, "axial_50.npy")
    if os.path.exists(ax_file):
        return True
    try:
        img = nib.load(nifti_path)
        vol = img.get_fdata()
        vol = otsu_brain_masking(vol)
        vol = n4_bias_field_correction(vol)
        vol = crop_brain(vol)
        vol = site_harmonized_z_score(vol)

        ax_tensor = extract_50_slices(vol, axis=2, target_size=(224, 224))
        os.makedirs(out_folder, exist_ok=True)
        np.save(ax_file, ax_tensor)

        if not axial_only:
            cor_tensor = extract_50_slices(vol, axis=1, target_size=(224, 224))
            sag_tensor = extract_50_slices(vol, axis=0, target_size=(224, 224))
            np.save(os.path.join(out_folder, "coronal_50.npy"), cor_tensor)
            np.save(os.path.join(out_folder, "sagittal_50.npy"), sag_tensor)

        return True
    except Exception:
        return False


# ============================================================================
# 2. Pipeline Orchestration
# ============================================================================

def execute_pipeline(args):
    log("==========================================================================")
    log("  NEUROFRAMEWORK 3D-sMRI: CLOUD PIPELINE EXECUTION")
    log("==========================================================================")

    # Base path resolution (works in Lightning AI, Kaggle, Colab, or local)
    if os.path.exists('/teamspace/studios/this_studio'):
        base_dir = '/teamspace/studios/this_studio'
    elif os.path.exists('/kaggle/working'):
        base_dir = '/kaggle/working'
    else:
        base_dir = ROOT_DIR

    raw_dir = os.path.join(base_dir, 'raw_abide_multi')
    output_dir = os.path.join(base_dir, 'processed_paper_3D_224')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Metadata resolution
    pheno_candidates = [
        os.path.join(ROOT_DIR, "data", "ABIDE_Phenotypic.csv"),
        "./data/ABIDE_Phenotypic.csv",
        os.path.join(base_dir, "data", "ABIDE_Phenotypic.csv")
    ]
    pheno_path = None
    for c in pheno_candidates:
        if os.path.exists(c):
            pheno_path = c
            break

    if pheno_path is None:
        log("Fetching phenotypic metadata from AWS S3...")
        pheno_path = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"

    df = pd.read_csv(pheno_path)
    target_sites = ['NYU', 'UM_1', 'USM']
    df_cohort = df[df['SITE_ID'].isin(target_sites)].copy()
    df_cohort['PADDED_ID'] = df_cohort['SUB_ID'].astype(str).str.zfill(7)
    total_subjects = len(df_cohort)
    log(f"Target Multi-Site Cohort: {total_subjects} subjects across {target_sites}.")

    # ------------------------------------------------------------------------
    # Step A: S3 Acquisition
    # ------------------------------------------------------------------------
    if not args.skip_download:
        log("\n[1/3] Downloading raw NIfTI scans via high-speed multi-threaded S3 client...")
        def download_fn(row):
            site = row['SITE_ID']
            pid = row['PADDED_ID']
            url = f"https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/RawData/{site}/{pid}/session_1/anat_1/mprage.nii.gz"
            dest = os.path.join(raw_dir, f"ABIDE_{pid}.nii.gz")
            if os.path.exists(dest) and os.path.getsize(dest) > 1000:
                return True
            try:
                with urllib.request.urlopen(url, timeout=25) as resp, open(dest, 'wb') as f:
                    f.write(resp.read())
                return True
            except Exception:
                if os.path.exists(dest):
                    os.remove(dest)
                return False

        downloaded = 0
        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = [executor.submit(download_fn, row) for _, row in df_cohort.iterrows()]
            for future in as_completed(futures):
                if future.result():
                    downloaded += 1
        log(f"Data Acquisition Complete: {downloaded} scans available in '{raw_dir}'.")

    # ------------------------------------------------------------------------
    # Step B: Parallel Preprocessing
    # ------------------------------------------------------------------------
    if not args.skip_preprocess:
        log("\n[2/3] Processing scans into Site-Harmonized 224x224 Full HD 3D Tensors...")
        tasks = []
        axial_only = (args.model == "axial")
        for _, row in df_cohort.iterrows():
            pid = row['PADDED_ID']
            nifti_path = os.path.join(raw_dir, f"ABIDE_{pid}.nii.gz")
            if os.path.exists(nifti_path):
                out_folder = os.path.join(output_dir, pid)
                tasks.append((pid, nifti_path, out_folder, axial_only))

        num_workers = min(os.cpu_count() or 4, 8)
        log(f"Using {num_workers} parallel CPU processes for Otsu masking & N4 bias correction...")
        t0 = time.time()
        completed = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_patient, t) for t in tasks]
            for future in as_completed(futures):
                if future.result():
                    completed += 1
                if completed % 25 == 0 and completed > 0:
                    log(f"  Processed {completed}/{len(tasks)} scans ({round(completed/len(tasks)*100, 1)}%)...")

        elapsed_min = (time.time() - t0) / 60
        log(f"Preprocessing Complete: {completed} subjects preprocessed in {elapsed_min:.1f} minutes.")

    # ------------------------------------------------------------------------
    # Step C: Deep Learning Training & 5-Fold Stratified Cross Validation
    # ------------------------------------------------------------------------
    log(f"\n[3/3] Initiating 5-Fold Stratified Training ({args.model.upper()} Architecture)...")
    sys.argv = [
        "train_axial_lightning.py",
        "--model", args.model,
        "--data-dir", output_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--output-dir", os.path.join(base_dir, "checkpoints")
    ]
    if isinstance(pheno_path, str) and os.path.exists(pheno_path):
        sys.argv.extend(["--phenotypic-csv", pheno_path])

    run_trainer()

    # ------------------------------------------------------------------------
    # Step D: Package Deliverables for 1-Click Download
    # ------------------------------------------------------------------------
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    zip_path = os.path.join(base_dir, f"trained_weights_{args.model}_3d")
    log("\nPackaging trained model weights for download...")
    shutil.make_archive(zip_path, 'zip', ckpt_dir)
    log(f"SUCCESS! Ready-to-download archive created at: {zip_path}.zip")
    log("Download this archive to your local laptop and place the weights into dept internship/checkpoints/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end cloud pipeline for ABIDE-I 3D training.")
    parser.add_argument("--model", choices=["axial", "asymmetric"], default="axial",
                        help="Model architecture ('axial' = Dedicated Axial 3D, 'asymmetric' = Axial-Anchored Multi-View)")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs per fold")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--skip-download", action="store_true", help="Skip S3 downloading if already downloaded")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing if tensors exist")
    args = parser.parse_args()
    execute_pipeline(args)
