"""
==============================================================================
ABIDE-I Structural MRI 3D Multi-Planar Preprocessing (128x128 NYU Pipeline)
------------------------------------------------------------------------------
Author: Clint Loyed
Target Site: NYU Langone Medical Center Cohort (N=184 Subjects)
Target Resolution: (50, 128, 128, 1) 3D Volumetric Tensors

Features:
  1. AWS S3 Automatic Downloader for NYU Scans
  2. Bounding Box Skull Stripping
  3. Z-Score Voxel Intensity Normalization [-3, +3]
  4. Multi-Planar 50-Slice Extractor (Axial, Coronal, Sagittal @ 128x128)
  5. Outputs 3D NumPy arrays to './processed_paper_3D/'
==============================================================================
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Cross-Platform Output Directory Configuration
if os.path.exists('/kaggle/working'):
    base_dir = '/kaggle/working/'
else:
    base_dir = './'

raw_dir = os.path.join(base_dir, 'raw_nyu')
output_dir = os.path.join(base_dir, 'processed_paper_3D')
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# 1. Dataset Acquisition (NYU Cohort)
print("📊 Fetching Official ABIDE Phenotypic Metadata for NYU Site...")
if os.path.exists('./data/ABIDE_Phenotypic.csv'):
    csv_url = './data/ABIDE_Phenotypic.csv'
else:
    csv_url = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"

df = pd.read_csv(csv_url)
nyu_df = df[df['SITE_ID'] == 'NYU'].copy()
nyu_df['PADDED_ID'] = nyu_df['SUB_ID'].astype(str).str.zfill(7)
print(f"🚀 Found {len(nyu_df)} total subjects for NYU. Processing into 128x128 3D Tensors...")

def download_patient(row):
    site = row['SITE_ID']
    pid = row['PADDED_ID']
    url = f"https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/RawData/{site}/{pid}/session_1/anat_1/mprage.nii.gz"
    dest = os.path.join(raw_dir, f"ABIDE_{pid}.nii.gz")
    
    if os.path.exists(dest):
        return True
    try:
        with urllib.request.urlopen(url, timeout=20) as response, open(dest, 'wb') as out_file:
            out_file.write(response.read())
        return True
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)
        return False

print("📥 Initializing AWS S3 Downloader for NYU Scans...")
successful_downloads = 0
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(download_patient, row): row['PADDED_ID'] for _, row in nyu_df.iterrows()}
    for future in as_completed(futures):
        if future.result():
            successful_downloads += 1

print(f"✅ Data Acquisition Complete. {successful_downloads} raw NYU scans available.")

# 2. 128x128 3D Preprocessing Mathematics

def crop_brain(volume):
    """Bounding Box Skull Stripping: Crops empty background space around the brain"""
    mask = volume > 0
    coords = np.array(np.nonzero(mask))
    if coords.size == 0:
        return volume
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    return volume[top_left[0]:bottom_right[0]+1,
                  top_left[1]:bottom_right[1]+1,
                  top_left[2]:bottom_right[2]+1]

def z_score_normalize(volume):
    """Z-Score Normalization: Rescales voxel brightness to zero-mean unit variance"""
    mask = volume > 0
    if not np.any(mask):
        return volume
    mean = np.mean(volume[mask])
    std = np.std(volume[mask])
    if std == 0:
        std = 1.0
    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[mask] = (volume[mask] - mean) / std
    return np.clip(normalized, -3, 3)

def extract_50_slices_128(volume, axis, target_size=(128, 128), num_slices=50):
    """Extracts 50 middle slices along specified axis into 128x128 resolution"""
    center = volume.shape[axis] // 2
    start = max(0, center - (num_slices // 2))
    end = min(volume.shape[axis], center + (num_slices // 2))
    
    slices_idx = np.linspace(start, end - 1, num_slices, dtype=int)
    extracted_tensor = np.zeros((num_slices, target_size[0], target_size[1], 1), dtype=np.float32)
    
    for i, idx in enumerate(slices_idx):
        if axis == 0:
            slice_data = volume[idx, :, :]
        elif axis == 1:
            slice_data = volume[:, idx, :]
        else:
            slice_data = volume[:, :, idx]
            
        resized = cv2.resize(slice_data, target_size, interpolation=cv2.INTER_CUBIC)
        extracted_tensor[i, :, :, 0] = resized
        
    return extracted_tensor

# 3. Batch Processing Pipeline
print("\n🚀 Executing 128x128 NYU 3D Processing Pipeline...")
processed_count = 0

for _, row in nyu_df.iterrows():
    pid = row['PADDED_ID']
    nifti_path = os.path.join(raw_dir, f"ABIDE_{pid}.nii.gz")
    
    if not os.path.exists(nifti_path):
        continue
    out_folder = os.path.join(output_dir, pid)
    os.makedirs(out_folder, exist_ok=True)
    
    try:
        img = nib.load(nifti_path)
        volume = img.get_fdata()
    except Exception:
        continue
        
    volume = crop_brain(volume)
    volume = z_score_normalize(volume)
    
    axial_tensor = extract_50_slices_128(volume, axis=2)
    coronal_tensor = extract_50_slices_128(volume, axis=1)
    sagittal_tensor = extract_50_slices_128(volume, axis=0)
    
    np.save(os.path.join(out_folder, "axial_50.npy"), axial_tensor)
    np.save(os.path.join(out_folder, "coronal_50.npy"), coronal_tensor)
    np.save(os.path.join(out_folder, "sagittal_50.npy"), sagittal_tensor)
    
    processed_count += 1
    if processed_count % 20 == 0:
        print(f"⚙️ Processed {processed_count} NYU subjects...")

print(f"\n🎉 SUCCESS! Fully processed {processed_count} NYU subjects into 128x128 3D Tensors in '{output_dir}'.")
