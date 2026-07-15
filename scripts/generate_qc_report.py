import json
import logging
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
import yaml

from preprocessing.pipeline import PreprocessingPipeline
from engine.readers.nifti import NiftiReader
from training.preprocess_autism import wrap_raw_mri
from schemas.processing import ExecutionContext

def get_bounding_box(tensor):
    non_zero = np.argwhere(tensor != 0)
    if non_zero.size == 0:
        return [0, 0, 0]
    min_idx = non_zero.min(axis=0)
    max_idx = non_zero.max(axis=0)
    return [int(x) for x in (max_idx - min_idx + 1)]

def run_qc_report():
    print("Initializing Quality Control (QC) Report Generator...")
    index_file = Path("d:/Coding/dept internship/data/abide_index.json")
    config_yaml = Path("d:/Coding/dept internship/configs/preprocessing.yaml")
    reports_dir = Path("d:/Coding/dept internship/reports")
    reports_dir.mkdir(exist_ok=True)
    
    with open(index_file, encoding="utf-8") as f:
        items = json.load(f)
        
    # Get first 5 subjects
    subjects = items[:5]
    
    reader = NiftiReader()
    ctx = ExecutionContext(
        logger=logging.getLogger("qc_report"),
        cache=None,
        config={},
        seed=42,
        device="cpu"
    )
    
    md_content = [
        "# Preprocessing Quality Control (QC) Report — V1.5.0",
        "",
        "This report provides visual slice grids and volumetric metrics verifying the performance of the upgraded preprocessing pipeline.",
        "",
        "## Volumetric and Intensity Metrics Table",
        "",
        "| Subject ID | Label | Brain Volume (ml) | Mask % | Bounding Box Size | Spacing (mm) | Mean | Std | Min | Max |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    # We will build step-by-step pipeline pipelines to extract intermediate states
    # 1. Pipeline up to Skull Strip
    with open(config_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    steps = cfg.get("profiles", {}).get("autism", [])
    
    # Setup sub-pipelines
    # Find steps indices
    reorient_step = None
    skull_strip_step = None
    crop_step = None
    bias_step = None
    resample_step = None
    normalize_step = None
    pad_step = None
    center_crop_step = None
    
    from preprocessing.base import TransformRegistry, mri_data_to_sitk
    
    reorient_tx = TransformRegistry.create("reorient", target="RAS")
    skull_strip_tx = TransformRegistry.create("skull_strip", strategy="morphological")
    crop_tx = TransformRegistry.create("crop_foreground")
    bias_tx = TransformRegistry.create("bias_correction", strategy="n4", mode="fast")
    resample_tx = TransformRegistry.create("resample", spacing=[1.5, 1.5, 1.5])
    normalize_tx = TransformRegistry.create("normalize", mode="z_score")
    pad_tx = TransformRegistry.create("pad", target_shape=[128, 128, 128])
    cc_tx = TransformRegistry.create("center_crop", target_shape=[128, 128, 128])
    
    for idx, item in enumerate(subjects):
        sub_id = item["subject_id"]
        label = item["label"]
        raw_path = Path(item["path"])
        
        print(f"\nProcessing QC for {sub_id} ({idx+1}/5)...")
        raw_mri = reader.read(raw_path)
        mri_data = wrap_raw_mri(raw_mri)
        
        # 1. Raw
        raw_arr = raw_mri.tensor
        
        # 2. Reorient + Skull Strip
        mri_reorient = reorient_tx.process(mri_data, ctx)
        mri_stripped = skull_strip_tx.process(mri_reorient, ctx)
        brain_mask = mri_stripped.brain_mask
        
        # Calculate volume & mask %
        # Head mask (Otsu) size
        head_otsu = sitk.OtsuThresholdImageFilter()
        head_otsu.SetInsideValue(0)
        head_otsu.SetOutsideValue(1)
        sitk_raw = mri_data_to_sitk(mri_reorient)
        head_mask_sitk = head_otsu.Execute(sitk_raw)
        head_mask_arr = sitk.GetArrayFromImage(head_mask_sitk)
        head_voxel_count = float((head_mask_arr != 0).sum())
        
        brain_voxel_count = float((brain_mask != 0).sum())
        spacing = sitk_raw.GetSpacing()
        voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
        brain_vol_ml = brain_voxel_count * voxel_vol_ml
        mask_pct = (brain_voxel_count / head_voxel_count) * 100.0 if head_voxel_count > 0 else 0.0
        
        bbox = get_bounding_box(brain_mask[0] if len(brain_mask.shape) == 4 else brain_mask)
        
        # 3. Crop + Bias Correction
        mri_cropped = crop_tx.process(mri_stripped, ctx)
        mri_biased = bias_tx.process(mri_cropped, ctx)
        
        # 4. Resample + Normalize + Pad + CenterCrop (Final tensor)
        mri_resampled = resample_tx.process(mri_biased, ctx)
        mri_normalized = normalize_tx.process(mri_resampled, ctx)
        mri_padded = pad_tx.process(mri_normalized, ctx)
        mri_final = cc_tx.process(mri_padded, ctx)
        
        final_tensor = mri_final.image
        
        f_mean = float(np.mean(final_tensor[final_tensor != 0])) if np.any(final_tensor != 0) else 0.0
        f_std = float(np.std(final_tensor[final_tensor != 0])) if np.any(final_tensor != 0) else 0.0
        f_min = float(np.min(final_tensor))
        f_max = float(np.max(final_tensor))
        
        # Write markdown row
        md_content.append(
            f"| {sub_id} | {label} | {brain_vol_ml:.1f} ml | {mask_pct:.1f}% | {bbox} | {list(spacing)} | {f_mean:.3f} | {f_std:.3f} | {f_min:.3f} | {f_max:.3f} |"
        )
        
        # Plot intermediate slices
        # Extract coronal/axial center slice
        raw_slice = raw_arr[raw_arr.shape[0] // 2]
        # For brain mask, we need to extract from aligned space
        stripped_arr = mri_stripped.image[0] if len(mri_stripped.image.shape) == 4 else mri_stripped.image
        stripped_slice = stripped_arr[stripped_arr.shape[0] // 2]
        
        biased_arr = mri_biased.image[0] if len(mri_biased.image.shape) == 4 else mri_biased.image
        biased_slice = biased_arr[biased_arr.shape[0] // 2]
        
        final_arr = final_tensor[0] if len(final_tensor.shape) == 4 else final_tensor
        final_slice = final_arr[final_arr.shape[0] // 2]
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(np.rot90(raw_slice), cmap="gray")
        axes[0].set_title(f"1. Raw Scan ({raw_arr.shape})")
        axes[0].axis("off")
        
        axes[1].imshow(np.rot90(stripped_slice), cmap="gray")
        axes[1].set_title(f"2. Brain Mask (Vol: {brain_vol_ml:.1f}ml)")
        axes[1].axis("off")
        
        axes[2].imshow(np.rot90(biased_slice), cmap="gray")
        axes[2].set_title("3. Bias Corrected")
        axes[2].axis("off")
        
        axes[3].imshow(np.rot90(final_slice), cmap="gray")
        axes[3].set_title(f"4. Final Tensor ({final_arr.shape})")
        axes[3].axis("off")
        
        plot_path = reports_dir / f"qc_{sub_id}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Slice grid saved to: {plot_path}")
        
    md_path = reports_dir / "preprocessing_qc_v1_5.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    print(f"\nQC Report generated successfully! Saved to: {md_path}")

if __name__ == "__main__":
    run_qc_report()
