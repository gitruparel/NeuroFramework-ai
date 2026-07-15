"""MRI spatial resampling mapping voxel spacing coordinates to uniform resolutions (e.g. 1.0mm)."""

import numpy as np
import SimpleITK as sitk
from typing import Any, List
from preprocessing.base import BaseTransform, TransformRegistry, mri_data_to_sitk, sitk_to_mri_data
from preprocessing.decorators import trace_transform
from schemas.mri import MRIData, ScanStatistics
from schemas.processing import ExecutionContext


@TransformRegistry.register("resample")
class Resampler(BaseTransform):
    """Resamples MRI volumes to uniform voxel spacing dimensions using linear interpolation."""

    def __init__(self, spacing: List[float], **kwargs: Any):
        super().__init__(spacing=spacing, **kwargs)

    @trace_transform
    def process(self, mri_data: MRIData, context: ExecutionContext) -> MRIData:
        target_spacing = self.params.get("spacing")
        
        # Convert to SITK
        sitk_img = mri_data_to_sitk(mri_data)
        
        original_spacing = sitk_img.GetSpacing()
        original_size = sitk_img.GetSize()
        
        if list(original_spacing) == list(target_spacing):
            context.logger.info("Resampler: Spacing is already at target spacing. Skipping.")
            return mri_data.model_copy(deep=True)
            
        # Compute target size dimensions
        new_size = [
            int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
            for i in range(3)
        ]
        
        # Configure SimpleITK Resampler
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetOutputSpacing(target_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(sitk_img.GetDirection())
        resample.SetOutputOrigin(sitk_img.GetOrigin())
        resample.SetTransform(sitk.Transform())
        
        resampled_img = resample.Execute(sitk_img)
        
        # Store inverse details in params
        self.params["original_spacing"] = str(list(original_spacing))
        active_tensor = mri_data.image if mri_data.image is not None else mri_data.raw.tensor
        self.params["original_shape"] = str(list(active_tensor.shape))
        
        # Convert back
        mri_copy = sitk_to_mri_data(resampled_img, mri_data)
        
        # Resample brain mask if present
        if mri_data.brain_mask is not None:
            mask_3d = mri_data.brain_mask[0] if len(mri_data.brain_mask.shape) == 4 else mri_data.brain_mask
            sitk_mask = sitk.GetImageFromArray(np.transpose(mask_3d.astype(np.uint8), (2, 1, 0)))
            sitk_mask.SetOrigin(sitk_img.GetOrigin())
            sitk_mask.SetSpacing(sitk_img.GetSpacing())
            sitk_mask.SetDirection(sitk_img.GetDirection())
            
            resample_mask = sitk.ResampleImageFilter()
            resample_mask.SetInterpolator(sitk.sitkNearestNeighbor)
            resample_mask.SetOutputSpacing(target_spacing)
            resample_mask.SetSize(new_size)
            resample_mask.SetOutputDirection(sitk_img.GetDirection())
            resample_mask.SetOutputOrigin(sitk_img.GetOrigin())
            resample_mask.SetTransform(sitk.Transform())
            
            resampled_mask = resample_mask.Execute(sitk_mask)
            mask_arr = np.transpose(sitk.GetArrayFromImage(resampled_mask), (2, 1, 0)).astype(np.float32)
            if len(mri_data.brain_mask.shape) == 4:
                mask_arr = np.expand_dims(mask_arr, axis=0)
            mri_copy.brain_mask = mask_arr

        
        # Recalculate statistics
        new_tensor = mri_copy.image
        mri_copy.statistics = ScanStatistics(
            min=float(np.min(new_tensor)),
            max=float(np.max(new_tensor)),
            mean=float(np.mean(new_tensor)),
            std=float(np.std(new_tensor)),
            shape=list(new_tensor.shape),
            dtype=str(new_tensor.dtype)
        )
        
        return mri_copy
