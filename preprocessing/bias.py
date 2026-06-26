"""MRI N4 Bias Field Correction using SimpleITK."""

import numpy as np
import SimpleITK as sitk
from typing import Any
from preprocessing.base import BaseTransform, TransformRegistry, mri_data_to_sitk, sitk_to_mri_data
from preprocessing.decorators import trace_transform
from schemas.mri import MRIData, ScanStatistics
from schemas.processing import ExecutionContext


@TransformRegistry.register("bias_correction")
class BiasFieldCorrector(BaseTransform):
    """Corrects low-frequency intensity bias field inhomogeneities using N4 correction."""

    def __init__(self, strategy: str = "n4", **kwargs: Any):
        super().__init__(strategy=strategy, **kwargs)

    @trace_transform
    def process(self, mri_data: MRIData, context: ExecutionContext) -> MRIData:
        strategy = self.params.get("strategy", "n4")
        if strategy.lower() != "n4":
            raise ValueError(f"BiasFieldCorrector: Strategy '{strategy}' is not supported. Use 'n4'.")
            
        sitk_img = mri_data_to_sitk(mri_data)
        sitk_img_float = sitk.Cast(sitk_img, sitk.sitkFloat32)
        
        # Configure corrector
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        
        # Incorporate brain mask if present to restrict corrector field estimation
        if mri_data.brain_mask is not None:
            context.logger.info("BiasFieldCorrector: Found brain_mask. Running masked N4 correction.")
            
            mask_tensor = mri_data.brain_mask
            if len(mask_tensor.shape) == 4:
                mask_3d = mask_tensor[0]
            else:
                mask_3d = mask_tensor
                
            # Convert mask to SITK matching target image parameters
            sitk_mask = sitk.GetImageFromArray(np.transpose(mask_3d.astype(np.uint8), (2, 1, 0)))
            sitk_mask.SetOrigin(sitk_img.GetOrigin())
            sitk_mask.SetSpacing(sitk_img.GetSpacing())
            sitk_mask.SetDirection(sitk_img.GetDirection())
            sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)
            
            corrected_img = corrector.Execute(sitk_img_float, sitk_mask)
        else:
            context.logger.info("BiasFieldCorrector: Running standard unmasked N4 correction.")
            corrected_img = corrector.Execute(sitk_img_float)
            
        mri_copy = sitk_to_mri_data(corrected_img, mri_data)
        
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
