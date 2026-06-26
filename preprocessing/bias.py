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
        import time
        start_time = time.time()

        mode = self.params.get("mode", "fast").lower()
        if mode == "off":
            context.logger.info("BiasFieldCorrector: mode is 'off'. Skipping bias correction.")
            return mri_data

        strategy = self.params.get("strategy", "n4")
        if strategy.lower() != "n4":
            raise ValueError(f"BiasFieldCorrector: Strategy '{strategy}' is not supported. Use 'n4'.")

        # Set default parameters based on mode
        if mode == "fast":
            default_shrink = 4
            default_iterations = [30, 20, 10]
        elif mode == "full":
            default_shrink = 2
            default_iterations = [50, 50, 30, 20]
        else:
            default_shrink = 1
            default_iterations = [50, 50, 50, 50]

        shrink_factor = int(self.params.get("shrink_factor", default_shrink))
        max_iterations = self.params.get("max_iterations", default_iterations)
        convergence_threshold = float(self.params.get("convergence_threshold", 1e-6))

        context.logger.info(
            f"BiasFieldCorrector: Running N4 correction. Mode: {mode}, Shrink Factor: {shrink_factor}, "
            f"Iterations: {max_iterations}, Threshold: {convergence_threshold}"
        )

        sitk_img = mri_data_to_sitk(mri_data)
        sitk_img_float = sitk.Cast(sitk_img, sitk.sitkFloat32)

        # Prepare brain mask if present
        sitk_mask = None
        if mri_data.brain_mask is not None:
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

        # Execution using optional shrink factor
        if shrink_factor > 1:
            shrunk_img = sitk.Shrink(sitk_img_float, [shrink_factor] * 3)
            shrunk_mask = sitk.Shrink(sitk_mask, [shrink_factor] * 3) if sitk_mask is not None else None

            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations(max_iterations)
            corrector.SetConvergenceThreshold(convergence_threshold)

            if shrunk_mask is not None:
                corrected_shrunk = corrector.Execute(shrunk_img, shrunk_mask)
            else:
                corrected_shrunk = corrector.Execute(shrunk_img)

            # Get bias field = original_shrunk / corrected_shrunk
            bias_field_shrunk = sitk.Divide(shrunk_img, corrected_shrunk)

            # Clamp shrunk bias field values to prevent division by zero or NaN values
            bias_np = sitk.GetArrayFromImage(bias_field_shrunk)
            bias_np = np.clip(bias_np, 0.01, 100.0)
            bias_np = np.nan_to_num(bias_np, nan=1.0, posinf=1.0, neginf=1.0)
            bias_field_shrunk_clamped = sitk.GetImageFromArray(bias_np)
            bias_field_shrunk_clamped.CopyInformation(bias_field_shrunk)

            # Resample bias field back to original size
            bias_field = sitk.Resample(
                bias_field_shrunk_clamped,
                sitk_img_float,
                sitk.Transform(),
                sitk.sitkBSpline,
                1.0,
                sitk_img_float.GetPixelID(),
            )

            # Clamp full-resolution bias field values
            bias_np_full = sitk.GetArrayFromImage(bias_field)
            bias_np_full = np.clip(bias_np_full, 0.01, 100.0)
            bias_np_full = np.nan_to_num(bias_np_full, nan=1.0, posinf=1.0, neginf=1.0)
            bias_field_clamped = sitk.GetImageFromArray(bias_np_full)
            bias_field_clamped.CopyInformation(bias_field)

            # Correct original image
            corrected_img = sitk.Divide(sitk_img_float, bias_field_clamped)
        else:
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations(max_iterations)
            corrector.SetConvergenceThreshold(convergence_threshold)

            if sitk_mask is not None:
                corrected_img = corrector.Execute(sitk_img_float, sitk_mask)
            else:
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

        elapsed = time.time() - start_time
        context.logger.info(f"BiasFieldCorrector: Completed in {elapsed:.4f} seconds.")

        return mri_copy
