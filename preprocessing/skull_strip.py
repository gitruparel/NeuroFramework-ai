"""MRI skull stripping implementing the strategy pattern with threshold and deep learning placeholders."""

import numpy as np
import SimpleITK as sitk
from typing import Any, Dict
from preprocessing.base import BaseTransform, BaseTransformStrategy, TransformRegistry, mri_data_to_sitk
from preprocessing.decorators import trace_transform
from schemas.mri import MRIData, ScanStatistics
from schemas.processing import ExecutionContext


class BaseSkullStripperStrategy(BaseTransformStrategy):
    """Abstract base strategy for skull stripping algorithms."""
    pass


class ThresholdSkullStripperStrategy(BaseSkullStripperStrategy):
    """Slower baseline thresholding strategy using Otsu and morphology cleanup."""

    def execute(self, mri_data: MRIData, context: ExecutionContext, params: Dict[str, Any]) -> np.ndarray:
        context.logger.info("SkullStripper [ThresholdStrategy]: Segmenting using Otsu thresholding.")
        sitk_img = mri_data_to_sitk(mri_data)
        
        # Run Otsu
        otsu = sitk.OtsuThresholdImageFilter()
        otsu.SetInsideValue(0)
        otsu.SetOutsideValue(1)
        mask_img = otsu.Execute(sitk_img)
        
        # Morphological operations to clean up scalp boundaries and fill holes
        mask_img = sitk.BinaryMorphologicalClosing(mask_img, [3, 3, 3])
        mask_img = sitk.BinaryMorphologicalOpening(mask_img, [1, 1, 1])
        
        # Convert SimpleITK image to NumPy array (transpose back to match coordinates)
        mask_arr = np.transpose(sitk.GetArrayFromImage(mask_img), (2, 1, 0)).astype(np.float32)
        
        # Ensure dimensions match image
        active_tensor = mri_data.image if mri_data.image is not None else mri_data.raw.tensor
        if len(active_tensor.shape) == 4 and len(mask_arr.shape) == 3:
            mask_arr = np.expand_dims(mask_arr, axis=0)
            
        return mask_arr


class SynthStripSkullStripperStrategy(BaseSkullStripperStrategy):
    """Placeholder strategy for DL-based SynthStrip algorithm."""

    def execute(self, mri_data: MRIData, context: ExecutionContext, params: Dict[str, Any]) -> np.ndarray:
        context.logger.warning(
            "SkullStripper [SynthStripStrategy]: Deep Learning SynthStrip is not implemented in development mode. "
            "Falling back to baseline ThresholdStrategy."
        )
        return ThresholdSkullStripperStrategy().execute(mri_data, context, params)


class FastSurferSkullStripperStrategy(BaseSkullStripperStrategy):
    """Placeholder strategy for DL-based FastSurfer segmentation strip."""

    def execute(self, mri_data: MRIData, context: ExecutionContext, params: Dict[str, Any]) -> np.ndarray:
        context.logger.warning(
            "SkullStripper [FastSurferStrategy]: Deep Learning FastSurfer is not implemented in development mode. "
            "Falling back to baseline ThresholdStrategy."
        )
        return ThresholdSkullStripperStrategy().execute(mri_data, context, params)


@TransformRegistry.register("skull_strip")
class SkullStripper(BaseTransform):
    """Isolates brain volume from background scalp tissues using a strategy pattern."""

    def __init__(self, strategy: str = "threshold", **kwargs: Any):
        super().__init__(strategy=strategy, **kwargs)

    @trace_transform
    def process(self, mri_data: MRIData, context: ExecutionContext) -> MRIData:
        strategy_name = self.params.get("strategy", "threshold").lower()
        
        # Resolve strategy
        if strategy_name == "threshold":
            strategy = ThresholdSkullStripperStrategy()
        elif strategy_name == "synthstrip":
            strategy = SynthStripSkullStripperStrategy()
        elif strategy_name == "fastsurfer":
            strategy = FastSurferSkullStripperStrategy()
        else:
            context.logger.warning(
                f"SkullStripper: Unknown strategy '{strategy_name}'. Falling back to ThresholdStrategy."
            )
            strategy = ThresholdSkullStripperStrategy()
            
        # Execute selected strategy to extract binary mask
        mask = strategy.execute(mri_data, context, self.params)
        
        # Mask active image
        tensor = mri_data.image
        new_tensor = tensor * mask
        
        mri_copy = mri_data.model_copy(deep=True)
        mri_copy.image = new_tensor
        mri_copy.brain_mask = mask
        
        # Recalculate statistics
        mri_copy.statistics = ScanStatistics(
            min=float(np.min(new_tensor)),
            max=float(np.max(new_tensor)),
            mean=float(np.mean(new_tensor)),
            std=float(np.std(new_tensor)),
            shape=list(new_tensor.shape),
            dtype=str(new_tensor.dtype)
        )
        
        return mri_copy
