"""Spatial transforms for MRI volumes including foreground cropping, padding, and center cropping."""

import numpy as np
from typing import Any, Dict, List
from preprocessing.base import BaseTransform, TransformRegistry
from preprocessing.decorators import trace_transform
from schemas.mri import MRIData, ScanStatistics
from schemas.processing import ExecutionContext


@TransformRegistry.register("crop_foreground")
class ForegroundCropper(BaseTransform):
    """Trims zero-intensity margins from MRI borders."""

    @trace_transform
    def process(self, mri_data: MRIData, context: ExecutionContext) -> MRIData:
        tensor = mri_data.image
        
        # Determine non-zero bounds across spatial dims
        if len(tensor.shape) == 4:
            non_zero = np.argwhere(np.any(tensor != 0, axis=0))
        else:
            non_zero = np.argwhere(tensor != 0)
            
        if non_zero.size == 0:
            context.logger.warning("ForegroundCropper: No foreground voxels detected. Skipping crop.")
            return mri_data.model_copy(deep=True)

        min_idx = non_zero.min(axis=0)
        max_idx = non_zero.max(axis=0)
        
        # Perform crop
        if len(tensor.shape) == 4:
            new_tensor = tensor[:, min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1]
        else:
            new_tensor = tensor[min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1]
            
        # Shift translation component of affine to map crop origin correctly
        new_affine = mri_data.affine.copy()
        new_affine[:3, 3] = mri_data.affine[:3, :3] @ min_idx + mri_data.affine[:3, 3]
        
        # Save provenance details for inverse transform playback
        self.params["original_shape"] = str(list(tensor.shape))
        self.params["crop_offsets"] = str([int(x) for x in min_idx])
        
        mri_copy = mri_data.model_copy(deep=True)
        mri_copy.image = new_tensor
        mri_copy.affine = new_affine
        mri_copy.metadata.image.dimensions = list(new_tensor.shape)
        
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


@TransformRegistry.register("pad")
class Pad(BaseTransform):
    """Symmetrically pads volume spatial grid to target shape size."""

    def __init__(self, target_shape: List[int], mode: str = "constant", value: float = 0.0, **kwargs: Any):
        super().__init__(target_shape=target_shape, mode=mode, value=value, **kwargs)

    @trace_transform
    def process(self, mri_data: MRIData, context: ExecutionContext) -> MRIData:
        tensor = mri_data.image
        target_shape = self.params.get("target_shape")
        mode = self.params.get("mode", "constant")
        value = self.params.get("value", 0.0)
        
        spatial_shape = list(tensor.shape[-3:])
        diff = [t - s for t, s in zip(target_shape, spatial_shape)]
        
        pad_width = []
        pad_offsets = []
        for d in diff:
            if d > 0:
                before = d // 2
                after = d - before
                pad_width.append((before, after))
                pad_offsets.append(before)
            else:
                pad_width.append((0, 0))
                pad_offsets.append(0)
                
        # Symmetrically pad spatial dimensions
        if len(tensor.shape) == 4:
            new_tensor = np.pad(tensor, [(0, 0)] + pad_width, mode=mode, constant_values=value)
        else:
            new_tensor = np.pad(tensor, pad_width, mode=mode, constant_values=value)
            
        # Adjust affine translation origin to match shifted coordinate grid
        new_affine = mri_data.affine.copy()
        new_affine[:3, 3] = mri_data.affine[:3, 3] - mri_data.affine[:3, :3] @ pad_offsets
        
        # Save inverse params
        self.params["original_shape"] = str(list(tensor.shape))
        self.params["pad_offsets"] = str(list(pad_offsets))
        
        mri_copy = mri_data.model_copy(deep=True)
        mri_copy.image = new_tensor
        mri_copy.affine = new_affine
        mri_copy.metadata.image.dimensions = list(new_tensor.shape)
        
        mri_copy.statistics = ScanStatistics(
            min=float(np.min(new_tensor)),
            max=float(np.max(new_tensor)),
            mean=float(np.mean(new_tensor)),
            std=float(np.std(new_tensor)),
            shape=list(new_tensor.shape),
            dtype=str(new_tensor.dtype)
        )
        
        return mri_copy


@TransformRegistry.register("center_crop")
class CenterCrop(BaseTransform):
    """Symmetrically crops volume spatial grid to target shape size."""

    def __init__(self, target_shape: List[int], **kwargs: Any):
        super().__init__(target_shape=target_shape, **kwargs)

    @trace_transform
    def process(self, mri_data: MRIData, context: ExecutionContext) -> MRIData:
        tensor = mri_data.image
        target_shape = self.params.get("target_shape")
        
        spatial_shape = list(tensor.shape[-3:])
        diff = [s - t for s, t in zip(spatial_shape, target_shape)]
        
        slices = []
        crop_offsets = []
        for s, t, d in zip(spatial_shape, target_shape, diff):
            if d > 0:
                before = d // 2
                slices.append(slice(before, before + t))
                crop_offsets.append(before)
            else:
                slices.append(slice(0, s))
                crop_offsets.append(0)
                
        if len(tensor.shape) == 4:
            new_tensor = tensor[:, slices[0], slices[1], slices[2]]
        else:
            new_tensor = tensor[slices[0], slices[1], slices[2]]
            
        new_affine = mri_data.affine.copy()
        new_affine[:3, 3] = mri_data.affine[:3, 3] + mri_data.affine[:3, :3] @ crop_offsets
        
        # Save inverse params
        self.params["original_shape"] = str(list(tensor.shape))
        self.params["crop_offsets"] = str(list(crop_offsets))
        
        mri_copy = mri_data.model_copy(deep=True)
        mri_copy.image = new_tensor
        mri_copy.affine = new_affine
        mri_copy.metadata.image.dimensions = list(new_tensor.shape)
        
        mri_copy.statistics = ScanStatistics(
            min=float(np.min(new_tensor)),
            max=float(np.max(new_tensor)),
            mean=float(np.mean(new_tensor)),
            std=float(np.std(new_tensor)),
            shape=list(new_tensor.shape),
            dtype=str(new_tensor.dtype)
        )
        
        return mri_copy
