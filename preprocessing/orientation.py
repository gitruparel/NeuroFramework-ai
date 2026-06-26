"""MRI orientation standardization mapping coordinates to target axcodes (e.g. RAS)."""

import numpy as np
import nibabel as nib
from typing import Any
from preprocessing.base import BaseTransform, TransformRegistry
from preprocessing.decorators import trace_transform
from schemas.mri import MRIData, ScanStatistics
from schemas.processing import ExecutionContext


@TransformRegistry.register("reorient")
class Reorient(BaseTransform):
    """Reorients volume coordinate layout to a target system configuration (default RAS)."""

    def __init__(self, target: str = "RAS", **kwargs: Any):
        super().__init__(target=target, **kwargs)

    @trace_transform
    def process(self, mri_data: MRIData, context: ExecutionContext) -> MRIData:
        tensor = mri_data.image
        target_axcodes = self.params.get("target", "RAS")
        
        # Spatial dimensions count
        spatial_shape = tensor.shape[1:] if len(tensor.shape) == 4 else tensor.shape
        
        # Resolve original axcodes and transform orientation mapping
        orig_ornt = nib.orientations.io_orientation(mri_data.affine)
        orig_axcodes = "".join(nib.orientations.ornt2axcodes(orig_ornt))
        
        targ_ornt = nib.orientations.axcodes2ornt(list(target_axcodes))
        transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
        
        if np.array_equal(orig_ornt, targ_ornt):
            context.logger.info(f"Reorient: Coordinate space is already aligned to '{target_axcodes}'. Skipping.")
            return mri_data.model_copy(deep=True)
            
        # Re-index coordinate dimensions and flip directions
        if len(tensor.shape) == 4:
            new_channels = []
            for c in range(tensor.shape[0]):
                new_channels.append(nib.orientations.apply_orientation(tensor[c], transform))
            new_tensor = np.stack(new_channels, axis=0)
        else:
            new_tensor = nib.orientations.apply_orientation(tensor, transform)
            
        # Recalculate affine mapping offsets
        new_affine = mri_data.affine @ nib.orientations.inv_ornt_aff(transform, spatial_shape)
        
        # Save provenance details for inverse transform playback
        self.params["original_orientation"] = orig_axcodes
        self.params["target"] = target_axcodes
        
        mri_copy = mri_data.model_copy(deep=True)
        mri_copy.image = new_tensor
        mri_copy.affine = new_affine
        mri_copy.metadata.image.orientation = target_axcodes
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
