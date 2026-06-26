"""Preprocessing pipeline coordinator orchestrating dynamic profiles, step caching, and pipeline graph visualization."""

import yaml
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch
from core.exceptions import ConfigurationError
from preprocessing.base import BaseTransform, TransformRegistry
from preprocessing.decorators import compute_mri_state_hash
from schemas.mri import MRIData
from schemas.processing import ExecutionContext, ProcessingRecord


class PreprocessingPipeline:
    """Orchestrator pipeline executing registered transforms sequentially with cache checks and hooks."""

    def __init__(self, steps: List[BaseTransform]):
        self.steps = steps
        self.pre_hooks: List[Callable[[MRIData, ExecutionContext], MRIData]] = []
        self.post_hooks: List[Callable[[MRIData, ExecutionContext], MRIData]] = []

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, profile_name: str) -> "PreprocessingPipeline":
        """Compiles pipeline from specified disease profile configuration in YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessing configuration file not found: {path}")
            
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        profiles = config.get("profiles", {})
        if profile_name not in profiles:
            raise ConfigurationError(
                f"Profile '{profile_name}' is not defined. Available: {list(profiles.keys())}"
            )

        steps_configs = profiles[profile_name]
        steps = []
        
        for step_cfg in steps_configs:
            transform_name = step_cfg.get("transform")
            params = step_cfg.get("params", {})
            if not transform_name:
                raise ConfigurationError("Preprocessing configuration step is missing 'transform' key.")
            
            # Instantiate dynamically via registry
            transform = TransformRegistry.create(transform_name, **params)
            steps.append(transform)

        return cls(steps)

    def register_pre_hook(self, hook_func: Callable[[MRIData, ExecutionContext], MRIData]) -> None:
        """Registers callback executed before pipeline starts."""
        self.pre_hooks.append(hook_func)

    def register_post_hook(self, hook_func: Callable[[MRIData, ExecutionContext], MRIData]) -> None:
        """Registers callback executed after pipeline finishes."""
        self.post_hooks.append(hook_func)

    def visualize(self) -> str:
        """Generates clear ASCII workflow layout showing execution graph."""
        nodes = ["Load"]
        for step in self.steps:
            nodes.append(step.__class__.__name__)
        nodes.append("Standardize")
        nodes.append("MRIData")
        return " -> ".join(nodes)

    def process(self, mri_data: MRIData, context: ExecutionContext) -> MRIData:
        """Executes pipeline steps with caching, dtype standardization, and lifecycle hooks."""
        current = mri_data

        # Initialize processed image and affine fields to raw values if None
        if current.image is None or current.affine is None:
            current = current.model_copy(deep=True)
            if current.image is None:
                current.image = current.raw.tensor.copy()
            if current.affine is None:
                current.affine = current.raw.affine.copy()

        # 1. Execute Pre-hooks
        for hook in self.pre_hooks:
            current = hook(current, context)

        # 2. Sequential Transform Execution
        for step in self.steps:
            step_name = step.__class__.__name__
            
            # Calculate caching signatures
            input_hash = compute_mri_state_hash(current)
            cache_key = self._compute_step_cache_hash(input_hash, step_name, step.params)
            
            # Check context cache if manager is provided
            cached_result = None
            if context.cache is not None:
                cached_result = context.cache.get(cache_key)
                
            if cached_result is not None:
                context.logger.info(f"Pipeline Cache HIT for step '{step_name}'")
                current = cached_result
            else:
                # Cache miss: run transform and save output
                current = step.process(current, context)
                if context.cache is not None:
                    context.cache.set(cache_key, current)

        # 3. Data type and Dimension Standardization (Guarantee float32 & C x X x Y x Z)
        current = self._standardize_outputs(current, context)

        # 4. Execute Post-hooks
        for hook in self.post_hooks:
            current = hook(current, context)

        return current

    def _compute_step_cache_hash(self, input_hash: str, transform_name: str, params: Dict[str, Any], version: str = "1.0.0") -> str:
        """Calculates unique SHA256 caching signature combining state, class, params, and code version."""
        sha256 = hashlib.sha256()
        param_str = str(sorted(list(params.items())))
        sha256.update(f"{input_hash}_{transform_name}_{param_str}_{version}".encode("utf-8"))
        return sha256.hexdigest()

    def _standardize_outputs(self, mri_data: MRIData, context: ExecutionContext) -> MRIData:
        """Ensures voxel arrays are standard float32 tensors with layout dimensions C x X x Y x Z."""
        tensor = mri_data.image if mri_data.image is not None else mri_data.raw.tensor
        
        # Ensure array is float32
        if isinstance(tensor, np.ndarray):
            if tensor.dtype != np.float32:
                tensor = tensor.astype(np.float32)
        elif isinstance(tensor, torch.Tensor):
            if tensor.dtype != torch.float32:
                tensor = tensor.float()
        
        # Standardise dimensions to shape format layout: (C, X, Y, Z)
        if len(tensor.shape) == 3:
            # Add channel dimension if missing
            if isinstance(tensor, np.ndarray):
                tensor = np.expand_dims(tensor, axis=0)
            elif isinstance(tensor, torch.Tensor):
                tensor = tensor.unsqueeze(0)
        
        mri_data.image = tensor
        return mri_data


def reverse_preprocessing(mri_data: MRIData) -> MRIData:
    """Reverts spatial/orientation transformations of MRIData in reverse order.
    
    Uses parameters stored inside ProcessingRecord.params.
    """
    import ast
    import numpy as np
    import nibabel as nib
    import SimpleITK as sitk

    # Work on a copy of MRIData
    current = mri_data.model_copy(deep=True)
    if current.image is None:
        return current
        
    # Iterate in reverse order through the history
    for record in reversed(current.history):
        if not isinstance(record, ProcessingRecord):
            continue
            
        step_name = record.step_name
        params = record.params
        
        if step_name == "Reorient":
            orig_orient = params.get("original_orientation")
            targ_orient = params.get("target")
            if orig_orient and targ_orient:
                orig_ornt = nib.orientations.axcodes2ornt(targ_orient)
                targ_ornt = nib.orientations.axcodes2ornt(orig_orient)
                transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
                
                tensor = current.image
                if len(tensor.shape) == 4:
                    new_channels = []
                    for c in range(tensor.shape[0]):
                        new_channels.append(nib.orientations.apply_orientation(tensor[c], transform))
                    new_tensor = np.stack(new_channels, axis=0)
                else:
                    new_tensor = nib.orientations.apply_orientation(tensor, transform)
                    
                spatial_shape = tensor.shape[1:] if len(tensor.shape) == 4 else tensor.shape
                new_affine = current.affine @ nib.orientations.inv_ornt_aff(transform, spatial_shape)
                
                current.image = new_tensor
                current.affine = new_affine
                current.metadata.image.orientation = orig_orient
                current.metadata.image.dimensions = list(new_tensor.shape)
                
        elif step_name == "Resampler":
            orig_spacing_str = params.get("original_spacing")
            orig_shape_str = params.get("original_shape")
            if orig_spacing_str and orig_shape_str:
                orig_spacing = ast.literal_eval(orig_spacing_str)
                orig_shape = ast.literal_eval(orig_shape_str)
                
                # Re-create SimpleITK image from current.image spatial part
                tensor_3d = current.image[0] if len(current.image.shape) == 4 else current.image
                sitk_img = sitk.GetImageFromArray(np.transpose(tensor_3d, (2, 1, 0)))
                
                origin = [float(x) for x in current.affine[:3, 3]]
                spacing = [float(np.linalg.norm(current.affine[:3, i])) for i in range(3)]
                dir_matrix = np.zeros((3, 3))
                for i in range(3):
                    if spacing[i] > 0:
                        dir_matrix[:, i] = current.affine[:3, i] / spacing[i]
                    else:
                        dir_matrix[:, i] = current.affine[:3, i]
                        
                sitk_img.SetOrigin(origin)
                sitk_img.SetSpacing(spacing)
                sitk_img.SetDirection(dir_matrix.flatten().tolist())
                
                spatial_orig_shape = orig_shape[-3:]
                
                resample = sitk.ResampleImageFilter()
                resample.SetInterpolator(sitk.sitkLinear)
                resample.SetOutputSpacing(orig_spacing)
                resample.SetSize(spatial_orig_shape)
                resample.SetOutputDirection(sitk_img.GetDirection())
                resample.SetOutputOrigin(sitk_img.GetOrigin())
                resample.SetTransform(sitk.Transform())
                
                resampled_img = resample.Execute(sitk_img)
                
                arr = sitk.GetArrayFromImage(resampled_img)
                resampled_tensor_3d = np.transpose(arr, (2, 1, 0))
                
                if len(current.image.shape) == 4:
                    new_tensor = np.expand_dims(resampled_tensor_3d, axis=0)
                else:
                    new_tensor = resampled_tensor_3d
                    
                new_dir_matrix = np.array(resampled_img.GetDirection()).reshape(3, 3)
                new_affine = np.eye(4)
                new_affine[:3, :3] = new_dir_matrix * np.array(orig_spacing)
                new_affine[:3, 3] = resampled_img.GetOrigin()
                
                current.image = new_tensor
                current.affine = new_affine
                current.metadata.image.voxel_dims = list(orig_spacing)
                current.metadata.image.dimensions = list(new_tensor.shape)
                
        elif step_name in ("ForegroundCropper", "CenterCrop"):
            orig_shape_str = params.get("original_shape")
            crop_offsets_str = params.get("crop_offsets")
            if orig_shape_str and crop_offsets_str:
                orig_shape = ast.literal_eval(orig_shape_str)
                crop_offsets = ast.literal_eval(crop_offsets_str)
                
                new_tensor = np.zeros(orig_shape, dtype=current.image.dtype)
                
                if len(orig_shape) == 4:
                    c, x, y, z = current.image.shape
                    ox, oy, oz = crop_offsets
                    new_tensor[:, ox:ox+x, oy:oy+y, oz:oz+z] = current.image
                else:
                    x, y, z = current.image.shape
                    ox, oy, oz = crop_offsets
                    new_tensor[ox:ox+x, oy:oy+y, oz:oz+z] = current.image
                    
                new_affine = current.affine.copy()
                new_affine[:3, 3] = current.affine[:3, 3] - current.affine[:3, :3] @ crop_offsets
                
                current.image = new_tensor
                current.affine = new_affine
                current.metadata.image.dimensions = list(orig_shape)
                
        elif step_name == "Pad":
            orig_shape_str = params.get("original_shape")
            pad_offsets_str = params.get("pad_offsets")
            if orig_shape_str and pad_offsets_str:
                orig_shape = ast.literal_eval(orig_shape_str)
                pad_offsets = ast.literal_eval(pad_offsets_str)
                
                if len(orig_shape) == 4:
                    ox, oy, oz = pad_offsets
                    _, sx, sy, sz = orig_shape
                    new_tensor = current.image[:, ox:ox+sx, oy:oy+sy, oz:oz+sz]
                else:
                    ox, oy, oz = pad_offsets
                    sx, sy, sz = orig_shape
                    new_tensor = current.image[ox:ox+sx, oy:oy+sy, oz:oz+sz]
                    
                new_affine = current.affine.copy()
                new_affine[:3, 3] = current.affine[:3, 3] + current.affine[:3, :3] @ pad_offsets
                
                current.image = new_tensor
                current.affine = new_affine
                current.metadata.image.dimensions = list(orig_shape)
                
        elif step_name == "Resize":
            orig_size_str = params.get("original_size")
            orig_spacing_str = params.get("original_spacing")
            if orig_size_str and orig_spacing_str:
                orig_size = ast.literal_eval(orig_size_str)
                orig_spacing = ast.literal_eval(orig_spacing_str)
                
                # Re-create SimpleITK image from current.image spatial part
                tensor_3d = current.image[0] if len(current.image.shape) == 4 else current.image
                sitk_img = sitk.GetImageFromArray(np.transpose(tensor_3d, (2, 1, 0)))
                
                origin = [float(x) for x in current.affine[:3, 3]]
                spacing = [float(np.linalg.norm(current.affine[:3, i])) for i in range(3)]
                dir_matrix = np.zeros((3, 3))
                for i in range(3):
                    if spacing[i] > 0:
                        dir_matrix[:, i] = current.affine[:3, i] / spacing[i]
                    else:
                        dir_matrix[:, i] = current.affine[:3, i]
                        
                sitk_img.SetOrigin(origin)
                sitk_img.SetSpacing(spacing)
                sitk_img.SetDirection(dir_matrix.flatten().tolist())
                
                resample = sitk.ResampleImageFilter()
                resample.SetInterpolator(sitk.sitkLinear)
                resample.SetOutputSpacing(orig_spacing)
                resample.SetSize(orig_size)
                resample.SetOutputDirection(sitk_img.GetDirection())
                resample.SetOutputOrigin(sitk_img.GetOrigin())
                resample.SetTransform(sitk.Transform())
                
                resampled_img = resample.Execute(sitk_img)
                
                arr = sitk.GetArrayFromImage(resampled_img)
                resampled_tensor_3d = np.transpose(arr, (2, 1, 0))
                
                if len(current.image.shape) == 4:
                    new_tensor = np.expand_dims(resampled_tensor_3d, axis=0)
                else:
                    new_tensor = resampled_tensor_3d
                    
                new_dir_matrix = np.array(resampled_img.GetDirection()).reshape(3, 3)
                new_affine = np.eye(4)
                new_affine[:3, :3] = new_dir_matrix * np.array(orig_spacing)
                new_affine[:3, 3] = resampled_img.GetOrigin()
                
                current.image = new_tensor
                current.affine = new_affine
                current.metadata.image.voxel_dims = list(orig_spacing)
                current.metadata.image.dimensions = list(new_tensor.shape)
                
    return current
