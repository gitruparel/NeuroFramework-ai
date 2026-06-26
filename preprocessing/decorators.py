"""Trace decorators logging step duration, memory variation, input/output signatures, and warnings."""

import functools
import hashlib
import time
import warnings
import gc
from typing import Any, Callable
from schemas.mri import MRIData
from schemas.processing import ProcessingRecord, ExecutionContext


def compute_mri_state_hash(mri_data: MRIData) -> str:
    """Computes quick SHA256 signature of the current image state (tensor, spacing, and affine)."""
    sha256 = hashlib.sha256()
    
    tensor = mri_data.image if mri_data.image is not None else mri_data.raw.tensor
    affine = mri_data.affine if mri_data.affine is not None else mri_data.raw.affine
    
    # 1. Mix statistics and spatial layout
    shape_str = str(list(tensor.shape))
    spacing_str = str(list(mri_data.metadata.image.voxel_dims))
    affine_str = str(list(affine.flatten()))
    
    sha256.update(f"{shape_str}_{spacing_str}_{affine_str}".encode("utf-8"))
    
    # 2. Mix voxel data samples/variance
    mean_val = float(np_mean_safe(tensor))
    std_val = float(np_std_safe(tensor))
    sha256.update(f"{mean_val:.6f}_{std_val:.6f}".encode("utf-8"))
    
    # 3. Mix brain mask if present
    if mri_data.brain_mask is not None:
        mask_mean = float(np_mean_safe(mri_data.brain_mask))
        sha256.update(f"mask_{mask_mean:.6f}".encode("utf-8"))

    return sha256.hexdigest()


def np_mean_safe(arr) -> float:
    """Computes mean safely without importing numpy globally if not loaded."""
    import numpy as np
    return float(np.mean(arr))


def np_std_safe(arr) -> float:
    """Computes std safely without importing numpy globally if not loaded."""
    import numpy as np
    return float(np.std(arr))


def trace_transform(func: Callable) -> Callable:
    """Decorator capturing performance telemetry and appending ProcessingRecord to history."""
    @functools.wraps(func)
    def wrapper(self: Any, mri_data: MRIData, context: ExecutionContext) -> MRIData:
        step_name = self.__class__.__name__
        context.logger.info(f"Preprocessing Step: Starting '{step_name}'")
        
        # Initialize processed image and affine fields to raw values if None
        if mri_data.image is None or mri_data.affine is None:
            mri_data = mri_data.model_copy(deep=True)
            if mri_data.image is None:
                mri_data.image = mri_data.raw.tensor.copy()
            if mri_data.affine is None:
                mri_data.affine = mri_data.raw.affine.copy()

        # Calculate input hash signature
        input_hash = compute_mri_state_hash(mri_data)
        
        # Force garbage collection to measure memory accurately
        gc.collect()
        import os
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            mem_before = 0.0

        # Execute transform inside warning capture
        start_time = time.perf_counter()
        warnings_caught = []
        
        try:
            with warnings.catch_warnings(record=True) as w_list:
                warnings.simplefilter("always")
                result_data = func(self, mri_data, context)
                
                # Capture warning messages
                if w_list:
                    for warning_item in w_list:
                        warnings_caught.append(str(warning_item.message))
        except Exception as e:
            duration = time.perf_counter() - start_time
            context.logger.error(
                f"Preprocessing Step: Failed '{step_name}' after {duration:.4f}s. Error: {str(e)}"
            )
            raise e

        duration = time.perf_counter() - start_time
        
        # Measure final memory usage
        gc.collect()
        try:
            mem_after = process.memory_info().rss / (1024 * 1024)
            memory_mb = max(mem_after - mem_before, 0.0)
        except NameError:
            memory_mb = 0.0

        # Calculate output state signature
        output_hash = compute_mri_state_hash(result_data)

        # Build processing log record
        record = ProcessingRecord(
            step_name=step_name,
            duration=duration,
            params={k: str(v) for k, v in self.params.items()},
            warnings=warnings_caught,
            memory_mb=memory_mb,
            input_hash=input_hash,
            output_hash=output_hash,
        )
        
        # Append record to execution history list
        result_data.history.append(record)
        context.logger.info(f"Preprocessing Step: Finished '{step_name}' in {duration:.4f}s")
        
        return result_data

    return wrapper
