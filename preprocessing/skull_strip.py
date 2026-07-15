"""MRI skull stripping implementing the strategy pattern with threshold, morphological, and deep learning wrappers."""

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


class MorphologicalSkullStripperStrategy(BaseSkullStripperStrategy):
    """Morphological brain extraction using double Otsu connected components and structuring sweeps."""

    def execute(self, mri_data: MRIData, context: ExecutionContext, params: Dict[str, Any]) -> np.ndarray:
        context.logger.info("SkullStripper [MorphologicalStrategy]: Segmenting brain parenchyma.")
        sitk_img = mri_data_to_sitk(mri_data)
        
        # Step 1: Head mask (Otsu threshold on the whole scan to filter empty air)
        head_otsu = sitk.OtsuThresholdImageFilter()
        head_otsu.SetInsideValue(0)
        head_otsu.SetOutsideValue(1)
        head_mask = head_otsu.Execute(sitk_img)
        
        # Step 2: Tissue mask (Otsu threshold only on the head tissue)
        head_img = sitk.Mask(sitk_img, head_mask)
        brain_otsu = sitk.OtsuThresholdImageFilter()
        brain_otsu.SetInsideValue(0)
        brain_otsu.SetOutsideValue(1)
        tissue_mask = brain_otsu.Execute(head_img)
        
        # Get voxel physical spacing and volume mapping in milliliters
        spacing = sitk_img.GetSpacing()
        voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
        
        mask_img = None
        
        # Step 3: Sweep structuring element radius to find isolated component satisfying brain volume
        for radius in [3, 2, 1]:
            eroded_mask = sitk.BinaryErode(tissue_mask, [radius, radius, radius], sitk.sitkBall)
            label_img = sitk.ConnectedComponent(eroded_mask)
            label_shape = sitk.LabelShapeStatisticsImageFilter()
            label_shape.Execute(label_img)
            
            num_labels = label_shape.GetNumberOfLabels()
            if num_labels > 0:
                largest_label = max(range(1, num_labels + 1), key=lambda l: label_shape.GetNumberOfPixels(l))
                pixel_count = label_shape.GetNumberOfPixels(largest_label)
                volume_ml = pixel_count * voxel_vol_ml
                
                # Check volume (350ml minimum boundary for pediatric/normal brains)
                if volume_ml >= 350.0:
                    context.logger.info(
                        f"MorphologicalStrategy: Brain isolated with radius {radius}. "
                        f"Volume: {volume_ml:.1f}ml ({pixel_count} voxels)."
                    )
                    brain_eroded_mask = sitk.Equal(label_img, largest_label)
                    
                    # Step 4: Restore boundaries and close holes
                    dilated = sitk.BinaryDilate(brain_eroded_mask, [radius, radius, radius], sitk.sitkBall)
                    closed = sitk.BinaryMorphologicalClosing(dilated, [5, 5, 5])
                    mask_img = sitk.BinaryFillhole(closed)
                    break
                else:
                    context.logger.warning(
                        f"MorphologicalStrategy: Component at radius {radius} was too small "
                        f"({volume_ml:.1f}ml < 350.0ml). Decrementing radius."
                    )
                    
        # Step 5: Fallback if no components satisfy volume constraints
        if mask_img is None:
            context.logger.warning(
                "MorphologicalStrategy: No morphological component met the 350ml brain volume threshold. "
                "Falling back to Otsu head mask."
            )
            mask_img = head_mask
            
        # Convert SimpleITK image to NumPy array
        mask_arr = np.transpose(sitk.GetArrayFromImage(mask_img), (2, 1, 0)).astype(np.float32)
        
        # Ensure dimensions match image
        active_tensor = mri_data.image if mri_data.image is not None else mri_data.raw.tensor
        if len(active_tensor.shape) == 4 and len(mask_arr.shape) == 3:
            mask_arr = np.expand_dims(mask_arr, axis=0)
            
        return mask_arr


class HDBetSkullStripperStrategy(BaseSkullStripperStrategy):
    """Deep learning-based HD-BET brain extractor wrapper with morphological fallback."""

    def execute(self, mri_data: MRIData, context: ExecutionContext, params: Dict[str, Any]) -> np.ndarray:
        import subprocess
        import shutil
        import tempfile
        from pathlib import Path
        
        hd_bet_installed = False
        try:
            import HD_BET
            hd_bet_installed = True
        except ImportError:
            pass
            
        hd_bet_cli = shutil.which("hd-bet") is not None
        
        if not hd_bet_installed and not hd_bet_cli:
            context.logger.warning(
                "SkullStripper [HDBetStrategy]: HD-BET dependencies or CLI binary not found. "
                "Gracefully falling back to MorphologicalStrategy."
            )
            return MorphologicalSkullStripperStrategy().execute(mri_data, context, params)
            
        context.logger.info("SkullStripper [HDBetStrategy]: Running HD-BET brain extraction.")
        
        temp_dir = tempfile.mkdtemp()
        try:
            in_file = Path(temp_dir) / "input.nii.gz"
            out_file = Path(temp_dir) / "output.nii.gz"
            mask_file = Path(temp_dir) / "output_mask.nii.gz"
            
            sitk_img = mri_data_to_sitk(mri_data)
            sitk.WriteImage(sitk_img, str(in_file))
            
            # Run via PyTorch module if installed, else run shell CLI subprocess
            if hd_bet_installed:
                from HD_BET.run_hd_bet import run_hd_bet
                run_hd_bet(str(in_file), str(out_file), keep_mask=True, overwrite_existing=True)
            else:
                subprocess.run(["hd-bet", "-i", str(in_file), "-o", str(out_file)], check=True)
                
            if mask_file.exists():
                mask_img = sitk.ReadImage(str(mask_file))
            elif out_file.exists():
                out_img = sitk.ReadImage(str(out_file))
                mask_img = sitk.BinaryThreshold(out_img, 1e-5, 1e10, 1, 0)
            else:
                raise FileNotFoundError("HD-BET output files not generated.")
                
            mask_arr = np.transpose(sitk.GetArrayFromImage(mask_img), (2, 1, 0)).astype(np.float32)
            
            active_tensor = mri_data.image if mri_data.image is not None else mri_data.raw.tensor
            if len(active_tensor.shape) == 4 and len(mask_arr.shape) == 3:
                mask_arr = np.expand_dims(mask_arr, axis=0)
                
            return mask_arr
            
        except Exception as e:
            context.logger.error(f"HDBetStrategy failed with error: {e}. Falling back to MorphologicalStrategy.")
            return MorphologicalSkullStripperStrategy().execute(mri_data, context, params)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class SynthStripSkullStripperStrategy(BaseSkullStripperStrategy):
    """Deep learning-based FreeSurfer SynthStrip wrapper with morphological fallback."""

    def execute(self, mri_data: MRIData, context: ExecutionContext, params: Dict[str, Any]) -> np.ndarray:
        import subprocess
        import shutil
        import tempfile
        from pathlib import Path
        
        synthstrip_cli = shutil.which("mri_synthstrip") is not None
        
        if not synthstrip_cli:
            context.logger.warning(
                "SkullStripper [SynthStripStrategy]: mri_synthstrip CLI command not found. "
                "Gracefully falling back to MorphologicalStrategy."
            )
            return MorphologicalSkullStripperStrategy().execute(mri_data, context, params)
            
        context.logger.info("SkullStripper [SynthStripStrategy]: Running SynthStrip brain extraction.")
        
        temp_dir = tempfile.mkdtemp()
        try:
            in_file = Path(temp_dir) / "input.nii.gz"
            out_file = Path(temp_dir) / "output.nii.gz"
            mask_file = Path(temp_dir) / "output_mask.nii.gz"
            
            sitk_img = mri_data_to_sitk(mri_data)
            sitk.WriteImage(sitk_img, str(in_file))
            
            # Execute FreeSurfer SynthStrip CLI
            subprocess.run([
                "mri_synthstrip",
                "-i", str(in_file),
                "-o", str(out_file),
                "-m", str(mask_file)
            ], check=True)
            
            if mask_file.exists():
                mask_img = sitk.ReadImage(str(mask_file))
            elif out_file.exists():
                out_img = sitk.ReadImage(str(out_file))
                mask_img = sitk.BinaryThreshold(out_img, 1e-5, 1e10, 1, 0)
            else:
                raise FileNotFoundError("SynthStrip output files not generated.")
                
            mask_arr = np.transpose(sitk.GetArrayFromImage(mask_img), (2, 1, 0)).astype(np.float32)
            
            active_tensor = mri_data.image if mri_data.image is not None else mri_data.raw.tensor
            if len(active_tensor.shape) == 4 and len(mask_arr.shape) == 3:
                mask_arr = np.expand_dims(mask_arr, axis=0)
                
            return mask_arr
            
        except Exception as e:
            context.logger.error(f"SynthStripStrategy failed with error: {e}. Falling back to MorphologicalStrategy.")
            return MorphologicalSkullStripperStrategy().execute(mri_data, context, params)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class FastSurferSkullStripperStrategy(BaseSkullStripperStrategy):
    """Placeholder strategy for DL-based FastSurfer segmentation strip with morphological fallback."""

    def execute(self, mri_data: MRIData, context: ExecutionContext, params: Dict[str, Any]) -> np.ndarray:
        context.logger.warning(
            "SkullStripper [FastSurferStrategy]: Deep Learning FastSurfer is not implemented. "
            "Gracefully falling back to MorphologicalStrategy."
        )
        return MorphologicalSkullStripperStrategy().execute(mri_data, context, params)


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
        elif strategy_name == "morphological":
            strategy = MorphologicalSkullStripperStrategy()
        elif strategy_name == "hdbet":
            strategy = HDBetSkullStripperStrategy()
        elif strategy_name == "synthstrip":
            strategy = SynthStripSkullStripperStrategy()
        elif strategy_name == "fastsurfer":
            strategy = FastSurferSkullStripperStrategy()
        else:
            context.logger.warning(
                f"SkullStripper: Unknown strategy '{strategy_name}'. Falling back to MorphologicalStrategy."
            )
            strategy = MorphologicalSkullStripperStrategy()
            
        # Execute selected strategy to extract binary mask
        mask = strategy.execute(mri_data, context, self.params)
        
        # Mask active image
        tensor = mri_data.image if mri_data.image is not None else mri_data.raw.tensor
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
