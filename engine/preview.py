"""Preview images generation for MRI volumes."""

from typing import List
import numpy as np
from schemas.mri import PreviewImage, RawMRI


class PreviewGenerator:
    """Generates orthogonal 2D slice previews (axial, coronal, sagittal) from 3D MRI scans."""

    def generate(self, raw: RawMRI) -> List[PreviewImage]:
        tensor = raw.tensor
        shape = tensor.shape
        previews = []

        if raw.format == "image":
            # For 2D images, return the original grayscale slice preview
            # Tensor is shaped (x, y, 1) or similar.
            slice_data = tensor[:, :, 0]
            scaled = self._scale_to_grayscale(slice_data)
            previews.append(
                PreviewImage(
                    image=scaled,
                    plane="2D Image",
                    slice_idx=0,
                    title="Original 2D Image View",
                )
            )
            return previews

        # Check if 3D
        if len(shape) == 3 and shape[2] > 1:
            # 1. Axial Plane (xy slice, looking down Z)
            mid_z = shape[2] // 2
            axial_slice = tensor[:, :, mid_z]
            previews.append(
                PreviewImage(
                    image=self._scale_to_grayscale(axial_slice),
                    plane="axial",
                    slice_idx=mid_z,
                    title=f"Axial Orthogonal Slice (Z={mid_z})",
                )
            )

            # 2. Coronal Plane (xz slice, looking down Y)
            mid_y = shape[1] // 2
            coronal_slice = tensor[:, mid_y, :]
            previews.append(
                PreviewImage(
                    image=self._scale_to_grayscale(coronal_slice),
                    plane="coronal",
                    slice_idx=mid_y,
                    title=f"Coronal Orthogonal Slice (Y={mid_y})",
                )
            )

            # 3. Sagittal Plane (yz slice, looking down X)
            mid_x = shape[0] // 2
            sagittal_slice = tensor[mid_x, :, :]
            previews.append(
                PreviewImage(
                    image=self._scale_to_grayscale(sagittal_slice),
                    plane="sagittal",
                    slice_idx=mid_x,
                    title=f"Sagittal Orthogonal Slice (X={mid_x})",
                )
            )

        return previews

    def _scale_to_grayscale(self, slice_data: np.ndarray) -> np.ndarray:
        """Min-max scales intensities to [0, 255] and converts to uint8."""
        min_val = np.min(slice_data)
        max_val = np.max(slice_data)
        
        if max_val == min_val:
            return np.zeros_like(slice_data, dtype=np.uint8)
        
        scaled = (slice_data - min_val) / (max_val - min_val) * 255.0
        return np.clip(scaled, 0, 255).astype(np.uint8)
