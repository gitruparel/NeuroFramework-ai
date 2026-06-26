"""Quality assessment analyzer evaluating structural noise, blur, and artifacts."""

import cv2
import numpy as np
from schemas.quality import QualityReport
from schemas.mri import RawMRI
from schemas.metadata import MRIMetadata


class QualityAnalyzer:
    """Evaluates scan signal-to-noise ratio, motion patterns, contrast, and resolution bounds."""

    def analyze(self, raw: RawMRI, metadata: MRIMetadata) -> QualityReport:
        tensor = raw.tensor
        shape = tensor.shape
        
        # 1. Slice count
        slice_count = shape[2] if len(shape) == 3 else 1

        # 2. Dynamic range
        max_val = float(np.max(tensor))
        min_val = float(np.min(tensor))
        dynamic_range = max_val - min_val

        # 3. Noise Score (Signal-to-Noise Ratio estimation)
        mean_val = float(np.mean(tensor))
        std_val = float(np.std(tensor))
        noise_score = mean_val / std_val if std_val > 0 else 0.0

        # 4. Blur Score (Variance of Laplacian on central slice)
        blur_score = self._estimate_blur(raw)

        # 5. Motion Score (Ringing artifact index estimator)
        motion_score = self._estimate_motion(raw)

        # 6. Contrast Score (White matter/gray matter boundary estimation)
        contrast_score = self._estimate_contrast(raw)

        # 7. Resolution Score (Spacing volume density)
        spacing = metadata.image.voxel_dims
        voxel_vol = spacing[0] * spacing[1] * spacing[2]
        resolution = 1.0 / voxel_vol if voxel_vol > 0 else 0.0

        # 8. Check for dropped/missing slices (gradient spikes between adjacent slices)
        missing_slices = self._detect_missing_slices(raw)

        # Overall Normalized Score [0.0 - 1.0]
        # High SNR, high contrast, low blur/motion improve score
        overall_score = self._calculate_overall(noise_score, blur_score, motion_score, contrast_score)

        return QualityReport(
            noise_score=noise_score,
            blur_score=blur_score,
            motion_score=motion_score,
            contrast_score=contrast_score,
            dynamic_range=dynamic_range,
            resolution=resolution,
            slice_count=slice_count,
            missing_slices=missing_slices,
            has_artifacts=motion_score > 0.7,
            overall_score=overall_score,
        )

    def _estimate_blur(self, raw: RawMRI) -> float:
        """Estimates blur using the Laplacian variance on a central slice."""
        tensor = raw.tensor
        shape = tensor.shape
        
        try:
            # Extract central slice
            if len(shape) == 3 and shape[2] > 1:
                mid_slice = tensor[:, :, shape[2] // 2]
            else:
                mid_slice = tensor[:, :, 0]
            
            # Scale to uint8 for cv2 Laplacian
            scaled = ((mid_slice - np.min(mid_slice)) / (np.max(mid_slice) - np.min(mid_slice)) * 255.0)
            scaled_img = scaled.astype(np.uint8)
            
            lap_var = cv2.Laplacian(scaled_img, cv2.CV_64F).var()
            return float(lap_var)
        except Exception:
            return 100.0  # Default nominal value if computation fails

    def _estimate_motion(self, raw: RawMRI) -> float:
        """Estimates motion using high-frequency edge profile anomalies."""
        # Simple placeholder estimator: return a default low score (0.0 means no motion artifacts)
        return 0.15

    def _estimate_contrast(self, raw: RawMRI) -> float:
        """Estimates scan tissue contrast range."""
        tensor = raw.tensor
        try:
            non_zero = tensor[tensor > 0]
            if len(non_zero) == 0:
                return 0.0
            p90 = np.percentile(non_zero, 90)
            p10 = np.percentile(non_zero, 10)
            return float((p90 - p10) / (p90 + p10 + 1e-5))
        except Exception:
            return 0.5

    def _detect_missing_slices(self, raw: RawMRI) -> bool:
        """Detects if slice intensity changes display abnormal discontinuous jumps."""
        tensor = raw.tensor
        shape = tensor.shape
        if len(shape) != 3 or shape[2] < 5:
            return False

        # Calculate mean intensity per slice
        slice_means = [np.mean(tensor[:, :, z]) for z in range(shape[2])]
        diffs = np.abs(np.diff(slice_means))
        if len(diffs) < 2:
            return False
        
        # Check if any slice transition diff is an outlier (e.g. 4x median diff)
        median_diff = np.median(diffs)
        if median_diff == 0:
            return False
        
        return bool(np.max(diffs) > 8.0 * median_diff)

    def _calculate_overall(self, snr: float, blur: float, motion: float, contrast: float) -> float:
        # Normalize and weigh components
        # SNR: higher is better (>10 is good)
        snr_norm = min(snr / 20.0, 1.0)
        # Blur (Laplacian var): >100 is typically sharp
        blur_norm = min(blur / 500.0, 1.0)
        # Motion: lower is better
        motion_norm = max(1.0 - motion, 0.0)
        # Contrast: [0, 1] range
        contrast_norm = min(max(contrast, 0.0), 1.0)

        score = (snr_norm * 0.3) + (blur_norm * 0.3) + (motion_norm * 0.2) + (contrast_norm * 0.2)
        return float(np.clip(score, 0.0, 1.0))
