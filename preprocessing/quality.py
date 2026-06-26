"""MRI quality metrics assessment."""

from schemas.mri import MRIData
from schemas.quality import QualityReport


class ImageQualityAssessor:
    """Estimates metrics (e.g. signal-to-noise ratio) for quality checks."""

    def assess_quality(self, mri_data: MRIData) -> QualityReport:
        # Skeleton implementation
        return QualityReport(
            snr=15.2,
            cnr=8.4,
            has_artifacts=False,
            resolution_valid=True,
            overall_score=0.92
        )
