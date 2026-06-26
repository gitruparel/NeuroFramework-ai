"""Orchestrator engine coordinating the ingestion, validation, and metadata extraction pipeline."""

from pathlib import Path
from typing import Any
import numpy as np
from core.exceptions import BrainAIError, MRIProcessingError
from engine.decorators import log_pipeline_step
from engine.readers.factory import ReaderFactory
from engine.metadata import MetadataExtractor
from engine.validator import ValidationEngine
from engine.preview import PreviewGenerator
from engine.quality import QualityAnalyzer
from engine.cache import MRICache
from schemas.mri import MRIData, ScanStatistics


class MRIEngine:
    """Universal MRI loading public interface orchestrating processing, stats, previews, and quality checks."""

    def __init__(self):
        self.metadata_extractor = MetadataExtractor()
        self.validation_engine = ValidationEngine()
        self.preview_generator = PreviewGenerator()
        self.quality_analyzer = QualityAnalyzer()
        self.cache = MRICache()

    def load(self, path: str | Path) -> MRIData:
        """Loads and processes any supported MRI image file (NIfTI, DICOM, JPG, PNG)."""
        p = Path(path)
        
        # 1. Cache check
        hash_key = self.cache.compute_hash(p)
        cached_data = self.cache.get(hash_key)
        if cached_data is not None:
            return cached_data

        # 2. Execution History
        history = []

        # 3. Read RawMRI
        raw = self._run_reader(p, history)

        # 4. Extract metadata
        metadata = self._run_metadata(raw, history)

        # 5. Calculate statistics
        stats = self._run_statistics(raw, history)

        # 6. Validate scan
        validation = self._run_validation(p, raw, metadata, history)

        # 7. Generate previews
        previews = self._run_previews(raw, history)

        # 8. Analyze quality
        quality = self._run_quality(raw, metadata, history)

        # Pack into MRIData
        mri_data = MRIData(
            raw=raw,
            metadata=metadata,
            quality=quality,
            validation=validation,
            history=history,
            preview=previews,
            statistics=stats,
        )

        # 9. Store in cache
        self.cache.set(hash_key, mri_data)

        return mri_data

    @log_pipeline_step("Volume Reading")
    def _run_reader(self, path: Path, history: list[str]) -> Any:
        reader = ReaderFactory.create(path)
        return reader.read(path)

    @log_pipeline_step("Metadata Extraction")
    def _run_metadata(self, raw: Any, history: list[str]) -> Any:
        return self.metadata_extractor.extract(raw)

    @log_pipeline_step("Statistics Calculation")
    def _run_statistics(self, raw: Any, history: list[str]) -> ScanStatistics:
        tensor = raw.tensor
        return ScanStatistics(
            min=float(np.min(tensor)),
            max=float(np.max(tensor)),
            mean=float(np.mean(tensor)),
            std=float(np.std(tensor)),
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
        )

    @log_pipeline_step("Pipeline Validation")
    def _run_validation(self, path: Path, raw: Any, metadata: Any, history: list[str]) -> Any:
        return self.validation_engine.run_all(path, raw, metadata)

    @log_pipeline_step("Previews Generation")
    def _run_previews(self, raw: Any, history: list[str]) -> Any:
        return self.preview_generator.generate(raw)

    @log_pipeline_step("Quality Assessment")
    def _run_quality(self, raw: Any, metadata: Any, history: list[str]) -> Any:
        return self.quality_analyzer.analyze(raw, metadata)
