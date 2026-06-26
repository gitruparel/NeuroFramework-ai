"""Unit test suite for the Stage 3A Preprocessing Framework."""

import logging
from pathlib import Path
from typing import Any
import numpy as np
import pytest
from core.exceptions import ConfigurationError
from preprocessing.base import BaseTransform, TransformRegistry
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.decorators import trace_transform
from schemas.mri import MRIData, RawMRI
from schemas.processing import ExecutionContext, ProcessingRecord
from schemas.metadata import MRIMetadata, ImageMetadata, PatientMetadata
from schemas.quality import QualityReport
from schemas.validation import ValidationReport, FileValidationReport, MRIValidationReport


# 1. Create Mock Transforms for framework verification
@TransformRegistry.register("mock_bias")
class MockBiasCorrection(BaseTransform):
    """Mock transform that changes intensities slightly."""

    @trace_transform
    def process(self, mri_data: MRIData, context: ExecutionContext) -> MRIData:
        mri_copy = mri_data.model_copy(deep=True)
        mri_copy.image = mri_copy.image * 0.95
        return mri_copy


@TransformRegistry.register("mock_crop")
class MockCropForeground(BaseTransform):
    """Mock transform that reduces array size."""

    @trace_transform
    def process(self, mri_data: MRIData, context: ExecutionContext) -> MRIData:
        mri_copy = mri_data.model_copy(deep=True)
        mri_copy.image = mri_copy.image[..., 1:-1]
        return mri_copy


# Simple in-memory mock cache for testing
class MockPipelineCache:
    def __init__(self):
        self._store = {}
        self.hits = 0
        self.sets = 0

    def get(self, key: str):
        if key in self._store:
            self.hits += 1
            return self._store[key]
        return None

    def set(self, key: str, value: Any):
        self.sets += 1
        self._store[key] = value


@pytest.fixture
def mock_mri_data() -> MRIData:
    """Fixture returning a mock valid MRIData instance."""
    raw = RawMRI(
        tensor=np.random.randint(10, 100, size=(10, 10, 10), dtype=np.int16),
        affine=np.eye(4),
        header={},
        format="nifti",
        source_path=Path("mock_volume.nii")
    )
    
    metadata = MRIMetadata(
        image=ImageMetadata(voxel_dims=[1.0, 1.0, 1.0], dimensions=[10, 10, 10], modality="T1w"),
        patient=PatientMetadata(patient_id="mock_patient")
    )
    
    quality = QualityReport(
        noise_score=10.0, blur_score=100.0, motion_score=0.1, contrast_score=0.5,
        dynamic_range=90.0, resolution=1.0, slice_count=10, overall_score=0.8
    )
    
    validation = ValidationReport(
        file_validation=FileValidationReport(exists=True, readable=True, header_valid=True, corrupt=False),
        mri_validation=MRIValidationReport(
            voxel_spacing_valid=True, dimensions_valid=True, intensity_valid=True,
            orientation_valid=True, metadata_complete=True, empty_slices_detected=False
        ),
        is_valid=True
    )

    return MRIData(
        raw=raw,
        metadata=metadata,
        quality=quality,
        validation=validation,
        history=[],
        preview=[],
        statistics={"min": 0, "max": 100, "mean": 50, "std": 10, "shape": [10, 10, 10], "dtype": "int16"}
    )


@pytest.fixture
def execution_context() -> ExecutionContext:
    """Fixture returning execution parameters context."""
    logger = logging.getLogger("test_preprocessing")
    return ExecutionContext(
        logger=logger,
        cache=None,
        config={},
        seed=42,
        device="cpu"
    )


def test_pipeline_config_compilation():
    """Verify pipelines dynamically load profiles defined in configurations."""
    yaml_content = """
profiles:
  autism:
    - transform: mock_bias
      params: { strategy: "n4" }
    - transform: mock_crop
      params: {}
  invalid:
    - params: {} # missing transform key
"""
    # Write temporary YAML file
    import tempfile
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
        f.write(yaml_content)
        temp_yaml_path = f.name

    try:
        # Load valid profile
        pipeline = PreprocessingPipeline.from_yaml(temp_yaml_path, "autism")
        assert len(pipeline.steps) == 2
        assert isinstance(pipeline.steps[0], MockBiasCorrection)
        assert isinstance(pipeline.steps[1], MockCropForeground)
        
        # Load non-existent profile
        with pytest.raises(ConfigurationError):
            PreprocessingPipeline.from_yaml(temp_yaml_path, "alzheimer")
            
        # Load corrupted profile
        with pytest.raises(ConfigurationError):
            PreprocessingPipeline.from_yaml(temp_yaml_path, "invalid")
    finally:
        Path(temp_yaml_path).unlink()


def test_pipeline_graph_visualization():
    """Verify visualizer outputs correct flow representation."""
    steps = [
        MockBiasCorrection(strategy="n4"),
        MockCropForeground()
    ]
    pipeline = PreprocessingPipeline(steps)
    vis = pipeline.visualize()
    
    assert "Load -> MockBiasCorrection -> MockCropForeground -> Standardize -> MRIData" in vis


def test_decorator_tracing_provenance(mock_mri_data, execution_context):
    """Verify trace_transform captures durations, shapes, hashes, and appends records."""
    transform = MockBiasCorrection(strategy="n4")
    
    input_shape = mock_mri_data.raw.tensor.shape
    res = transform.process(mock_mri_data, execution_context)
    
    # Execution history should contain a record
    assert len(res.history) == 1
    record = res.history[0]
    
    assert isinstance(record, ProcessingRecord)
    assert record.step_name == "MockBiasCorrection"
    assert record.duration > 0.0
    assert record.params == {"strategy": "n4"}
    assert record.input_hash != ""
    assert record.output_hash != ""
    assert record.input_hash != record.output_hash  # Hash changed since tensor changed


def test_pipeline_execution_hooks(mock_mri_data, execution_context):
    """Verify lifecycle pre- and post-hooks run in sequence."""
    steps = [MockBiasCorrection()]
    pipeline = PreprocessingPipeline(steps)

    pre_run = False
    post_run = False

    def pre_hook(data: MRIData, ctx: ExecutionContext) -> MRIData:
        nonlocal pre_run
        pre_run = True
        return data

    def post_hook(data: MRIData, ctx: ExecutionContext) -> MRIData:
        nonlocal post_run
        post_run = True
        return data

    pipeline.register_pre_hook(pre_hook)
    pipeline.register_post_hook(post_hook)

    res = pipeline.process(mock_mri_data, execution_context)
    
    assert pre_run is True
    assert post_run is True


def test_pipeline_datatype_standardization(mock_mri_data, execution_context):
    """Verify output voxel tensors standardize to float32 and C x X x Y x Z."""
    steps = [MockBiasCorrection()]
    pipeline = PreprocessingPipeline(steps)
    
    # Input has shape (10, 10, 10) and type int16
    assert len(mock_mri_data.raw.tensor.shape) == 3
    assert mock_mri_data.raw.tensor.dtype == np.int16

    res = pipeline.process(mock_mri_data, execution_context)
    
    # Output should have shape (1, 10, 10, 10) and float32
    assert len(res.image.shape) == 4
    assert res.image.shape[0] == 1
    assert res.image.dtype == np.float32
    # Raw tensor remains immutable
    assert len(res.raw.tensor.shape) == 3
    assert res.raw.tensor.dtype == np.int16


def test_pipeline_intermediate_caching(mock_mri_data, execution_context):
    """Verify intermediate step caching blocks redundant step executions."""
    steps = [MockBiasCorrection()]
    pipeline = PreprocessingPipeline(steps)
    
    cache = MockPipelineCache()
    execution_context.cache = cache

    import copy
    data1 = copy.deepcopy(mock_mri_data)
    data2 = copy.deepcopy(mock_mri_data)

    # 1. Run first time (cache miss, executes transform, stores in cache)
    res1 = pipeline.process(data1, execution_context)
    assert cache.sets == 1
    assert cache.hits == 0

    # 2. Run second time (cache hit, loads from cache, bypasses transform execution)
    res2 = pipeline.process(data2, execution_context)
    assert cache.sets == 1
    assert cache.hits == 1
    assert len(res2.history) == 1  # Verify history still exists
