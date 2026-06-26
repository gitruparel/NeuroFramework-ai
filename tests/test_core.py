"""Unit tests for core configuration and interfaces."""

import pytest
from core.config import settings
from core.exceptions import BrainAIError, ConfigurationError
from schemas.metadata import MRIMetadata, ImageMetadata, PatientMetadata


def test_core_settings():
    """Verify core settings load defaults correctly."""
    assert settings.log_level in ["INFO", "DEBUG", "WARNING", "ERROR"]
    assert settings.device in ["cpu", "cuda"]


def test_custom_exceptions():
    """Verify custom exception hierarchy is functioning as expected."""
    with pytest.raises(BrainAIError):
        raise ConfigurationError("Invalid environment specification")


def test_mri_metadata_schema():
    """Verify that MRIMetadata Pydantic schema parses attributes correctly."""
    meta = MRIMetadata(
        image=ImageMetadata(
            voxel_dims=[1.0, 1.0, 1.0],
            dimensions=[256, 256, 176],
            modality="T1w"
        ),
        patient=PatientMetadata(
            patient_id="test_patient_001"
        )
    )
    assert meta.patient.patient_id == "test_patient_001"
    assert meta.image.dimensions[0] == 256
