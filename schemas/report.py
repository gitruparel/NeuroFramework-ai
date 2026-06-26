"""Schemas representing compiled analysis reports."""

from pathlib import Path
from pydantic import BaseModel, Field


class Report(BaseModel):
    """Schema representing structured metrics of generated PDF analysis report."""

    report_id: str = Field(..., description="Unique generated report identifier.")
    patient_id: str = Field(..., description="Target patient identifier.")
    pdf_path: Path = Field(..., description="Absolute path location pointing to completed PDF artifact.")
    summary: str = Field(..., description="Compiled textual summary of MRI metrics and prediction results.")
    created_at: str = Field(..., description="Creation ISO-formatted timestamp.")
