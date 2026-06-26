"""Pydantic schemas representing pipeline step execution logs and lifecycle dependency context."""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ProcessingRecord(BaseModel):
    """Structured step output parameters tracking pipeline provenance."""

    step_name: str = Field(..., description="Name of the transform class executed.")
    duration: float = Field(..., description="Voxel array calculation duration in seconds.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Dictionary of execution parameters passed to transform.")
    warnings: List[str] = Field(default_factory=list, description="Warnings captured during computation.")
    memory_mb: float = Field(default=0.0, description="Approximate peak memory variation in MB.")
    input_hash: str = Field(..., description="Cryptographic SHA256 signature of input volume state.")
    output_hash: str = Field(..., description="Cryptographic SHA256 signature of output volume state.")


class ExecutionContext(BaseModel):
    """Lifecycle tracking parameters passed across transform steps."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    logger: logging.Logger = Field(..., description="Logger instance routing trace messages.")
    cache: Any = Field(default=None, description="Pipeline Cache manager instance.")
    config: Dict[str, Any] = Field(default_factory=dict, description="Global configuration dictionary parameters.")
    seed: int = Field(default=42, description="Random seed parameter for reproducibility.")
    device: str = Field(default="cpu", description="Computation device target (cpu vs cuda).")
