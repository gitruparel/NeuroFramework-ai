"""Pydantic schemas representing dataset properties and standard sample tensors."""

from typing import Any, Dict, List, Optional
import torch
from pydantic import BaseModel, ConfigDict, Field


class DatasetInfo(BaseModel):
    """Dataset-level metadata and documentation parameters."""

    name: str = Field(..., description="Dataset identification name (e.g. ABIDE, ADNI, BraTS).")
    version: str = Field(default="1.0.0", description="Dataset release version.")
    modality: str = Field(default="T1w", description="Imaging modality details (e.g. T1w, T2w, Multimodal).")
    num_subjects: int = Field(default=0, description="Total unique subjects/patients count.")
    labels: List[str] = Field(default_factory=list, description="List of target diagnosis classes present.")
    license: Optional[str] = Field(default=None, description="Dataset legal license terms.")
    citation: Optional[str] = Field(default=None, description="Dataset citation details.")


class DatasetSample(BaseModel):
    """Unified container wrapping loaded scan tensors, labels, and metadata for model training."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: torch.Tensor = Field(..., description="Input scan intensity float tensor (C, X, Y, Z) or (C, H, W).")
    label: Optional[int] = Field(default=None, description="Classification integer target label.")
    mask: Optional[torch.Tensor] = Field(default=None, description="Segmentation target mask tensor (C, X, Y, Z).")
    subject_id: str = Field(..., description="Source patient identifier.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Scan metadata parameters dict.")
    dataset_name: str = Field(..., description="Name of the source dataset of origin.")


def collate_dataset_samples(batch: List[Any]) -> Dict[str, Any]:
    """Custom collate function that handles DatasetSample objects by unpacking them to dicts."""
    if not isinstance(batch[0], DatasetSample):
        return torch.utils.data.default_collate(batch)

    images = torch.stack([sample.image for sample in batch], dim=0)
    
    labels = []
    for sample in batch:
        if sample.label is not None:
            labels.append(sample.label)
        else:
            labels.append(-1)
    labels = torch.tensor(labels, dtype=torch.long)
    
    collated = {
        "image": images,
        "label": labels,
        "subject_id": [sample.subject_id for sample in batch],
        "metadata": [sample.metadata for sample in batch],
        "dataset_name": [sample.dataset_name for sample in batch]
    }
    
    if any(sample.mask is not None for sample in batch):
        masks = []
        for sample in batch:
            if sample.mask is not None:
                masks.append(sample.mask)
            else:
                masks.append(torch.zeros_like(sample.image))
        collated["mask"] = torch.stack(masks, dim=0)
        
    return collated
