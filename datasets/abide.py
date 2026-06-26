"""Custom PyTorch dataset implementation for the ABIDE (Autism) dataset."""

from pathlib import Path
from typing import Any
import torch
from datasets.base import MRIDataset, DatasetRegistry
from schemas.dataset import DatasetSample
from engine.readers.nifti import NiftiReader


def resolve_raw_path(path_str: str, raw_dir: Path | None) -> Path:
    """Resolves raw filepath relative to the new raw directory if provided."""
    path = Path(path_str)
    if raw_dir is None:
        return path
    
    parts = list(path.parts)
    # Find parts matching 'sub-*'
    sub_idx = -1
    for idx, part in enumerate(parts):
        if part.startswith("sub-"):
            sub_idx = idx
            break
            
    if sub_idx != -1:
        # Include site directory (one level above 'sub-')
        start_idx = max(0, sub_idx - 1)
        rel_path = Path(*parts[start_idx:])
        return Path(raw_dir) / rel_path
    
    # Fallback to filename
    return Path(raw_dir) / path.name


@DatasetRegistry.register("abide")
class ABIDEDataset(MRIDataset):
    """PyTorch Dataset loader for ABIDE autism scans, supporting cached preprocessed volumes."""

    def __init__(self, *args: Any, preprocessed_dir: str | Path | None = None, raw_dir: str | Path | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir is not None else None
        self.raw_dir = Path(raw_dir) if raw_dir is not None else None
        self.reader = NiftiReader()

    def __getitem__(self, index: int) -> DatasetSample:
        item = self.items[index]
        subject_id = item["subject_id"]
        path_str = item["path"]
        label_name = item["label"]
        label_idx = self.label_map.get(label_name, -1)

        # 1. Attempt loading from offline preprocessed cache folder
        if self.preprocessed_dir is not None:
            cache_path = self.preprocessed_dir / f"{subject_id}.pt"
            if cache_path.exists():
                cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
                tensor = cache_data["image"]
                
                # Convert if numpy
                import numpy as np
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor).float()
                else:
                    tensor = tensor.float()
                    
                if len(tensor.shape) == 3:
                    tensor = tensor.unsqueeze(0)

                if self.transform:
                    tensor = self.transform(tensor)

                return DatasetSample(
                    image=tensor,
                    label=label_idx,
                    subject_id=subject_id,
                    metadata=cache_data.get("metadata", {}),
                    dataset_name="abide",
                )

        # 2. Fallback to raw NIfTI volume loader (slower)
        resolved_path = resolve_raw_path(path_str, self.raw_dir)
        raw_mri = self.reader.read(resolved_path)
        tensor = torch.from_numpy(raw_mri.tensor).float()
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        if self.transform:
            tensor = self.transform(tensor)

        return DatasetSample(
            image=tensor,
            label=label_idx,
            subject_id=subject_id,
            metadata=raw_mri.header,
            dataset_name="abide",
        )
