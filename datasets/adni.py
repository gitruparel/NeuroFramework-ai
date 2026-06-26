"""Custom PyTorch dataset implementation for the ADNI (Alzheimer) dataset."""

from pathlib import Path
import torch
from datasets.base import MRIDataset, DatasetRegistry
from schemas.dataset import DatasetSample
from engine.readers.factory import ReaderFactory


@DatasetRegistry.register("adni")
class ADNIDataset(MRIDataset):
    """PyTorch Dataset loader for ADNI Alzheimer scans (supports NIfTI & DICOM)."""

    def __getitem__(self, index: int) -> DatasetSample:
        item = self.items[index]
        subject_id = item["subject_id"]
        path_str = item["path"]
        label_name = item["label"]

        # Instantiate raw format reader (bypasses full MRIEngine pipeline)
        path = Path(path_str)
        reader = ReaderFactory.create(path)
        raw_mri = reader.read(path)

        # Standardise dimensions to channel format: (1, X, Y, Z)
        tensor = torch.from_numpy(raw_mri.tensor).float()
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        # Apply transforms if any
        if self.transform:
            tensor = self.transform(tensor)

        # Resolve classification label map
        label_idx = self.label_map.get(label_name, -1)

        return DatasetSample(
            image=tensor,
            label=label_idx,
            subject_id=subject_id,
            metadata=raw_mri.header,
            dataset_name="adni",
        )
