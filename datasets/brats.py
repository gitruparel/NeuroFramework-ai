"""Custom PyTorch dataset implementation for the BraTS (Tumor Segmentation) dataset."""

from pathlib import Path
import torch
from core.exceptions import MRIProcessingError
from datasets.base import MRIDataset, DatasetRegistry
from schemas.dataset import DatasetSample
from engine.readers.nifti import NiftiReader


@DatasetRegistry.register("brats")
class BraTSDataset(MRIDataset):
    """PyTorch Dataset loader for BraTS brain tumor segmentation scans (4 stacked modal channels)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reader = NiftiReader()
        self.modalities = ["t1", "t1ce", "t2", "flair"]

    def __getitem__(self, index: int) -> DatasetSample:
        item = self.items[index]
        subject_id = item["subject_id"]
        channels_dict = item.get("channels", {})
        mask_path_str = item.get("mask")

        # 1. Load and stack all 4 modalities
        channel_tensors = []
        loaded_metadata = {}
        
        for mod in self.modalities:
            if mod not in channels_dict:
                raise MRIProcessingError(f"Missing required modality channel '{mod}' for subject {subject_id}")
            
            mod_path = Path(channels_dict[mod])
            raw_mri = self.reader.read(mod_path)
            channel_tensors.append(torch.from_numpy(raw_mri.tensor).float())
            loaded_metadata[f"{mod}_header"] = raw_mri.header

        # Stack into shape: (4, X, Y, Z)
        stacked_image = torch.stack(channel_tensors, dim=0)

        # 2. Load segmentation target mask if present
        mask_tensor = None
        if mask_path_str:
            mask_path = Path(mask_path_str)
            raw_mask = self.reader.read(mask_path)
            # Add channel dimension: (X, Y, Z) -> (1, X, Y, Z)
            mask_tensor = torch.from_numpy(raw_mask.tensor).long().unsqueeze(0)

        # Apply transforms if any (transforms typically apply to image or both image and mask)
        if self.transform:
            # Simple transform application (complex spatial transforms for segmentations
            # will be integrated in Stage 2 Preprocessing Framework transforms)
            stacked_image = self.transform(stacked_image)

        return DatasetSample(
            image=stacked_image,
            mask=mask_tensor,
            subject_id=subject_id,
            metadata=loaded_metadata,
            dataset_name="brats",
        )
