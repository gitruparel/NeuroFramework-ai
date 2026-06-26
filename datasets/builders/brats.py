"""Concrete index builder for the BraTS (Tumor Segmentation) dataset."""

from pathlib import Path
from typing import Any, Dict, List
from datasets.builders.base import BaseDatasetBuilder


class BraTSDatasetBuilder(BaseDatasetBuilder):
    """Indexes BraTS dataset structured as raw/brats/<subject_id>/ containing multi-modal volumes.

    Modality channels expected:
    - T1
    - T1ce (contrast-enhanced T1)
    - T2
    - FLAIR
    - Segmentation mask (seg)
    """

    def build_index(self, root_dir: Path) -> List[Dict[str, Any]]:
        root = Path(root_dir)
        index = []
        if not root.exists() or not root.is_dir():
            return index

        # Every subdirectory right under the root represents a subject/patient folder
        for sub_dir in root.iterdir():
            if not sub_dir.is_dir():
                continue

            subject_id = sub_dir.name
            
            # Map files dynamically by looking at typical file suffixes
            channels = {}
            mask_path = None
            
            for file in sub_dir.glob("*.nii*"):
                name = file.name.lower()
                resolved_path = str(file.resolve().as_posix())
                
                # Check for segmentation target
                if "_seg" in name:
                    mask_path = resolved_path
                elif "_t1ce" in name:
                    channels["t1ce"] = resolved_path
                elif "_t1" in name:
                    channels["t1"] = resolved_path
                elif "_t2" in name:
                    channels["t2"] = resolved_path
                elif "_flair" in name:
                    channels["flair"] = resolved_path

            # We index the subject and include whichever channels/masks were discovered
            index.append({
                "subject_id": subject_id,
                "channels": channels,
                "mask": mask_path,
                "metadata": {
                    "modality": "Multimodal",
                    "available_channels": list(channels.keys()),
                    "has_mask": mask_path is not None
                }
            })

        return sorted(index, key=lambda x: x["subject_id"])
