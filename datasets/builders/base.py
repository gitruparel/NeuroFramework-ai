"""Abstract Base Class for dataset folder normalization and indexing."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class BaseDatasetBuilder(ABC):
    """Parses raw datasets and indexes scans/metadata into a unified catalog structure."""

    @abstractmethod
    def build_index(self, root_dir: Path) -> List[Dict[str, Any]]:
        """Scans the directory and compiles list of standardized dictionaries per scan/subject.

        Format returned for classification:
        [
            {
                "subject_id": "sub-01",
                "path": "data/raw/abide/CONTROL/sub-01/scan.nii.gz",
                "label": "CONTROL",
                "metadata": {}
            }
        ]

        Format returned for segmentation:
        [
            {
                "subject_id": "patient-001",
                "channels": {
                    "t1": "path/to/t1.nii.gz",
                    "t1ce": "path/to/t1ce.nii.gz",
                    "t2": "path/to/t2.nii.gz",
                    "flair": "path/to/flair.nii.gz"
                },
                "mask": "path/to/seg.nii.gz",
                "metadata": {}
            }
        ]
        """
        pass
