"""Concrete index builder for the ABIDE (Autism) dataset."""

from pathlib import Path
from typing import Any, Dict, List
from datasets.builders.base import BaseDatasetBuilder


class ABIDEDatasetBuilder(BaseDatasetBuilder):
    """Indexes ABIDE dataset structured as raw/abide/<label>/<subject_id>/scan.nii.gz."""

    def build_index(self, root_dir: Path) -> List[Dict[str, Any]]:
        root = Path(root_dir)
        index = []
        if not root.exists() or not root.is_dir():
            return index

        # Discover category subfolders (e.g. CONTROL, ASD)
        for class_dir in root.iterdir():
            if not class_dir.is_dir():
                continue
            
            label_name = class_dir.name.upper()
            if label_name not in ("CONTROL", "ASD"):
                # Fallback: if class folders are named differently but represent ASD/CONTROL
                if "CONTROL" in label_name:
                    inferred_label = "CONTROL"
                elif "ASD" in label_name or "AUTISM" in label_name:
                    inferred_label = "ASD"
                else:
                    continue
            else:
                inferred_label = label_name

            # Look for subjects directories
            for sub_dir in class_dir.iterdir():
                if not sub_dir.is_dir():
                    continue

                subject_id = sub_dir.name
                
                # Check for scan files (NIfTI)
                scans = list(sub_dir.glob("*.nii*"))
                if not scans:
                    continue

                # Take the first NIfTI scan found
                scan_path = scans[0]
                index.append({
                    "subject_id": subject_id,
                    "path": str(scan_path.resolve().as_posix()),
                    "label": inferred_label,
                    "metadata": {
                        "modality": "T1w",
                        "inferred_label": inferred_label,
                    }
                })

        return sorted(index, key=lambda x: x["subject_id"])
