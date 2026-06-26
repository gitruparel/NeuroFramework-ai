"""Concrete index builder for the ADNI (Alzheimer) dataset."""

from pathlib import Path
from typing import Any, Dict, List
from datasets.builders.base import BaseDatasetBuilder


class ADNIDatasetBuilder(BaseDatasetBuilder):
    """Indexes ADNI dataset structured as raw/adni/<label>/<subject_id>/scan.nii or DICOM folders."""

    def build_index(self, root_dir: Path) -> List[Dict[str, Any]]:
        root = Path(root_dir)
        index = []
        if not root.exists() or not root.is_dir():
            return index

        # Discover category subfolders (e.g. CN, AD, MCI)
        for class_dir in root.iterdir():
            if not class_dir.is_dir():
                continue

            label_name = class_dir.name.upper()
            if label_name not in ("CN", "AD", "MCI"):
                # Fallback matching
                if "CN" in label_name or "NORMAL" in label_name or "CONTROL" in label_name:
                    inferred_label = "CN"
                elif "AD" in label_name or "ALZHEIMER" in label_name:
                    inferred_label = "AD"
                elif "MCI" in label_name:
                    inferred_label = "MCI"
                else:
                    continue
            else:
                inferred_label = label_name

            for sub_dir in class_dir.iterdir():
                if not sub_dir.is_dir():
                    continue

                subject_id = sub_dir.name
                
                # Check for NIfTI scans first
                niftis = list(sub_dir.glob("*.nii*"))
                if niftis:
                    index.append({
                        "subject_id": subject_id,
                        "path": str(niftis[0].resolve().as_posix()),
                        "label": inferred_label,
                        "metadata": {"modality": "T1w", "format": "nifti"}
                    })
                    continue

                # Check for DICOM folders or .dcm files
                dicoms = list(sub_dir.glob("*.dcm"))
                if dicoms:
                    index.append({
                        "subject_id": subject_id,
                        "path": str(dicoms[0].resolve().as_posix()),
                        "label": inferred_label,
                        "metadata": {"modality": "T1w", "format": "dicom"}
                    })
                    continue

                # Check if sub_dir itself is a DICOM series directory (contains DICOM files inside subfolders)
                sub_dicoms = list(sub_dir.rglob("*.dcm"))
                if sub_dicoms:
                    # In DICOM, we index the parent directory containing the slice series
                    parent_dir = sub_dicoms[0].parent
                    index.append({
                        "subject_id": subject_id,
                        "path": str(parent_dir.resolve().as_posix()),
                        "label": inferred_label,
                        "metadata": {"modality": "T1w", "format": "dicom_series"}
                    })

        return sorted(index, key=lambda x: x["subject_id"])
