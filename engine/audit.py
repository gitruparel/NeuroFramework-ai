"""Dataset auditing engine traversing files and directories to compile distributions."""

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
from core.exceptions import MRIProcessingError
from engine.pipeline import MRIEngine
from schemas.audit import AuditReport, VoxelSpacingDistribution


class DatasetAuditor:
    """Traverses dataset directories (e.g. BraTS, ADNI, ABIDE) to analyze shapes, corruptions, and scanner distributions."""

    def __init__(self):
        self.engine = MRIEngine()

    def audit(self, dataset_path: str | Path) -> AuditReport:
        root = Path(dataset_path)
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Invalid dataset path: {dataset_path}")

        # Scan folder layout to find target files
        scan_paths = self._collect_scans(root)
        subjects = self._identify_subjects(scan_paths, root)

        corrupt_files = []
        formats_found = set()
        spacings = []
        dimensions = set()
        class_balance = {}
        missing_metadata_count = 0
        scanner_manufacturers = {}

        for path in scan_paths:
            try:
                # Load using the public MRIEngine (handles format detection, validation, cache)
                mri_data = self.engine.load(path)
                
                # Check validation layer or exception corruption
                if not mri_data.validation.is_valid:
                    corrupt_files.append(str(path))
                    continue

                # Formats
                formats_found.add(mri_data.raw.format)

                # Spacings
                spacings.append(mri_data.metadata.image.voxel_dims)

                # Dimensions
                dimensions.add(tuple(mri_data.metadata.image.dimensions))

                # Class Balance (Inferred from path subfolders, e.g. /AD/, /CN/ or /tumor/)
                cls_label = self._infer_class(path, root)
                class_balance[cls_label] = class_balance.get(cls_label, 0) + 1

                # Missing Metadata
                is_missing = (
                    not mri_data.metadata.patient.patient_id or
                    mri_data.metadata.patient.patient_id == "Anonymous_Patient"
                )
                if is_missing:
                    missing_metadata_count += 1

                # Scanner Manufacturers
                mfr = mri_data.metadata.scanner.manufacturer or "Unknown"
                scanner_manufacturers[mfr] = scanner_manufacturers.get(mfr, 0) + 1

            except Exception:
                corrupt_files.append(str(path))

        # Calculate spacing distribution
        spacing_dist = None
        if spacings:
            spacings_arr = np.array(spacings)
            spacing_dist = VoxelSpacingDistribution(
                min_spacing=list(np.min(spacings_arr, axis=0)),
                max_spacing=list(np.max(spacings_arr, axis=0)),
                mean_spacing=list(np.mean(spacings_arr, axis=0)),
            )

        unique_dims = [list(d) for d in dimensions]

        return AuditReport(
            dataset_path=str(root.resolve()),
            total_subjects=len(subjects),
            total_files=len(scan_paths),
            formats_present=sorted(list(formats_found)),
            corrupt_files=corrupt_files,
            missing_files=[],  # Populated if catalog-based matching is available
            voxel_spacings=spacing_dist,
            unique_dimensions=unique_dims,
            class_balance=class_balance,
            missing_metadata_count=missing_metadata_count,
            scanner_manufacturers=scanner_manufacturers,
        )

    def _collect_scans(self, root: Path) -> List[Path]:
        """Collects candidate files. If a subfolder only contains DICOM series, it treats folder as a single scan."""
        paths = []
        
        # Walk directories
        for entry in root.rglob("*"):
            if entry.is_dir():
                # Check if it contains DICOM series
                dicoms = list(entry.glob("*.dcm"))
                if dicoms:
                    # Treat the directory as a single DICOM loadable path
                    paths.append(entry)
            elif entry.is_file():
                ext = entry.name.lower()
                # Skip .dcm files inside directories we already added as folders
                if entry.parent in paths and ext.endswith(".dcm"):
                    continue
                if ext.endswith((".nii", ".nii.gz", ".dcm", ".png", ".jpg", ".jpeg")):
                    paths.append(entry)
        
        # Remove duplicate directory references if any
        return sorted(list(set(paths)))

    def _identify_subjects(self, paths: List[Path], root: Path) -> Set[str]:
        """Groups files by subfolders right under the root directory to count subjects."""
        subjects = set()
        for p in paths:
            try:
                rel = p.relative_to(root)
                parts = rel.parts
                if parts:
                    # If the first part is a known class label folder, the subject folder is the next subfolder
                    if len(parts) > 1 and parts[0].upper() in ("AD", "CN", "CONTROL", "TUMOR", "AUTISM", "NORMAL"):
                        subjects.add(parts[1])
                    else:
                        subjects.add(parts[0])
            except Exception:
                subjects.add(p.stem)
        return subjects

    def _infer_class(self, path: Path, root: Path) -> str:
        """Heuristically infers pathology class from directory subfolders or filename patterns."""
        # Check folder names like /AD/, /CN/, /tumor/, /control/
        for parent in path.parents:
            if parent == root:
                break
            name = parent.name.upper()
            if name in ("AD", "CN", "CONTROL", "TUMOR", "AUTISM", "NORMAL"):
                return name
        
        # Check filename substrings
        stem = path.stem.upper()
        for label in ("AD", "CN", "CONTROL", "TUMOR", "AUTISM", "NORMAL"):
            if label in stem:
                return label
                
        return "Unknown"
