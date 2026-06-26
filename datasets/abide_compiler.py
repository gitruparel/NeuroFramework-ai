"""Concrete compiler for the ABIDE dataset matching BIDS structures and phenotypic CSV records."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from datasets.base_compiler import DatasetCompiler
from core.exceptions import ValidationFailedError


class ABIDECompiler(DatasetCompiler):
    """Compiles the ABIDE dataset from a BIDS layout and phenotypic CSV."""

    def build_index(self, raw_dir: Path, csv_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        if not csv_path or not csv_path.exists():
            raise ValidationFailedError(f"Phenotypic CSV path is missing or invalid: {csv_path}")

        # 1. Read phenotypic CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValidationFailedError(f"Failed to read phenotypic CSV: {e}")

        # Standardize CSV columns
        required_cols = ["SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN", "SITE_ID"]
        for col in required_cols:
            if col not in df.columns:
                raise ValidationFailedError(f"Required column '{col}' missing from phenotypic CSV.")

        # Create mapping dictionary by integer SUB_ID
        csv_map = {}
        for _, row in df.iterrows():
            try:
                sub_id_val = row["SUB_ID"]
                # Convert to int for matching
                sub_id_int = int(float(sub_id_val))
            except (ValueError, TypeError):
                continue

            dx = row["DX_GROUP"]
            sex_val = row["SEX"]
            age_val = row["AGE_AT_SCAN"]
            site = row["SITE_ID"]

            # Map label
            label = None
            if dx == 1:
                label = "ASD"
            elif dx == 2:
                label = "CONTROL"

            # Map sex
            sex = None
            if sex_val == 1:
                sex = "Male"
            elif sex_val == 2:
                sex = "Female"

            # Optional/rich metadata
            fiq = row.get("FIQ")
            viq = row.get("VIQ")
            piq = row.get("PIQ")

            def clean_meta(val):
                if val is None or pd.isna(val) or str(val).strip() in ("-9999", "-9999.0", "NaN", "nan", ""):
                    return None
                try:
                    f = float(val)
                    if f.is_integer():
                        return int(f)
                    return f
                except (ValueError, TypeError):
                    return str(val).strip()

            csv_map[sub_id_int] = {
                "site": str(site).strip(),
                "label": label,
                "age": clean_meta(age_val),
                "sex": sex,
                "fiq": clean_meta(fiq),
                "viq": clean_meta(viq),
                "piq": clean_meta(piq),
                "scanner_site": str(site).strip(),
            }

        # 2. Scan BIDS directory structure
        index = []
        pattern = re.compile(r"sub-(\d+)")
        
        # Traverse directories looking for T1w scans
        for filepath in raw_dir.rglob("*_T1w.nii*"):
            if not filepath.is_file():
                continue

            # Extract subject ID from path or filename
            match = pattern.search(filepath.name)
            if not match:
                # Try to extract from parent folder name
                match = pattern.search(filepath.parent.name)
                if not match:
                    # Try grandparent
                    match = pattern.search(filepath.parent.parent.name)

            if not match:
                continue

            sub_id_str = f"sub-{match.group(1)}"
            try:
                sub_id_int = int(match.group(1))
            except ValueError:
                continue

            # Match to phenotypic record
            if sub_id_int not in csv_map:
                # Every MRI has a phenotypic row validation check later
                # We can either raise immediately or add to index with None to trigger validation
                csv_data = {}
            else:
                csv_data = csv_map[sub_id_int]

            index.append({
                "subject_id": sub_id_str,
                "site": csv_data.get("site", "Unknown"),
                "path": str(filepath.resolve().as_posix()),
                "label": csv_data.get("label"),
                "age": csv_data.get("age"),
                "sex": csv_data.get("sex"),
                "fiq": csv_data.get("fiq"),
                "viq": csv_data.get("viq"),
                "piq": csv_data.get("piq"),
                "dataset": "ABIDE_I",
                "modality": "T1w",
                "scanner_site": csv_data.get("scanner_site", "Unknown"),
            })

        # Validation constraints check:
        # "Validate:
        # * every MRI has a phenotypic row
        # * every phenotypic row with MRI is indexed"
        # Let's count how many phenotypic rows match to MRI
        indexed_ints = {int(re.search(r"sub-(\d+)", item["subject_id"]).group(1)) for item in index if re.search(r"sub-(\d+)", item["subject_id"])}

        # Verify that all scanned MRIs have matching phenotypic rows
        for item in index:
            try:
                sub_id_int = int(re.search(r"sub-(\d+)", item["subject_id"]).group(1))
            except (ValueError, TypeError, AttributeError):
                raise ValidationFailedError(f"MRI scanned subject ID {item['subject_id']} has malformed ID.")
            
            if sub_id_int not in csv_map:
                raise ValidationFailedError(f"MRI scan found at {item['path']} does not have a matching phenotypic CSV record.")
            if not item.get("label"):
                raise ValidationFailedError(f"Subject {item['subject_id']} is missing a valid diagnosis label (DX_GROUP).")

        return sorted(index, key=lambda x: x["subject_id"])
