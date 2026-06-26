"""Abstract base class for dataset compilers defining common interface, serialization, validation, and splitting logic."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from core.exceptions import ConfigurationError, ValidationFailedError


class DatasetCompiler(ABC):
    """Abstract base compiler for compiling medical datasets with stratification and K-fold support."""

    def __init__(self, output_dir: str | Path = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def build_index(self, raw_dir: Path, csv_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Scans raw directory and matches entries with csv_path to compile a list of indexed samples."""
        pass

    def validate(self, index: List[Dict[str, Any]]) -> None:
        """Validates consistency of compiled index.

        Raises:
            ValidationFailedError: If any consistency check fails.
        """
        if not index:
            raise ValidationFailedError("Index is empty.")

        subject_ids = set()
        mri_paths = set()

        for entry in index:
            # Check required fields
            for field in ["subject_id", "path", "label"]:
                if field not in entry or not entry[field]:
                    raise ValidationFailedError(f"Missing required field '{field}' in entry: {entry}")

            sub_id = entry["subject_id"]
            path_str = entry["path"]
            path = Path(path_str)

            # Check paths exist
            if not path.exists():
                raise ValidationFailedError(f"MRI file path does not exist: {path_str}")

            # Check duplicates
            if sub_id in subject_ids:
                raise ValidationFailedError(f"Duplicate subject ID found: {sub_id}")
            if path_str in mri_paths:
                raise ValidationFailedError(f"Duplicate MRI path found: {path_str}")

            subject_ids.add(sub_id)
            mri_paths.add(path_str)

    def generate_splits(
        self,
        index: List[Dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, List[str]]:
        """Creates stratified train, validation, and test splits at the patient/subject level."""
        subjects = sorted(list({entry["subject_id"] for entry in index}))
        
        # Get label for each unique subject (assuming 1 scan per subject)
        sub_to_label = {entry["subject_id"]: entry["label"] for entry in index}
        labels = [sub_to_label[sub] for sub in subjects]

        # Stratified split: first split out test set
        test_size = val_ratio + test_ratio
        
        # In case we have very few samples, handle stratification gracefully
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or any(np.sum(np.array(labels) == lbl) < 2 for lbl in unique_labels):
            # Fallback to non-stratified split if classes are too small
            train_subs, temp_subs = train_test_split(
                subjects, test_size=test_size, random_state=seed
            )
            val_subs, test_subs = train_test_split(
                temp_subs, test_size=test_ratio / test_size, random_state=seed
            )
        else:
            train_subs, temp_subs, train_labels, temp_labels = train_test_split(
                subjects, labels, test_size=test_size, random_state=seed, stratify=labels
            )
            
            # Split remainder into val and test
            unique_temp_labels = np.unique(temp_labels)
            if len(unique_temp_labels) < 2 or any(np.sum(np.array(temp_labels) == lbl) < 2 for lbl in unique_temp_labels):
                val_subs, test_subs = train_test_split(
                    temp_subs, test_size=test_ratio / test_size, random_state=seed
                )
            else:
                val_subs, test_subs = train_test_split(
                    temp_subs, test_size=test_ratio / test_size, random_state=seed, stratify=temp_labels
                )

        return {
            "train": sorted(list(train_subs)),
            "val": sorted(list(val_subs)),
            "test": sorted(list(test_subs)),
        }

    def generate_kfold(
        self,
        index: List[Dict[str, Any]],
        n_splits: int = 5,
        seed: int = 42,
    ) -> List[Dict[str, List[str]]]:
        """Generates 5-Fold stratified patient-level splits."""
        subjects = np.array(sorted(list({entry["subject_id"] for entry in index})))
        sub_to_label = {entry["subject_id"]: entry["label"] for entry in index}
        labels = np.array([sub_to_label[sub] for sub in subjects])

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or any(np.sum(labels == lbl) < n_splits for lbl in unique_labels):
            # Fallback to non-stratified KFold
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            folds = []
            for train_idx, val_idx in kf.split(subjects):
                folds.append({
                    "train": sorted(subjects[train_idx].tolist()),
                    "val": sorted(subjects[val_idx].tolist()),
                })
            return folds

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []
        for train_idx, val_idx in skf.split(subjects, labels):
            folds.append({
                "train": sorted(subjects[train_idx].tolist()),
                "val": sorted(subjects[val_idx].tolist()),
            })
        return folds

    def calculate_statistics(
        self,
        index: List[Dict[str, Any]],
        splits: Dict[str, List[str]],
        kfold: List[Dict[str, List[str]]],
    ) -> Dict[str, Any]:
        """Calculates rich statistical counts and distributions for verification."""
        total_scans = len(index)
        unique_subjects = len({entry["subject_id"] for entry in index})

        labels = [entry["label"] for entry in index]
        asd_count = sum(1 for l in labels if l == "ASD")
        control_count = sum(1 for l in labels if l == "CONTROL")

        genders = [entry.get("sex") for entry in index if entry.get("sex")]
        male_count = sum(1 for g in genders if g == "Male")
        female_count = sum(1 for g in genders if g == "Female")

        ages = [entry["age"] for entry in index if entry.get("age") is not None]
        mean_age = float(np.mean(ages)) if ages else 0.0

        sites = {}
        for entry in index:
            site = entry.get("site", "Unknown")
            sites[site] = sites.get(site, 0) + 1

        # Class balance per split
        sub_to_label = {entry["subject_id"]: entry["label"] for entry in index}
        
        def get_balance(subs: List[str]) -> Dict[str, int]:
            bal: Dict[str, int] = {}
            for s in subs:
                lbl = sub_to_label.get(s, "Unknown")
                bal[lbl] = bal.get(lbl, 0) + 1
            return bal

        split_balances = {
            split_name: {
                "count": len(subs),
                "class_balance": get_balance(subs)
            }
            for split_name, subs in splits.items()
        }

        kfold_balances = [
            {
                "train_count": len(f["train"]),
                "val_count": len(f["val"]),
                "train_balance": get_balance(f["train"]),
                "val_balance": get_balance(f["val"])
            }
            for f in kfold
        ]

        return {
            "total_mri_scans": total_scans,
            "indexed_subjects": unique_subjects,
            "asd_count": asd_count,
            "control_count": control_count,
            "male_count": male_count,
            "female_count": female_count,
            "mean_age": mean_age,
            "site_distribution": sites,
            "split_info": split_balances,
            "kfold_info": kfold_balances,
        }

    def compile(self, raw_dir: str | Path, csv_path: Optional[str | Path] = None) -> Dict[str, Any]:
        """Main entry point: indexes, validates, partitions, computes stats, and serializes outputs."""
        raw_path = Path(raw_dir)
        csv_file = Path(csv_path) if csv_path else None

        # 1. Build Index
        index = self.build_index(raw_path, csv_file)

        # 2. Validate Consistency
        self.validate(index)

        # 3. Generate Splits
        splits = self.generate_splits(index)

        # 4. Generate K-Fold Splits
        kfold = self.generate_kfold(index)

        # 5. Compute Statistics
        stats = self.calculate_statistics(index, splits, kfold)

        # 6. Save JSON Deliverables
        prefix = self.__class__.__name__.replace("Compiler", "").lower()
        
        index_path = self.output_dir / f"{prefix}_index.json"
        splits_path = self.output_dir / f"{prefix}_splits.json"
        kfold_path = self.output_dir / f"{prefix}_kfold.json"
        stats_path = self.output_dir / f"{prefix}_statistics.json"

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=4)
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=4)
        with open(kfold_path, "w", encoding="utf-8") as f:
            json.dump(kfold, f, indent=4)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)

        return {
            "index_path": index_path,
            "splits_path": splits_path,
            "kfold_path": kfold_path,
            "statistics_path": stats_path,
            "statistics": stats
        }
