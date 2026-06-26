"""Orchestrators for dataset indexing, statistics calculation, and subject-level split partition configuration."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.exceptions import ConfigurationError
from datasets.builders.abide import ABIDEDatasetBuilder
from datasets.builders.adni import ADNIDatasetBuilder
from datasets.builders.brats import BraTSDatasetBuilder


class SplitManager:
    """Configures and saves reproducible train/val/test splits partitioned at the patient level."""

    def split_subjects(
        self,
        subject_ids: List[str],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, List[str]]:
        """Partitions subject IDs into train, val, and test splits."""
        if not (0.0 <= train_ratio <= 1.0) or not (0.0 <= val_ratio <= 1.0) or not (0.0 <= test_ratio <= 1.0):
            raise ConfigurationError("Split ratios must be positive and between 0.0 and 1.0.")
        
        # Allow slight floating point variances (e.g. 0.8 + 0.1 + 0.1 = 1.0)
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
            raise ConfigurationError(f"Split ratios sum must be equal to 1.0 (got {train_ratio + val_ratio + test_ratio:.4f})")

        # Sort first to guarantee reproducibility regardless of os listing order
        subjects = sorted(list(set(subject_ids)))
        rng = random.Random(seed)
        rng.shuffle(subjects)

        total = len(subjects)
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)

        return {
            "train": subjects[:train_end],
            "val": subjects[train_end:val_end],
            "test": subjects[val_end:],
        }

    def kfold_split(self, subject_ids: List[str], n_splits: int = 5, seed: int = 42) -> List[Dict[str, List[str]]]:
        """Generates K-Fold train/val partitions at the subject level."""
        subjects = sorted(list(set(subject_ids)))
        rng = random.Random(seed)
        rng.shuffle(subjects)

        folds = []
        n_subjects = len(subjects)
        fold_size = n_subjects // n_splits

        for fold_idx in range(n_splits):
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size if fold_idx < n_splits - 1 else n_subjects
            
            val_set = subjects[val_start:val_end]
            train_set = [s for s in subjects if s not in val_set]
            
            folds.append({
                "train": train_set,
                "val": val_set,
            })
        return folds

    def stratified_kfold_split(
        self,
        subject_ids: List[str],
        labels: List[Any],
        n_splits: int = 5,
        seed: int = 42,
    ) -> List[Dict[str, List[str]]]:
        """Generates class-stratified K-Fold partitions to balance training targets."""
        # Pair subjects with labels
        subjects_with_labels = list(zip(subject_ids, labels, strict=False))
        
        # Group by label
        classes = {}
        for sub, lbl in subjects_with_labels:
            classes.setdefault(lbl, []).append(sub)

        # Sort each class list and shuffle
        rng = random.Random(seed)
        for lbl in classes:
            classes[lbl].sort()
            rng.shuffle(classes[lbl])

        # Subdivide each class into K buckets
        buckets = [[] for _ in range(n_splits)]
        for lbl, class_subs in classes.items():
            for idx, sub in enumerate(class_subs):
                buckets[idx % n_splits].append(sub)

        # Form train/val folds from buckets
        folds = []
        for val_idx in range(n_splits):
            val_set = buckets[val_idx]
            train_set = []
            for idx, bucket in enumerate(buckets):
                if idx != val_idx:
                    train_set.extend(bucket)
            folds.append({
                "train": train_set,
                "val": val_set,
            })
        return folds

    def group_kfold_split(
        self,
        subject_ids: List[str],
        groups: List[Any],
        n_splits: int = 5,
    ) -> List[Dict[str, List[str]]]:
        """Generates Group K-Fold partitions ensuring identical groups reside in the same split."""
        # Pair subjects with groups
        sub_group_pairs = list(zip(subject_ids, groups, strict=False))
        
        # Map group to its list of subjects
        group_map = {}
        for sub, grp in sub_group_pairs:
            group_map.setdefault(grp, []).append(sub)
            
        unique_groups = sorted(list(group_map.keys()))
        
        # Subdivide groups into folds
        folds_groups = [[] for _ in range(n_splits)]
        for idx, grp in enumerate(unique_groups):
            folds_groups[idx % n_splits].append(grp)
            
        folds = []
        for val_idx in range(n_splits):
            val_grps = folds_groups[val_idx]
            val_set = []
            for grp in val_grps:
                val_set.extend(group_map[grp])
                
            train_set = []
            for idx, grps in enumerate(folds_groups):
                if idx != val_idx:
                    for grp in grps:
                        train_set.extend(group_map[grp])
            folds.append({
                "train": train_set,
                "val": val_set,
            })
        return folds


class DatasetManager:
    """Orchestrates indexing directory contents, split compilation, and metadata serialization."""

    def __init__(self, data_root: str | Path = "data"):
        self.data_root = Path(data_root)
        self.interim_dir = self.data_root / "interim"
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.split_manager = SplitManager()

    def process_dataset(
        self,
        name: str,
        raw_dir: str | Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Indexes directory, generates subject-level splits, and writes JSON configuration indexes."""
        raw_path = Path(raw_dir)
        if not raw_path.exists():
            raise ConfigurationError(f"Raw dataset path does not exist: {raw_dir}")

        # 1. Resolve builder
        name_lower = name.lower()
        if "abide" in name_lower:
            builder = ABIDEDatasetBuilder()
        elif "adni" in name_lower:
            builder = ADNIDatasetBuilder()
        elif "brats" in name_lower:
            builder = BraTSDatasetBuilder()
        else:
            raise ConfigurationError(f"Unsupported dataset builder name: {name}")

        # 2. Build index
        index = builder.build_index(raw_path)
        
        # Save index.json
        index_file = self.interim_dir / f"{name_lower}_index.json"
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)

        # 3. Compile statistics
        stats = self._calculate_stats(name, index)
        stats_file = self.interim_dir / f"{name_lower}_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        # 4. Generate splits
        subject_ids = [item["subject_id"] for item in index]
        
        # Ensure we filter out duplicate subject ids to split strictly at subject level
        unique_subjects = sorted(list(set(subject_ids)))
        splits = self.split_manager.split_subjects(
            unique_subjects,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed
        )
        
        splits_file = self.interim_dir / f"{name_lower}_splits.json"
        with open(splits_file, "w") as f:
            json.dump(splits, f, indent=2)

        return {
            "index_path": str(index_file),
            "statistics_path": str(stats_file),
            "splits_path": str(splits_file),
            "statistics": stats,
        }

    def _calculate_stats(self, name: str, index: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compiles index parameters into summary statistical counts."""
        total_files = len(index)
        unique_subjects = len(set(item["subject_id"] for item in index))
        
        # Count classes
        class_counts = {}
        for item in index:
            lbl = item.get("label")
            if lbl:
                class_counts[lbl] = class_counts.get(lbl, 0) + 1

        # Check missing modalities for multimodal directories (like BraTS)
        missing_modalities = {}
        if "brats" in name.lower():
            modalities = ["t1", "t1ce", "t2", "flair"]
            missing_count = 0
            for item in index:
                channels = item.get("channels", {})
                for mod in modalities:
                    if mod not in channels:
                        missing_count += 1
                        missing_modalities[mod] = missing_modalities.get(mod, 0) + 1
            missing_modalities["total_missing_channels"] = missing_count

        return {
            "dataset_name": name,
            "total_files": total_files,
            "unique_subjects": unique_subjects,
            "class_distribution": class_counts,
            "missing_modalities": missing_modalities,
        }
