"""Abstract base dataset definition and registry factory class."""

import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type
from core.interfaces import BaseDataset
from schemas.dataset import DatasetSample


class MRIDataset(BaseDataset):
    """Abstract PyTorch dataset indexing raw scans and returning uniform DatasetSample objects."""

    def __init__(
        self,
        index_file: str | Path,
        split_file: Optional[str | Path] = None,
        split_name: Optional[str] = None,  # "train", "val", or "test"
        transform: Optional[Callable] = None,
        label_map: Optional[Dict[str, int]] = None,
    ):
        self.index_file = Path(index_file)
        self.transform = transform
        self.label_map = label_map or {}

        # 1. Load catalog index
        if not self.index_file.exists():
            raise FileNotFoundError(f"Dataset index file not found: {self.index_file}")

        with open(self.index_file, encoding="utf-8") as f:
            self.all_items = json.load(f)

        # 2. Filter by split if split information is provided
        if split_file and split_name:
            split_path = Path(split_file)
            if not split_path.exists():
                raise FileNotFoundError(f"Split file not found: {split_path}")
            with open(split_path, encoding="utf-8") as f:
                splits = json.load(f)
            split_subjects = set(splits.get(split_name, []))
            self.items = [item for item in self.all_items if item["subject_id"] in split_subjects]
        else:
            self.items = self.all_items

    def __len__(self) -> int:
        return len(self.items)

    @abstractmethod
    def __getitem__(self, index: int) -> DatasetSample:
        """Loads voxel arrays quickly and returns standardized DatasetSample container."""
        pass


class DatasetRegistry:
    """Registry pattern keeping catalog classes decoupled from client scripts."""

    _registry: Dict[str, Type[MRIDataset]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[MRIDataset]], Type[MRIDataset]]:
        """Decorator to register dataset classes."""
        def decorator(subclass: Type[MRIDataset]) -> Type[MRIDataset]:
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> MRIDataset:
        """Instantiates the registered dataset using provided parameters."""
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise ValueError(f"Dataset '{name}' is not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[name_lower](**kwargs)
