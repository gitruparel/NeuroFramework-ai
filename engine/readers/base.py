"""Abstract base reader class definition."""

from abc import ABC, abstractmethod
from pathlib import Path
from schemas.mri import RawMRI


class BaseReader(ABC):
    """Abstract Base Class for all physical scan file formats."""

    @abstractmethod
    def read(self, path: Path) -> RawMRI:
        """Reads scan location returning loaded RawMRI instance containing raw tensors and headers."""
        pass
