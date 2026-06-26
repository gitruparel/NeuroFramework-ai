"""Abstract Base Classes defining key interfaces across the MRI Analysis Platform."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset
from schemas.mri import MRIData
from schemas.prediction import Prediction
from schemas.report import Report


class BasePreprocessor(ABC):
    """Abstract Base Class for MRI preprocessing pipelines."""

    @abstractmethod
    def process(self, mri_data: MRIData) -> MRIData:
        """Applies sequence of preprocessing transformations on raw/interim MRI data."""
        pass


class BaseDataset(Dataset, ABC):
    """Abstract Base Class for MRI datasets, extending PyTorch's Dataset."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        pass


class BaseModel(torch.nn.Module, ABC):
    """Abstract Base Class for MRI deep learning models, extending PyTorch's nn.Module."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch model forward pass."""
        pass


class BaseCallback(ABC):
    """Abstract Base Class for Trainer hooks."""

    def on_train_start(self, trainer: Any) -> None:
        """Hook before training loop starts."""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Hook after epoch finishes."""
        pass


class BaseTrainer(ABC):
    """Abstract Base Class for training engines."""

    @abstractmethod
    def train(self) -> None:
        """Starts model training execution loop."""
        pass

    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """Runs validation loop and returns calculated metrics."""
        pass


class BaseExplainer(ABC):
    """Abstract Base Class for attribution and explainability maps."""

    @abstractmethod
    def generate_heatmap(self, model: BaseModel, tensor: torch.Tensor, target_class: int) -> torch.Tensor:
        """Calculates saliency/attribution map indicating key brain regions for decision."""
        pass


class BaseReporter(ABC):
    """Abstract Base Class for compiling validation or prediction reports."""

    @abstractmethod
    def generate_report(self, prediction: Prediction, output_path: str) -> Report:
        """Compiles and builds the PDF artifact containing slice renders and findings."""
        pass
