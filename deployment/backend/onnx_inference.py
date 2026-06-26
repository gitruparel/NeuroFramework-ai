"""ONNX Runtime wrapper execution for optimized MRI inference."""

from pathlib import Path
from typing import Any, Dict
import numpy as np
import onnxruntime as ort
from core.exceptions import ModelExecutionError


class ONNXMRIInferenceEngine:
    """Wrapper encapsulating execution via ONNX runtime using standard execution providers."""

    def __init__(self, model_path: Path | str, providers: list[str] | None = None):
        self.model_path = Path(model_path)
        self.providers = providers or ["CPUExecutionProvider"]
        if not self.model_path.exists():
            raise ModelExecutionError(f"ONNX Model not found at {self.model_path}")
        
        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=self.providers)
        except Exception as e:
            raise ModelExecutionError(f"Failed to load ONNX model: {e}") from e

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """Executes model forward pass using ONNX session runtime."""
        try:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_array})
            return outputs[0]
        except Exception as e:
            raise ModelExecutionError(f"ONNX Model evaluation failed: {e}") from e
