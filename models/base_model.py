"""Base model wrapper defining standard PyTorch and metadata interfaces."""

from core.interfaces import BaseModel


class MRIModel(BaseModel):
    """Abstract model class implementing basic hooks and PyTorch forward signature."""

    def __init__(self, in_channels: int = 1, out_channels: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
