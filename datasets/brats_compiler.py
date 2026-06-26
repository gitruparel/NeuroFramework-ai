"""Placeholder compiler implementation for the BraTS dataset."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from datasets.base_compiler import DatasetCompiler


class BraTSCompiler(DatasetCompiler):
    """Compiles the BraTS dataset. Currently placeholder."""

    def build_index(self, raw_dir: Path, csv_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError("BraTSCompiler is not yet implemented.")
