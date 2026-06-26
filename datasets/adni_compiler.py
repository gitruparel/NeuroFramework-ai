"""Placeholder compiler implementation for the ADNI dataset."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from datasets.base_compiler import DatasetCompiler


class ADNICompiler(DatasetCompiler):
    """Compiles the ADNI dataset. Currently placeholder."""

    def build_index(self, raw_dir: Path, csv_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError("ADNICompiler is not yet implemented.")
