"""Factory and entry points for executing dataset compiler workflows."""

from pathlib import Path
from typing import Any, Dict, Optional
from datasets.base_compiler import DatasetCompiler
from datasets.abide_compiler import ABIDECompiler
from datasets.adni_compiler import ADNICompiler
from datasets.brats_compiler import BraTSCompiler

COMPILERS = {
    "abide": ABIDECompiler,
    "adni": ADNICompiler,
    "brats": BraTSCompiler,
}


def get_compiler(name: str, output_dir: str | Path = "data") -> DatasetCompiler:
    """Retrieves the compiler instance for the specified dataset name."""
    name_lower = name.lower()
    # Support names like "abide_i" or "brats2023"
    matched_key = None
    for key in COMPILERS:
        if key in name_lower:
            matched_key = key
            break

    if not matched_key:
        raise ValueError(f"No compiler registered for dataset '{name}'. Available: {list(COMPILERS.keys())}")

    return COMPILERS[matched_key](output_dir=output_dir)


def compile_dataset(
    name: str,
    raw_dir: str | Path,
    csv_path: Optional[str | Path] = None,
    output_dir: str | Path = "data",
) -> Dict[str, Any]:
    """Helper function to instantiate and run a dataset compiler in one call."""
    compiler = get_compiler(name, output_dir=output_dir)
    return compiler.compile(raw_dir=raw_dir, csv_path=csv_path)
