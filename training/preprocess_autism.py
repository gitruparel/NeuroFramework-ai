"""Standalone executable command-line script for offline preprocessing of the ABIDE dataset using multiprocessing."""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import torch
import numpy as np

from core.logging import setup_logger
from preprocessing.pipeline import PreprocessingPipeline
from schemas.mri import MRIData, RawMRI
from schemas.quality import QualityReport
from schemas.validation import ValidationReport, FileValidationReport, MRIValidationReport
from engine.metadata import MetadataExtractor
from schemas.processing import ExecutionContext
from engine.readers.nifti import NiftiReader

logger = setup_logger("preprocess_autism", "training/preprocess_autism.log")


def wrap_raw_mri(raw_mri: RawMRI) -> MRIData:
    """Helper wrapping raw NIfTI scans into the MRIData schema without slow QA checks."""
    metadata = MetadataExtractor().extract(raw_mri)
    return MRIData(
        raw=raw_mri,
        metadata=metadata,
        quality=QualityReport(
            noise_score=0.0, blur_score=0.0, motion_score=0.0, contrast_score=0.0,
            dynamic_range=0.0, resolution=1.0, slice_count=raw_mri.tensor.shape[-1], overall_score=1.0
        ),
        validation=ValidationReport(
            file_validation=FileValidationReport(exists=True, readable=True, header_valid=True, corrupt=False),
            mri_validation=MRIValidationReport(
                voxel_spacing_valid=True, dimensions_valid=True, intensity_valid=True,
                orientation_valid=True, metadata_complete=True, empty_slices_detected=False
            ),
            is_valid=True
        ),
        history=[],
        preview=[],
        statistics={"min": 0.0, "max": 1.0, "mean": 0.0, "std": 1.0, "shape": list(raw_mri.tensor.shape), "dtype": str(raw_mri.tensor.dtype)}
    )


def resolve_raw_path(path_str: str, raw_dir: Path | None) -> Path:
    """Resolves raw filepath relative to the new raw directory if provided."""
    path = Path(path_str)
    if raw_dir is None:
        return path
    
    parts = list(path.parts)
    # Find parts matching 'sub-*'
    sub_idx = -1
    for idx, part in enumerate(parts):
        if part.startswith("sub-"):
            sub_idx = idx
            break
            
    if sub_idx != -1:
        # Include site directory (one level above 'sub-')
        start_idx = max(0, sub_idx - 1)
        rel_path = Path(*parts[start_idx:])
        return Path(raw_dir) / rel_path
    
    # Fallback to filename
    return Path(raw_dir) / path.name


def process_single_subject(
    subject_id: str,
    path_str: str,
    preprocessed_dir: Path,
    config_yaml: Path,
    raw_dir: Path | None = None,
) -> tuple[bool, float, list[tuple[str, float]]]:
    """Worker function to preprocess a single subject and write its cached .pt file."""
    import time
    start_t = time.time()
    cache_path = preprocessed_dir / f"{subject_id}.pt"
    if cache_path.exists():
        return True, 0.0, []

    # Initialize a local reader and pipeline within process boundary
    reader = NiftiReader()
    pipeline = PreprocessingPipeline.from_yaml(config_yaml, "autism")
    ctx = ExecutionContext(
        logger=logging.getLogger("preprocess_worker"),
        cache=None,
        config={},
        seed=42,
        device="cpu"
    )

    try:
        resolved_path = resolve_raw_path(path_str, raw_dir)
        raw_mri = reader.read(resolved_path)
        mri_data = wrap_raw_mri(raw_mri)
        processed = pipeline.process(mri_data, ctx)

        # Collect step timings from history
        step_timings = []
        for record in processed.history:
            step_timings.append((record.step_name, record.duration))

        # Save preprocessed outputs
        torch.save({
            "image": processed.image,
            "affine": processed.affine,
            "metadata": processed.metadata.model_dump()
        }, cache_path)
        return True, time.time() - start_t, step_timings
    except Exception as e:
        # Since it's in a subprocess, printing to stdout/stderr is safer
        print(f"ERROR: Failed preprocessing for subject {subject_id}: {e}")
        return False, time.time() - start_t, []


def preprocess_abide_dataset(
    index_file: Path,
    preprocessed_dir: Path,
    config_yaml: Path,
    raw_dir: Path | None = None,
    max_workers: int | None = None,
    limit: int | None = None,
) -> None:
    """Preprocesses all raw ABIDE scans in parallel and caches outputs to disk."""
    import time
    logger.info("Starting offline dataset preprocessing...")
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    with open(index_file, encoding="utf-8") as f:
        items = json.load(f)

    # Filter out already preprocessed files first to avoid pool overhead
    todo_items = []
    for item in items:
        subject_id = item["subject_id"]
        cache_path = preprocessed_dir / f"{subject_id}.pt"
        if not cache_path.exists():
            todo_items.append(item)

    # Apply limit argument if specified
    if limit is not None:
        logger.info(f"Limiting preprocessing to first {limit} subjects.")
        todo_items = todo_items[:limit]

    if not todo_items:
        logger.info("All subjects already preprocessed. Nothing to do.")
        return

    logger.info(f"Submitting {len(todo_items)} subjects for preprocessing (using {max_workers or 'default'} workers)...")

    success_count = 0
    run_start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_subject,
                item["subject_id"],
                item["path"],
                preprocessed_dir,
                config_yaml,
                raw_dir
            ): item["subject_id"]
            for item in todo_items
        }

        for idx, future in enumerate(as_completed(futures)):
            sub_id = futures[future]
            try:
                success, sub_time, step_timings = future.result()
                if success:
                    success_count += 1
                
                # Performance reporting and ETA calculations
                elapsed_total = time.time() - run_start
                avg_time = elapsed_total / (idx + 1)
                remaining = len(todo_items) - (idx + 1)
                eta_seconds = remaining * avg_time

                hours = int(eta_seconds // 3600)
                minutes = int((eta_seconds % 3600) // 60)
                seconds = int(eta_seconds % 60)
                
                if hours > 0:
                    eta_str = f"{hours}h {minutes}m"
                elif minutes > 0:
                    eta_str = f"{minutes}m {seconds}s"
                else:
                    eta_str = f"{seconds}s"

                # Log progress
                logger.info(
                    f"[{idx + 1}/{len(todo_items)}] Subject: {sub_id} | "
                    f"Time: {sub_time:.1f}s | Average: {avg_time:.1f}s | ETA: {eta_str}"
                )
                if step_timings:
                    for step_name, elapsed in step_timings:
                        logger.info(f"  {step_name:<18} {elapsed:5.1f} s")
            except Exception as e:
                logger.error(f"Worker generated exception for subject {sub_id}: {e}")

    logger.info(f"Offline dataset preprocessing completed. Successfully preprocessed: {success_count}/{len(todo_items)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run offline preprocessing for the Autism dataset in parallel.")
    parser.add_argument("--index", default="data/abide_index.json", help="Path to index JSON file")
    parser.add_argument("--raw-dir", default=None, help="Optional raw dataset root directory override")
    parser.add_argument("--output-dir", default="data/processed/abide", help="Directory to save preprocessed tensors")
    parser.add_argument("--config", default="configs/preprocessing.yaml", help="Path to preprocessing configuration file")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (defaults to CPU count)")
    parser.add_argument("--limit", type=int, default=None, help="Limit preprocessing to first N subjects")

    args = parser.parse_args()

    preprocess_abide_dataset(
        index_file=Path(args.index),
        preprocessed_dir=Path(args.output_dir),
        config_yaml=Path(args.config),
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        max_workers=args.workers,
        limit=args.limit
    )
