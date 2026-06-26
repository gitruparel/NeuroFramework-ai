# Processing Pipeline

Details of the MRI processing pipeline from ingestion to inference.

## Ingestion
1. **DICOM/NIfTI Loader**: Loaded via `engine/reader.py` using NiBabel or SimpleITK.
2. **Metadata Extraction**: Extracted and normalized into Pydantic schema objects in `schemas/metadata.py`.
3. **Data Quality Assessment**: Evaluated in `preprocessing/quality.py` and validated against bounds defined in `schemas/validation.py`.

## Preprocessing Steps
1. **Resampling**: `preprocessing/resample.py` maps files to a uniform voxel spacing.
2. **Bias Field Correction**: `preprocessing/bias.py` applies N4 Bias Field Correction using SimpleITK.
3. **Skull Stripping**: `preprocessing/skull_strip.py` extracts the brain volume.
4. **Normalization**: `preprocessing/normalize.py` scales intensities (e.g. z-score or min-max normalization).
5. **Cropping**: `preprocessing/crop.py` trims redundant background bounding boxes to conserve computation.
