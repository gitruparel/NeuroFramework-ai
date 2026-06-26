# Architecture Design

This document details the system design of the AI-powered Structural MRI Analysis Platform.

## System Overview

```mermaid
graph TD
    raw_mri[Raw MRI Scan .nii.gz / DICOM] --> reader[Engine Reader]
    reader --> validator[Engine Validator]
    validator --> preprocessor[Preprocessing Pipeline]
    preprocessor --> dataset[MRI Dataset]
    dataset --> model[PyTorch / MONAI Models]
    model --> explainability[Explainability Attribution Map]
    model --> predictor[Predictor Engine]
    predictor --> report[ReportLab PDF Generator]
    predictor --> deployment[FastAPI Backend / Streamlit Frontend]
```

## core Modules
- `core/interfaces.py`: Strictly typed contracts using Abstract Base Classes (ABCs) that all components extend.
- `core/config.py`: Single source of truth configuration handling using Pydantic Settings.
- `core/logging.py`: Structured multithreaded and process-safe logging mapped to directory-specific files.
- `schemas/`: Pydantic models for domain entity safety.
