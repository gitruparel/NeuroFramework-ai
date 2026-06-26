# Deployment API Documentation

The platform exposes endpoints using FastAPI for running inference, checking processing status, and downloading report exports.

## Core Endpoints

### 1. Ingest & Run Inference
- **URL**: `/api/v1/analyze`
- **Method**: `POST`
- **Payload**: Multipart form upload containing MRI scans and optional preprocessing configuration.
- **Response**: Predictor reports schema.

### 2. Job Status
- **URL**: `/api/v1/jobs/{job_id}`
- **Method**: `GET`
- **Response**: Execution progress status (Ingesting, Preprocessing, Inference, Generating Report, Completed).

### 3. Retrieve Report
- **URL**: `/api/v1/reports/{report_id}`
- **Method**: `GET`
- **Response**: PDF binary stream.
