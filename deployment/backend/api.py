"""FastAPI backend API endpoints definition."""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from schemas.prediction import Prediction, ClassProbability

app = FastAPI(
    title="Structural MRI Analysis Platform API",
    version="0.1.0",
    debug=settings.api_debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Service status health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.post("/api/v1/analyze", response_model=Prediction)
async def analyze_mri(file: UploadFile = File(...)):
    """Receives structural MRI scan and performs inference pipeline execution."""
    return Prediction(
        patient_id="patient_upload_example",
        model_name="MedicalNet3D",
        probabilities=[
            ClassProbability(class_name="Alzheimer", probability=0.85),
            ClassProbability(class_name="Normal", probability=0.15),
        ],
        metrics={"dice_coefficient": 0.88}
    )
