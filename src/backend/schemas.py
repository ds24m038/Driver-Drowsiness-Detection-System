"""Pydantic schemas for FastAPI request/response models."""
from pydantic import BaseModel
from typing import Dict, Optional


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str


class PredictionRequest(BaseModel):
    """Prediction request model."""
    image_base64: Optional[str] = None
    # Note: For multipart/form-data, we'll use UploadFile directly in the endpoint


class PredictionResponse(BaseModel):
    """Prediction response model."""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: float

