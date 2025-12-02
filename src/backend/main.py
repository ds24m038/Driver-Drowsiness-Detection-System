"""FastAPI backend for driver drowsiness detection model serving."""
import time
import base64
import io
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from src.backend.models import load_model, predict_image
from src.backend.schemas import HealthResponse, PredictionResponse
from src.config.settings import MODEL_PATH, MODEL_INPUT_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Driver Drowsiness Detection API", version="1.0.0")

# Global model variable
model = None


def get_device():
    """Determine the best available device for inference.
    
    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


device = get_device()


@app.on_event("startup")
async def load_model_on_startup():
    """Load the model when the application starts."""
    global model
    try:
        logger.info(f"Loading model from {MODEL_PATH} on device {device}")
        model = load_model(MODEL_PATH, device=device)
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.warning(f"Model file not found at {MODEL_PATH}. API will be available but predictions will fail.")
        model = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess an image for model inference.
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed tensor (3, 227, 227)
    """
    # Resize to model input size
    image = image.resize(MODEL_INPUT_SIZE, Image.Resampling.BILINEAR)
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Convert to tensor and change from HWC to CHW
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    
    # Normalize with ImageNet stats (common practice)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img_tensor = (img_tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
    
    return img_tensor


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Driver Drowsiness Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if model is None:
        return HealthResponse(
            status="degraded",
            message="Model not loaded. Check model path."
        )
    return HealthResponse(
        status="healthy",
        message="API is running and model is loaded"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict driver drowsiness from an uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction response with class, confidence, and probabilities
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check the model path."
        )
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess
        image_tensor = preprocess_image(image)
        
        # Run inference
        predicted_class, confidence, probabilities = predict_image(model, image_tensor, device)
        
        # Return response
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(request: dict):
    """Predict driver drowsiness from a base64-encoded image.
    
    Args:
        request: Dict with "image_base64" key containing base64 string
        
    Returns:
        Prediction response with class, confidence, and probabilities
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check the model path."
        )
    
    try:
        # Decode base64 image
        image_base64 = request.get("image_base64", "")
        if not image_base64:
            raise HTTPException(status_code=400, detail="Missing image_base64 in request")
        
        # Remove data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess
        image_tensor = preprocess_image(image)
        
        # Run inference
        predicted_class, confidence, probabilities = predict_image(model, image_tensor, device)
        
        # Return response
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    from src.config.settings import API_HOST, API_PORT
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)

