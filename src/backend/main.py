"""FastAPI backend for driver drowsiness detection."""
import logging
import time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from PIL import Image
import io
import base64

from src.models.cnn_model import load_model, predict
from src.config.settings import MODEL_PATH, MODEL_CLASSES, API_HOST, API_PORT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Driver Drowsiness Detection API", version="1.0.0")

# Global model variable
model = None


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    prediction: str
    confidence: float
    probabilities: dict
    timestamp: float


@app.on_event("startup")
async def load_model_on_startup():
    """Load the model when the application starts."""
    global model
    try:
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            logger.warning(f"Model file not found at {model_path}. Please train the model first.")
            return
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(str(model_path), device=device)
        logger.info(f"Model loaded successfully on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Predict driver drowsiness from an uploaded image.
    
    Accepts:
    - multipart/form-data with image file
    
    Returns:
    - JSON with prediction, confidence, probabilities, and timestamp
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists and restart the service."
        )
    
    try:
        # Read image file
        contents = await file.read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Run prediction
        result = predict(model, image)
        
        # Add timestamp
        result["timestamp"] = time.time()
        
        logger.info(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/base64")
async def predict_base64(data: dict):
    """
    Predict driver drowsiness from a base64-encoded image.
    
    Request body:
    {
        "image": "base64_encoded_image_string"
    }
    
    Returns:
    - JSON with prediction, confidence, probabilities, and timestamp
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists and restart the service."
        )
    
    try:
        # Decode base64 image
        image_b64 = data.get("image", "")
        if not image_b64:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request body")
        
        # Remove data URL prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Run prediction
        result = predict(model, image)
        result["timestamp"] = time.time()
        
        logger.info(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)

