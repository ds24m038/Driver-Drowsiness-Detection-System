"""Inference worker that processes frames from Redis Streams and publishes predictions."""
import time
import io
import logging
from PIL import Image
import torch
import numpy as np

from src.config.redis_utils import (
    get_redis_client,
    consume_frames,
    acknowledge_frame,
    publish_prediction,
)
from src.backend.models import load_model, predict_image
from src.config.settings import MODEL_PATH, MODEL_INPUT_SIZE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Consumer group and name
CONSUMER_GROUP = "inference_workers"
CONSUMER_NAME = "worker_1"


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
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img_tensor = (img_tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
    
    return img_tensor


def process_frame(frame_data: bytes, model, device: str) -> tuple:
    """Process a single frame and return prediction.
    
    Args:
        frame_data: Binary image data
        model: Loaded PyTorch model
        device: Device to run inference on
        
    Returns:
        Tuple of (predicted_class, confidence)
    """
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(frame_data))
        
        # Preprocess
        image_tensor = preprocess_image(image)
        
        # Run inference
        predicted_class, confidence, _ = predict_image(model, image_tensor, device)
        
        return predicted_class, confidence
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return "alert", 0.5  # Default to alert on error


def get_device():
    """Determine the best available device for inference.
    
    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    """
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    """Main inference worker loop."""
    logger.info("Starting inference worker...")
    
    # Determine device
    device = get_device()
    
    # Load model
    logger.info(f"Loading model from {MODEL_PATH} on device {device}")
    try:
        model = load_model(MODEL_PATH, device=device)
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}")
        logger.error("Please train the model first using the training notebook.")
        return
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Connect to Redis
    logger.info("Connecting to Redis...")
    redis_client = get_redis_client()
    
    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("Connected to Redis successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return
    
    logger.info(f"Consumer group: {CONSUMER_GROUP}, Consumer name: {CONSUMER_NAME}")
    logger.info("Waiting for frames to process...")
    
    # Main processing loop
    while True:
        try:
            # Consume frames from stream
            messages = consume_frames(redis_client, CONSUMER_GROUP, CONSUMER_NAME, count=1)
            
            if not messages:
                time.sleep(0.1)  # Small delay if no messages
                continue
            
            for msg in messages:
                frame_id = msg["frame_id"]
                frame_data = msg["frame_data"]
                timestamp = msg["timestamp"]
                
                logger.info(f"Processing frame {frame_id}")
                
                # Process frame
                predicted_class, confidence = process_frame(frame_data, model, device)
                
                # Publish prediction
                prediction_timestamp = time.time()
                publish_prediction(
                    redis_client,
                    frame_id,
                    predicted_class,
                    confidence,
                    prediction_timestamp
                )
                
                logger.info(
                    f"Published prediction for frame {frame_id}: "
                    f"{predicted_class} (confidence: {confidence:.3f})"
                )
                
                # Acknowledge message
                acknowledge_frame(redis_client, CONSUMER_GROUP, msg["id"])
                
        except KeyboardInterrupt:
            logger.info("Stopping inference worker...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(1)  # Wait before retrying


if __name__ == "__main__":
    main()

