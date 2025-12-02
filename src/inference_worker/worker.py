"""Inference worker that processes frames from Redis Streams."""
import logging
import time
import base64
import io
from pathlib import Path
import torch
from PIL import Image
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.cnn_model import load_model, predict
from src.config.redis_client import get_redis_client
from src.config.settings import (
    MODEL_PATH,
    CONSUMER_GROUP_INFERENCE,
    STREAM_FRAMES,
    STREAM_PREDICTIONS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceWorker:
    """Worker that processes frames from Redis Streams and publishes predictions."""
    
    def __init__(self, consumer_name: str = "worker-1"):
        """Initialize the inference worker."""
        self.consumer_name = consumer_name
        self.redis_client = get_redis_client()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the PyTorch model."""
        try:
            model_path = Path(MODEL_PATH)
            if not model_path.exists():
                logger.error(f"Model file not found at {model_path}")
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            self.model = load_model(str(model_path), device=self.device)
            logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def process_frame(self, frame_data: str) -> dict:
        """
        Process a single frame and return prediction.
        
        Args:
            frame_data: Base64-encoded image data
            
        Returns:
            Prediction result dictionary
        """
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(frame_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Run inference
            result = predict(self.model, image, device=self.device)
            
            return result
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            raise
    
    def run(self):
        """Main worker loop - continuously process frames from Redis Streams."""
        logger.info(f"Starting inference worker: {self.consumer_name}")
        logger.info(f"Reading from stream: {STREAM_FRAMES}")
        logger.info(f"Publishing to stream: {STREAM_PREDICTIONS}")
        
        while True:
            try:
                # Read frames from stream
                messages = self.redis_client.read_frames(
                    consumer_group=CONSUMER_GROUP_INFERENCE,
                    consumer_name=self.consumer_name,
                    count=1,
                    block=1000  # Block for 1 second
                )
                
                for msg in messages:
                    msg_id = msg["id"]
                    data = msg["data"]
                    
                    frame_id = data.get("frame_id", "unknown")
                    frame_data = data.get("frame_data", "")
                    timestamp = float(data.get("timestamp", time.time()))
                    
                    if not frame_data:
                        logger.warning(f"Empty frame data for frame {frame_id}")
                        self.redis_client.ack_frame(CONSUMER_GROUP_INFERENCE, msg_id)
                        continue
                    
                    try:
                        # Process frame
                        result = self.process_frame(frame_data)
                        
                        # Publish prediction
                        self.redis_client.publish_prediction(
                            frame_id=frame_id,
                            prediction=result["prediction"],
                            confidence=result["confidence"],
                            probabilities=result["probabilities"],
                            timestamp=timestamp
                        )
                        
                        # Update current status
                        self.redis_client.set_current_status(result["prediction"])
                        
                        # Acknowledge message
                        self.redis_client.ack_frame(CONSUMER_GROUP_INFERENCE, msg_id)
                        
                        logger.debug(
                            f"Processed frame {frame_id}: {result['prediction']} "
                            f"(confidence: {result['confidence']:.4f})"
                        )
                    
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_id}: {e}")
                        # Still acknowledge to avoid reprocessing bad frames
                        self.redis_client.ack_frame(CONSUMER_GROUP_INFERENCE, msg_id)
            
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1)  # Wait before retrying


def main():
    """Main entry point for the inference worker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inference Worker for Driver Drowsiness Detection")
    parser.add_argument(
        "--consumer-name",
        type=str,
        default=f"worker-{int(time.time())}",
        help="Consumer name for Redis Streams"
    )
    
    args = parser.parse_args()
    
    try:
        worker = InferenceWorker(consumer_name=args.consumer_name)
        worker.run()
    except Exception as e:
        logger.error(f"Worker failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

