"""Alarm manager that tracks drowsiness state and manages alarm logic."""
import time
import logging

from src.config.redis_utils import (
    get_redis_client,
    consume_predictions,
    acknowledge_prediction,
    get_current_status,
    set_current_status,
    get_consecutive_drowsy_frames,
    set_consecutive_drowsy_frames,
    increment_consecutive_drowsy_frames,
    reset_consecutive_drowsy_frames,
    is_alarm_active,
    set_alarm_active,
    publish_alarm_event,
)
from src.config.settings import ALARM_THRESHOLD_FRAMES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Consumer group and name
CONSUMER_GROUP = "alarm_managers"
CONSUMER_NAME = "manager_1"


def process_prediction(redis_client, prediction: dict):
    """Process a prediction and update alarm state.
    
    Args:
        redis_client: Redis client instance
        prediction: Prediction message from stream
    """
    predicted_class = prediction["prediction"]
    frame_id = prediction["frame_id"]
    timestamp = prediction["timestamp"]
    
    current_status = get_current_status(redis_client)
    consecutive_frames = get_consecutive_drowsy_frames(redis_client)
    alarm_active = is_alarm_active(redis_client)
    
    logger.debug(
        f"Processing prediction for frame {frame_id}: "
        f"{predicted_class} (current: {current_status}, "
        f"consecutive: {consecutive_frames}, alarm: {alarm_active})"
    )
    
    # Update current status
    set_current_status(redis_client, predicted_class)
    
    if predicted_class == "drowsy":
        # Increment consecutive drowsy counter
        new_count = increment_consecutive_drowsy_frames(redis_client)
        
        logger.debug(f"Consecutive drowsy frames: {new_count}/{ALARM_THRESHOLD_FRAMES}")
        
        # Check if threshold exceeded
        if new_count >= ALARM_THRESHOLD_FRAMES and not alarm_active:
            # Activate alarm
            set_alarm_active(redis_client, True)
            publish_alarm_event(redis_client, "activated", time.time())
            logger.warning(
                f"ALARM ACTIVATED! Consecutive drowsy frames: {new_count} "
                f"(threshold: {ALARM_THRESHOLD_FRAMES})"
            )
    else:  # predicted_class == "alert"
        # Reset counter if driver is alert
        if consecutive_frames > 0:
            reset_consecutive_drowsy_frames(redis_client)
            logger.debug("Driver is alert - resetting consecutive drowsy counter")
        
        # Note: We don't automatically clear the alarm here
        # The alarm is cleared only when user clicks "I am awake" button


def main():
    """Main alarm manager loop."""
    logger.info("Starting alarm manager...")
    logger.info(f"Alarm threshold: {ALARM_THRESHOLD_FRAMES} consecutive drowsy frames")
    
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
    
    # Initialize state if needed
    if get_current_status(redis_client) is None:
        set_current_status(redis_client, "alert")
        reset_consecutive_drowsy_frames(redis_client)
        set_alarm_active(redis_client, False)
        logger.info("Initialized Redis state")
    
    logger.info(f"Consumer group: {CONSUMER_GROUP}, Consumer name: {CONSUMER_NAME}")
    logger.info("Waiting for predictions to process...")
    
    # Main processing loop
    while True:
        try:
            # Consume predictions from stream
            messages = consume_predictions(redis_client, CONSUMER_GROUP, CONSUMER_NAME, count=1)
            
            if not messages:
                time.sleep(0.1)  # Small delay if no messages
                continue
            
            for msg in messages:
                # Process prediction
                process_prediction(redis_client, msg)
                
                # Acknowledge message
                acknowledge_prediction(redis_client, CONSUMER_GROUP, msg["id"])
                
        except KeyboardInterrupt:
            logger.info("Stopping alarm manager...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(1)  # Wait before retrying


if __name__ == "__main__":
    main()

