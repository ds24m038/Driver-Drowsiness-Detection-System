"""Alarm manager that tracks drowsiness and manages alarm state."""
import logging
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.redis_client import get_redis_client
from src.config.settings import (
    CONSUMER_GROUP_ALARM,
    STREAM_PREDICTIONS,
    STREAM_ALARM,
    ALARM_THRESHOLD_FRAMES,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlarmManager:
    """Manages alarm state based on consecutive drowsy predictions."""
    
    def __init__(self, consumer_name: str = "alarm-manager-1"):
        """Initialize the alarm manager."""
        self.consumer_name = consumer_name
        self.redis_client = get_redis_client()
        self.alarm_active = False
    
    def process_prediction(self, prediction: str):
        """
        Process a prediction and update alarm state.
        
        Args:
            prediction: "alert" or "drowsy"
        """
        if prediction == "drowsy":
            # Increment consecutive drowsy frames counter
            count = self.redis_client.increment_drowsy_frames()
            logger.debug(f"Drowsy frame detected. Count: {count}/{ALARM_THRESHOLD_FRAMES}")
            
            # Check if threshold reached
            if count >= ALARM_THRESHOLD_FRAMES and not self.alarm_active:
                self.activate_alarm()
        else:
            # Reset counter on alert
            current_count = self.redis_client.get_drowsy_frames()
            if current_count > 0:
                self.redis_client.reset_drowsy_frames()
                logger.debug("Alert detected, resetting drowsy frames counter")
            
            # If alarm is active and we see alert, we keep alarm active until user resets
            # (This is by design - user must click "I am awake" button)
    
    def activate_alarm(self):
        """Activate the alarm."""
        self.alarm_active = True
        self.redis_client.set_alarm_active(True)
        self.redis_client.publish_alarm_event("activated")
        logger.warning("ALARM ACTIVATED - Driver drowsiness detected for threshold duration!")
    
    def deactivate_alarm(self):
        """Deactivate the alarm."""
        self.alarm_active = False
        self.redis_client.set_alarm_active(False)
        self.redis_client.reset_drowsy_frames()
        self.redis_client.publish_alarm_event("cleared")
        logger.info("Alarm deactivated and counters reset")
    
    def check_reset_signal(self):
        """Check if alarm reset was signaled by user."""
        if self.redis_client.check_reset_alarm():
            if self.alarm_active:
                self.deactivate_alarm()
            else:
                # Reset counters even if alarm wasn't active
                self.redis_client.reset_drowsy_frames()
                logger.info("Reset signal received, counters cleared")
    
    def run(self):
        """Main alarm manager loop - continuously process predictions."""
        logger.info(f"Starting alarm manager: {self.consumer_name}")
        logger.info(f"Alarm threshold: {ALARM_THRESHOLD_FRAMES} consecutive drowsy frames")
        logger.info(f"Reading from stream: {STREAM_PREDICTIONS}")
        
        while True:
            try:
                # Check for reset signal
                self.check_reset_signal()
                
                # Read predictions from stream
                messages = self.redis_client.read_predictions(
                    consumer_group=CONSUMER_GROUP_ALARM,
                    consumer_name=self.consumer_name,
                    count=1,
                    block=1000  # Block for 1 second
                )
                
                for msg in messages:
                    msg_id = msg["id"]
                    data = msg["data"]
                    
                    prediction = data.get("prediction", "")
                    frame_id = data.get("frame_id", "unknown")
                    
                    if not prediction:
                        logger.warning(f"Empty prediction for frame {frame_id}")
                        self.redis_client.ack_prediction(CONSUMER_GROUP_ALARM, msg_id)
                        continue
                    
                    try:
                        # Process prediction
                        self.process_prediction(prediction)
                        
                        # Acknowledge message
                        self.redis_client.ack_prediction(CONSUMER_GROUP_ALARM, msg_id)
                        
                        # Log status periodically
                        count = self.redis_client.get_drowsy_frames()
                        if count > 0:
                            logger.debug(
                                f"Frame {frame_id}: {prediction} | "
                                f"Drowsy count: {count}/{ALARM_THRESHOLD_FRAMES} | "
                                f"Alarm: {'ACTIVE' if self.alarm_active else 'inactive'}"
                            )
                    
                    except Exception as e:
                        logger.error(f"Error processing prediction for frame {frame_id}: {e}")
                        # Still acknowledge to avoid reprocessing
                        self.redis_client.ack_prediction(CONSUMER_GROUP_ALARM, msg_id)
            
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in alarm manager loop: {e}")
                time.sleep(1)  # Wait before retrying


def main():
    """Main entry point for the alarm manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Alarm Manager for Driver Drowsiness Detection")
    parser.add_argument(
        "--consumer-name",
        type=str,
        default=f"alarm-manager-{int(time.time())}",
        help="Consumer name for Redis Streams"
    )
    
    args = parser.parse_args()
    
    try:
        manager = AlarmManager(consumer_name=args.consumer_name)
        manager.run()
    except Exception as e:
        logger.error(f"Alarm manager failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

