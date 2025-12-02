"""Redis client utilities for stream and state management."""
import redis
import json
import logging
from typing import Optional, Dict, Any, List
import time

from src.config.settings import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REDIS_PASSWORD,
    STREAM_FRAMES,
    STREAM_PREDICTIONS,
    STREAM_ALARM,
    KEY_CURRENT_STATUS,
    KEY_CONSECUTIVE_DROWSY_FRAMES,
    KEY_ALARM_ACTIVE,
    KEY_RESET_ALARM,
    CONSUMER_GROUP_INFERENCE,
    CONSUMER_GROUP_ALARM,
)

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis client wrapper for stream and state operations."""
    
    def __init__(self, host=None, port=None, db=None, password=None):
        """Initialize Redis connection."""
        self.host = host or REDIS_HOST
        self.port = port or REDIS_PORT
        self.db = db or REDIS_DB
        self.password = password or REDIS_PASSWORD
        
        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # Test connection
        try:
            self.client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def ensure_consumer_group(self, stream_key: str, group_name: str):
        """Ensure consumer group exists for a stream."""
        try:
            self.client.xgroup_create(
                name=stream_key,
                groupname=group_name,
                id="0",
                mkstream=True
            )
            logger.info(f"Created consumer group {group_name} for stream {stream_key}")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group {group_name} already exists for {stream_key}")
            else:
                raise
    
    # Stream Operations
    
    def publish_frame(self, frame_id: str, frame_data: bytes, timestamp: Optional[float] = None):
        """
        Publish a frame to the frames stream.
        
        Args:
            frame_id: Unique frame identifier
            frame_data: Encoded image bytes (base64 or raw)
            timestamp: Optional timestamp, defaults to current time
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Encode frame data as base64 for storage
        import base64
        frame_b64 = base64.b64encode(frame_data).decode('utf-8')
        
        message = {
            "frame_id": frame_id,
            "frame_data": frame_b64,
            "timestamp": str(timestamp)
        }
        
        self.client.xadd(STREAM_FRAMES, message)
        logger.debug(f"Published frame {frame_id} to {STREAM_FRAMES}")
    
    def read_frames(self, consumer_group: str, consumer_name: str, count: int = 1, block: int = 1000):
        """
        Read frames from the frames stream using consumer group.
        
        Args:
            consumer_group: Consumer group name
            consumer_name: Consumer name
            count: Maximum number of messages to read
            block: Blocking time in milliseconds
            
        Returns:
            List of messages
        """
        self.ensure_consumer_group(STREAM_FRAMES, consumer_group)
        
        messages = self.client.xreadgroup(
            groupname=consumer_group,
            consumername=consumer_name,
            streams={STREAM_FRAMES: ">"},
            count=count,
            block=block
        )
        
        result = []
        for stream, msgs in messages:
            for msg_id, data in msgs:
                result.append({
                    "id": msg_id,
                    "data": data
                })
        
        return result
    
    def ack_frame(self, consumer_group: str, message_id: str):
        """Acknowledge processing of a frame message."""
        self.client.xack(STREAM_FRAMES, consumer_group, message_id)
    
    def publish_prediction(
        self,
        frame_id: str,
        prediction: str,
        confidence: float,
        probabilities: Dict[str, float],
        timestamp: Optional[float] = None
    ):
        """
        Publish a prediction to the predictions stream.
        
        Args:
            frame_id: Frame identifier
            prediction: "alert" or "drowsy"
            confidence: Confidence score
            probabilities: Dict of class probabilities
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        message = {
            "frame_id": frame_id,
            "prediction": prediction,
            "confidence": str(confidence),
            "probabilities": json.dumps(probabilities),
            "timestamp": str(timestamp)
        }
        
        self.client.xadd(STREAM_PREDICTIONS, message)
        logger.debug(f"Published prediction for frame {frame_id}: {prediction}")
    
    def read_predictions(self, consumer_group: str, consumer_name: str, count: int = 1, block: int = 1000):
        """
        Read predictions from the predictions stream.
        
        Args:
            consumer_group: Consumer group name
            consumer_name: Consumer name
            count: Maximum number of messages to read
            block: Blocking time in milliseconds
            
        Returns:
            List of prediction messages
        """
        self.ensure_consumer_group(STREAM_PREDICTIONS, consumer_group)
        
        messages = self.client.xreadgroup(
            groupname=consumer_group,
            consumername=consumer_name,
            streams={STREAM_PREDICTIONS: ">"},
            count=count,
            block=block
        )
        
        result = []
        for stream, msgs in messages:
            for msg_id, data in msgs:
                # Parse probabilities JSON
                if "probabilities" in data:
                    data["probabilities"] = json.loads(data["probabilities"])
                result.append({
                    "id": msg_id,
                    "data": data
                })
        
        return result
    
    def ack_prediction(self, consumer_group: str, message_id: str):
        """Acknowledge processing of a prediction message."""
        self.client.xack(STREAM_PREDICTIONS, consumer_group, message_id)
    
    def publish_alarm_event(self, event_type: str, timestamp: Optional[float] = None):
        """
        Publish an alarm event to the alarm stream.
        
        Args:
            event_type: "activated" or "cleared"
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        message = {
            "event_type": event_type,
            "timestamp": str(timestamp)
        }
        
        self.client.xadd(STREAM_ALARM, message)
        logger.info(f"Published alarm event: {event_type}")
    
    # State Operations
    
    def set_current_status(self, status: str):
        """Set the current driver status (alert or drowsy)."""
        if status not in ["alert", "drowsy"]:
            raise ValueError(f"Invalid status: {status}. Must be 'alert' or 'drowsy'")
        
        self.client.set(KEY_CURRENT_STATUS, status)
        logger.debug(f"Set current status to: {status}")
    
    def get_current_status(self) -> Optional[str]:
        """Get the current driver status."""
        status = self.client.get(KEY_CURRENT_STATUS)
        return status
    
    def increment_drowsy_frames(self) -> int:
        """Increment the consecutive drowsy frames counter."""
        count = self.client.incr(KEY_CONSECUTIVE_DROWSY_FRAMES)
        logger.debug(f"Incremented drowsy frames counter to: {count}")
        return count
    
    def reset_drowsy_frames(self):
        """Reset the consecutive drowsy frames counter to 0."""
        self.client.set(KEY_CONSECUTIVE_DROWSY_FRAMES, 0)
        logger.debug("Reset drowsy frames counter")
    
    def get_drowsy_frames(self) -> int:
        """Get the current consecutive drowsy frames count."""
        count = self.client.get(KEY_CONSECUTIVE_DROWSY_FRAMES)
        return int(count) if count else 0
    
    def set_alarm_active(self, active: bool):
        """Set the alarm active flag."""
        self.client.set(KEY_ALARM_ACTIVE, "true" if active else "false")
        logger.info(f"Set alarm active to: {active}")
    
    def get_alarm_active(self) -> bool:
        """Get the alarm active flag."""
        active = self.client.get(KEY_ALARM_ACTIVE)
        return active == "true" if active else False
    
    def signal_reset_alarm(self):
        """Signal that the alarm should be reset."""
        self.client.set(KEY_RESET_ALARM, "true")
        logger.info("Signaled alarm reset")
    
    def check_reset_alarm(self) -> bool:
        """Check if alarm reset was signaled and clear it."""
        reset = self.client.get(KEY_RESET_ALARM)
        if reset == "true":
            self.client.set(KEY_RESET_ALARM, "false")
            return True
        return False
    
    def reset_all_alarm_state(self):
        """Reset all alarm-related state."""
        self.reset_drowsy_frames()
        self.set_alarm_active(False)
        self.set_current_status("alert")
        logger.info("Reset all alarm state")


# Global Redis client instance
_redis_client: Optional[RedisClient] = None


def get_redis_client() -> RedisClient:
    """Get or create the global Redis client instance."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client

