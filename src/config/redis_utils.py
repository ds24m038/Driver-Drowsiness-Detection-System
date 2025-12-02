"""Redis utilities for stream operations and state management."""
import json
import time
import redis
from typing import Optional, Dict, Any
from src.config.settings import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    FRAMES_STREAM,
    PREDICTIONS_STREAM,
    ALARM_STREAM,
    CURRENT_STATUS_KEY,
    CONSECUTIVE_DROWSY_FRAMES_KEY,
    ALARM_ACTIVE_KEY,
)


def get_redis_client() -> redis.Redis:
    """Create and return a Redis client."""
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=False,  # Keep binary for image data
    )


def publish_frame(redis_client: redis.Redis, frame_id: str, frame_data: bytes, timestamp: float) -> str:
    """Publish a frame to the frames stream.
    
    Args:
        redis_client: Redis client instance
        frame_id: Unique frame identifier
        frame_data: Binary image data
        timestamp: Unix timestamp
        
    Returns:
        Message ID from Redis Stream
    """
    message = {
        "frame_id": frame_id.encode(),
        "timestamp": str(timestamp).encode(),
        "frame_data": frame_data,
    }
    return redis_client.xadd(FRAMES_STREAM, message)


def consume_frames(redis_client: redis.Redis, consumer_group: str, consumer_name: str, count: int = 1) -> list:
    """Consume frames from the frames stream.
    
    Args:
        redis_client: Redis client instance
        consumer_group: Consumer group name
        consumer_name: Consumer name
        count: Number of messages to read
        
    Returns:
        List of messages from the stream
    """
    try:
        # Create consumer group if it doesn't exist
        redis_client.xgroup_create(FRAMES_STREAM, consumer_group, id="0", mkstream=True)
    except redis.exceptions.ResponseError:
        # Group already exists
        pass
    
    messages = redis_client.xreadgroup(
        consumer_group,
        consumer_name,
        {FRAMES_STREAM: ">"},
        count=count,
        block=1000,  # Block for 1 second
    )
    
    if not messages:
        return []
    
    # Parse messages
    result = []
    for stream, msgs in messages:
        for msg_id, data in msgs:
            result.append({
                "id": msg_id,
                "frame_id": data[b"frame_id"].decode(),
                "timestamp": float(data[b"timestamp"].decode()),
                "frame_data": data[b"frame_data"],
            })
    
    return result


def acknowledge_frame(redis_client: redis.Redis, consumer_group: str, message_id: str):
    """Acknowledge a processed frame message.
    
    Args:
        redis_client: Redis client instance
        consumer_group: Consumer group name
        message_id: Message ID to acknowledge
    """
    redis_client.xack(FRAMES_STREAM, consumer_group, message_id)


def publish_prediction(
    redis_client: redis.Redis,
    frame_id: str,
    prediction: str,
    confidence: float,
    timestamp: float,
) -> str:
    """Publish a prediction to the predictions stream.
    
    Args:
        redis_client: Redis client instance
        frame_id: Frame identifier
        prediction: Predicted class ("alert" or "drowsy")
        confidence: Confidence score
        timestamp: Unix timestamp
        
    Returns:
        Message ID from Redis Stream
    """
    message = {
        "frame_id": frame_id.encode(),
        "prediction": prediction.encode(),
        "confidence": str(confidence).encode(),
        "timestamp": str(timestamp).encode(),
    }
    return redis_client.xadd(PREDICTIONS_STREAM, message)


def consume_predictions(redis_client: redis.Redis, consumer_group: str, consumer_name: str, count: int = 1) -> list:
    """Consume predictions from the predictions stream.
    
    Args:
        redis_client: Redis client instance
        consumer_group: Consumer group name
        consumer_name: Consumer name
        count: Number of messages to read
        
    Returns:
        List of prediction messages
    """
    try:
        redis_client.xgroup_create(PREDICTIONS_STREAM, consumer_group, id="0", mkstream=True)
    except redis.exceptions.ResponseError:
        pass
    
    messages = redis_client.xreadgroup(
        consumer_group,
        consumer_name,
        {PREDICTIONS_STREAM: ">"},
        count=count,
        block=1000,
    )
    
    if not messages:
        return []
    
    result = []
    for stream, msgs in messages:
        for msg_id, data in msgs:
            result.append({
                "id": msg_id,
                "frame_id": data[b"frame_id"].decode(),
                "prediction": data[b"prediction"].decode(),
                "confidence": float(data[b"confidence"].decode()),
                "timestamp": float(data[b"timestamp"].decode()),
            })
    
    return result


def acknowledge_prediction(redis_client: redis.Redis, consumer_group: str, message_id: str):
    """Acknowledge a processed prediction message."""
    redis_client.xack(PREDICTIONS_STREAM, consumer_group, message_id)


def publish_alarm_event(redis_client: redis.Redis, event_type: str, timestamp: float) -> str:
    """Publish an alarm event to the alarm stream.
    
    Args:
        redis_client: Redis client instance
        event_type: Event type ("activated" or "cleared")
        timestamp: Unix timestamp
        
    Returns:
        Message ID from Redis Stream
    """
    message = {
        "event_type": event_type.encode(),
        "timestamp": str(timestamp).encode(),
    }
    return redis_client.xadd(ALARM_STREAM, message)


def get_current_status(redis_client: redis.Redis) -> str:
    """Get the current driver status from Redis.
    
    Returns:
        Current status ("alert" or "drowsy"), defaults to "alert"
    """
    status = redis_client.get(CURRENT_STATUS_KEY)
    if status:
        return status.decode() if isinstance(status, bytes) else status
    return "alert"


def set_current_status(redis_client: redis.Redis, status: str):
    """Set the current driver status in Redis.
    
    Args:
        redis_client: Redis client instance
        status: Status to set ("alert" or "drowsy")
    """
    redis_client.set(CURRENT_STATUS_KEY, status)


def get_consecutive_drowsy_frames(redis_client: redis.Redis) -> int:
    """Get the consecutive drowsy frames counter.
    
    Returns:
        Number of consecutive drowsy frames
    """
    frames = redis_client.get(CONSECUTIVE_DROWSY_FRAMES_KEY)
    if frames:
        return int(frames.decode() if isinstance(frames, bytes) else frames)
    return 0


def set_consecutive_drowsy_frames(redis_client: redis.Redis, count: int):
    """Set the consecutive drowsy frames counter.
    
    Args:
        redis_client: Redis client instance
        count: Number of consecutive drowsy frames
    """
    redis_client.set(CONSECUTIVE_DROWSY_FRAMES_KEY, str(count))


def increment_consecutive_drowsy_frames(redis_client: redis.Redis) -> int:
    """Increment the consecutive drowsy frames counter.
    
    Returns:
        New counter value
    """
    return redis_client.incr(CONSECUTIVE_DROWSY_FRAMES_KEY)


def reset_consecutive_drowsy_frames(redis_client: redis.Redis):
    """Reset the consecutive drowsy frames counter to 0."""
    redis_client.set(CONSECUTIVE_DROWSY_FRAMES_KEY, "0")


def is_alarm_active(redis_client: redis.Redis) -> bool:
    """Check if the alarm is currently active.
    
    Returns:
        True if alarm is active, False otherwise
    """
    active = redis_client.get(ALARM_ACTIVE_KEY)
    if active:
        return active.decode() == "true" if isinstance(active, bytes) else active == "true"
    return False


def set_alarm_active(redis_client: redis.Redis, active: bool):
    """Set the alarm active flag.
    
    Args:
        redis_client: Redis client instance
        active: True to activate alarm, False to deactivate
    """
    redis_client.set(ALARM_ACTIVE_KEY, "true" if active else "false")


def reset_alarm_state(redis_client: redis.Redis):
    """Reset all alarm-related state in Redis."""
    set_current_status(redis_client, "alert")
    reset_consecutive_drowsy_frames(redis_client)
    set_alarm_active(redis_client, False)
    publish_alarm_event(redis_client, "cleared", time.time())

