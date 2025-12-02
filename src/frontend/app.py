"""Streamlit frontend for driver drowsiness detection."""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import io
import time
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.redis_client import get_redis_client
from src.config.settings import (
    FPS,
    KEY_CURRENT_STATUS,
    KEY_ALARM_ACTIVE,
    KEY_CONSECUTIVE_DROWSY_FRAMES,
    ALARM_THRESHOLD_FRAMES,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

# Initialize session state
if 'streaming' not in st.session_state:
    st.session_state.streaming = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'last_frame_time' not in st.session_state:
    st.session_state.last_frame_time = time.time()

# Initialize Redis client
try:
    redis_client = get_redis_client()
    redis_connected = True
except Exception as e:
    st.error(f"Failed to connect to Redis: {e}")
    redis_connected = False
    redis_client = None


def detect_face(image):
    """
    Detect face in image using OpenCV Haar Cascade.
    
    Returns:
        (x, y, w, h) bounding box or None if no face detected
    """
    # Load Haar Cascade classifier
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    
    if len(faces) > 0:
        # Return the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        return largest_face
    
    return None


def draw_face_rectangle(image, face_bbox, color=(0, 255, 0), thickness=3):
    """Draw rectangle around detected face."""
    if face_bbox is None:
        return image
    
    x, y, w, h = face_bbox
    image_with_rect = image.copy()
    cv2.rectangle(image_with_rect, (x, y), (x + w, y + h), color, thickness)
    return image_with_rect


def encode_image_to_base64(image):
    """Encode PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def publish_frame_to_redis(image, frame_id):
    """Publish frame to Redis Streams."""
    if not redis_connected:
        return False
    
    try:
        # Convert PIL to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()
        
        # Publish to Redis
        redis_client.publish_frame(
            frame_id=frame_id,
            frame_data=img_bytes,
            timestamp=time.time()
        )
        return True
    except Exception as e:
        logger.error(f"Error publishing frame: {e}")
        return False


def get_status_from_redis():
    """Get current status and alarm state from Redis."""
    if not redis_connected:
        return "unknown", False, 0
    
    try:
        status = redis_client.get_current_status() or "unknown"
        alarm_active = redis_client.get_alarm_active()
        drowsy_count = redis_client.get_drowsy_frames()
        return status, alarm_active, drowsy_count
    except Exception as e:
        logger.error(f"Error reading from Redis: {e}")
        return "unknown", False, 0


def reset_alarm():
    """Reset alarm state in Redis."""
    if not redis_connected:
        return False
    
    try:
        redis_client.signal_reset_alarm()
        redis_client.reset_all_alarm_state()
        return True
    except Exception as e:
        logger.error(f"Error resetting alarm: {e}")
        return False


def play_alarm_sound():
    """Play alarm sound using audio element."""
    # Generate a simple beep sound using HTML5 audio
    # In Streamlit, we'll use st.audio with a data URI
    pass  # Will be handled in the UI


# Main UI
st.title("🚗 Driver Drowsiness Detection System")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    if redis_connected:
        st.success("✅ Redis Connected")
    else:
        st.error("❌ Redis Disconnected")
    
    st.markdown("---")
    
    # Start/Stop streaming
    if st.button("🎥 Start Monitoring", disabled=not redis_connected):
        st.session_state.streaming = True
        st.rerun()
    
    if st.button("⏹️ Stop Monitoring"):
        st.session_state.streaming = False
        st.rerun()
    
    st.markdown("---")
    
    # "I am awake" button
    if st.button("✅ I am awake", type="primary", use_container_width=True):
        if reset_alarm():
            st.success("Alarm reset! Monitoring continues...")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Failed to reset alarm. Check Redis connection.")
    
    st.markdown("---")
    st.info("**Instructions:**\n\n1. Click 'Start Monitoring' to begin\n2. Allow webcam access\n3. System will detect drowsiness\n4. Click 'I am awake' to reset alarm")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Webcam Feed")
    
    # Webcam input
    camera_input = st.camera_input(
        "Camera",
        disabled=not st.session_state.streaming,
        key="camera"
    )
    
    if camera_input and st.session_state.streaming:
        # Convert to numpy array
        image = Image.open(camera_input)
        image_np = np.array(image)
        
        # Detect face
        face_bbox = detect_face(image_np)
        
        # Get status from Redis
        status, alarm_active, drowsy_count = get_status_from_redis()
        
        # Determine rectangle color
        if alarm_active:
            rect_color = (255, 0, 0)  # Red
            status_text = "🚨 DROWSY - ALARM ACTIVE"
        elif status == "drowsy":
            rect_color = (255, 165, 0)  # Orange
            status_text = "⚠️ Drowsy"
        else:
            rect_color = (0, 255, 0)  # Green
            status_text = "✅ Alert"
        
        # Draw rectangle
        if face_bbox:
            image_with_rect = draw_face_rectangle(image_np, face_bbox, color=rect_color)
        else:
            image_with_rect = image_np
            status_text = "👤 No face detected"
        
        # Display image
        st.image(image_with_rect, use_container_width=True, channels="RGB")
        
        # Publish frame to Redis (throttle to FPS)
        current_time = time.time()
        time_since_last = current_time - st.session_state.last_frame_time
        frame_interval = 1.0 / FPS
        
        if time_since_last >= frame_interval:
            frame_id = f"frame_{int(time.time() * 1000)}"
            publish_frame_to_redis(image, frame_id)
            st.session_state.last_frame_time = current_time
            st.session_state.frame_count += 1

with col2:
    st.subheader("📊 Status")
    
    # Get current status
    status, alarm_active, drowsy_count = get_status_from_redis()
    
    # Status display
    if alarm_active:
        st.error(f"## 🚨 ALARM ACTIVE")
        st.error("**Driver drowsiness detected!**\n\nPlease click 'I am awake' to reset.")
        
        # Visual alarm banner
        st.markdown(
            """
            <div style='background-color: #ff0000; padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;'>
                <h2 style='color: white; margin: 0;'>⚠️ WARNING ⚠️</h2>
                <p style='color: white; font-size: 18px; margin: 10px 0;'>DRIVER DROWSINESS DETECTED</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Audio alarm (using HTML5 audio)
        st.markdown(
            """
            <audio autoplay loop>
                <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIGGW57+OcTQ8OUKfk8LZjHAY4kdfyzHksBSR3x/DdkEAKFF606euoVRQKRp/g8r5sIQUrgc7y2Yk2CBhlue/jnE0PDlCn5PC2YxwGOJHX8sx5LAUkd8fw3ZBAC" type="audio/wav">
            </audio>
            """,
            unsafe_allow_html=True
        )
    elif status == "drowsy":
        st.warning(f"## ⚠️ Drowsy")
        st.info(f"Drowsy frames: {drowsy_count}/{ALARM_THRESHOLD_FRAMES}")
    elif status == "alert":
        st.success(f"## ✅ Alert")
        st.info("Driver is alert and attentive.")
    else:
        st.info("## ⏳ Waiting...")
        st.info("Waiting for predictions...")
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📈 Statistics")
    st.metric("Frames Processed", st.session_state.frame_count)
    st.metric("Drowsy Frame Count", f"{drowsy_count}/{ALARM_THRESHOLD_FRAMES}")
    st.metric("Alarm Threshold", f"{ALARM_THRESHOLD_FRAMES} frames")
    
    # Progress bar for drowsy frames
    if drowsy_count > 0:
        progress = min(drowsy_count / ALARM_THRESHOLD_FRAMES, 1.0)
        st.progress(progress)
        st.caption(f"Progress to alarm: {int(progress * 100)}%")

# Footer
st.markdown("---")
st.caption("Driver Drowsiness Detection System | Powered by PyTorch, FastAPI, Redis Streams, and Streamlit")

