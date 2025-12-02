"""Streamlit frontend for driver drowsiness detection with continuous video streaming."""
import time
import uuid
import logging
from typing import Optional, Tuple

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

from src.config.redis_utils import (
    get_redis_client,
    publish_frame,
    get_current_status,
    is_alarm_active,
    get_consecutive_drowsy_frames,
    reset_alarm_state,
)
from src.config.settings import ALARM_THRESHOLD_FRAMES, FPS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Load face detection cascade
@st.cache_resource
def load_face_cascade():
    """Load OpenCV Haar Cascade for face detection."""
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        return cv2.CascadeClassifier(cascade_path)
    except Exception as e:
        logger.error(f"Error loading face cascade: {e}")
        return None


def detect_face(image: np.ndarray, face_cascade) -> Optional[Tuple[int, int, int, int]]:
    """Detect face in image and return bounding box."""
    if face_cascade is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest_face)
    return None


def draw_bounding_box(image: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int], thickness: int = 3) -> np.ndarray:
    """Draw bounding box on image."""
    x, y, w, h = bbox
    image_copy = image.copy()
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
    return image_copy


def image_to_bytes(image: np.ndarray) -> bytes:
    """Convert numpy image array to bytes."""
    _, buffer = cv2.imencode(".jpg", image)
    return buffer.tobytes()


class VideoProcessor(VideoProcessorBase):
    """Video processor for continuous frame processing."""
    
    def __init__(self):
        self.face_cascade = load_face_cascade()
        self.last_frame_time = 0
        self.frame_count = 0
        self.redis_client = None
        try:
            from src.config.redis_utils import get_redis_client
            self.redis_client = get_redis_client()
        except Exception as e:
            logger.error(f"Failed to get Redis client: {e}")
    
    def recv(self, frame):
        """Process each video frame."""
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Detect face
            bbox = detect_face(img, self.face_cascade)
            
            # Get alarm state
            try:
                from src.config.redis_utils import get_current_status, is_alarm_active
                alarm_active = is_alarm_active(self.redis_client) if self.redis_client else False
                current_status = get_current_status(self.redis_client) if self.redis_client else "alert"
            except:
                alarm_active = False
                current_status = "alert"
            
            # Choose bounding box color
            if alarm_active:
                bbox_color = (0, 0, 255)  # Red
            elif current_status == "drowsy":
                bbox_color = (0, 165, 255)  # Orange
            else:
                bbox_color = (0, 255, 0)  # Green
            
            # Draw bounding box if face detected
            if bbox:
                img = draw_bounding_box(img, bbox, bbox_color, thickness=3)
                
                # Extract face region for inference
                x, y, w, h = bbox
                face_roi = img[y:y+h, x:x+w]
                
                # Publish frame to Redis Streams (throttle to ~5 FPS)
                current_time = time.time()
                if current_time - self.last_frame_time >= 1.0 / FPS:
                    if self.redis_client:
                        try:
                            from src.config.redis_utils import publish_frame
                            frame_id = str(uuid.uuid4())
                            frame_bytes = image_to_bytes(face_roi)
                            publish_frame(
                                self.redis_client,
                                frame_id,
                                frame_bytes,
                                current_time
                            )
                            self.last_frame_time = current_time
                            self.frame_count += 1
                            logger.info(f"Published frame {frame_id} to Redis (count: {self.frame_count})")
                        except Exception as e:
                            logger.error(f"Error publishing frame: {e}", exc_info=True)
            
            # Convert back to video frame
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            return frame


def main():
    """Main Streamlit application."""
    st.title("🚗 Driver Drowsiness Detection System")
    st.markdown("---")
    
    # Initialize session state
    if "redis_client" not in st.session_state:
        try:
            st.session_state.redis_client = get_redis_client()
            st.session_state.redis_client.ping()
            st.session_state.redis_connected = True
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
            st.session_state.redis_connected = False
            st.error(f"❌ Failed to connect to Redis: {e}")
            st.stop()
    
    if "detection_active" not in st.session_state:
        st.session_state.detection_active = False
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        if st.button("▶️ Start Detection", type="primary", use_container_width=True):
            st.session_state.detection_active = True
            st.rerun()
        
        if st.button("⏹️ Stop Detection", use_container_width=True):
            st.session_state.detection_active = False
            st.rerun()
        
        st.markdown("---")
        st.header("Status")
        
        # Get current status from Redis
        try:
            current_status = get_current_status(st.session_state.redis_client)
            alarm_active = is_alarm_active(st.session_state.redis_client)
            consecutive_frames = get_consecutive_drowsy_frames(st.session_state.redis_client)
            
            if current_status == "drowsy":
                st.error(f"⚠️ Status: **DROWSY**")
            else:
                st.success(f"✅ Status: **ALERT**")
            
            st.info(f"Consecutive drowsy frames: {consecutive_frames}/{ALARM_THRESHOLD_FRAMES}")
            
            if alarm_active:
                st.error("🚨 **ALARM ACTIVE**")
            else:
                st.success("✅ No alarm")
                
        except Exception as e:
            st.error(f"Error reading status: {e}")
            logger.error(f"Error reading status: {e}", exc_info=True)
        
        st.markdown("---")
        
        if st.button("🙋 I am Awake", type="primary", use_container_width=True):
            try:
                reset_alarm_state(st.session_state.redis_client)
                st.success("✅ Alarm reset!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Error resetting alarm: {e}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📹 Live Video Feed")
        
        if st.session_state.detection_active:
            st.success("🟢 Detection ACTIVE - Video streaming and processing continuously")
        else:
            st.info("⚪ Detection INACTIVE - Click 'Start Detection' to begin")
        
        # WebRTC video streamer for continuous video
        webrtc_ctx = webrtc_streamer(
            key="video",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )
        
        if webrtc_ctx.state.playing:
            st.success("✅ Video stream active - Processing frames continuously")
            
            # Display status based on Redis state
            try:
                current_status = get_current_status(st.session_state.redis_client)
                alarm_active = is_alarm_active(st.session_state.redis_client)
                
                if alarm_active:
                    st.error("## 🚨 ALARM ACTIVE - Please Wake Up!")
                elif current_status == "drowsy":
                    st.warning("## ⚠️ DROWSY - Stay Alert!")
                else:
                    st.success("## ✅ ALERT - You're awake!")
            except Exception as e:
                logger.error(f"Error getting status: {e}", exc_info=True)
        else:
            st.info("📹 Click 'START' above to begin video streaming")
    
    with col2:
        st.header("📊 System Information")
        
        st.subheader("Configuration")
        st.write(f"**Alarm Threshold:** {ALARM_THRESHOLD_FRAMES} frames ({ALARM_THRESHOLD_FRAMES / FPS:.1f} seconds)")
        st.write(f"**Frame Rate:** {FPS} FPS")
        
        st.subheader("Redis Connection")
        if st.session_state.redis_connected:
            st.success("✅ Connected")
        else:
            st.error("❌ Disconnected")
        
        st.markdown("---")
        st.subheader("How It Works")
        st.markdown("""
        1. Click **▶️ Start Detection**
        2. Click **START** on the video player
        3. Allow camera access
        4. **Video streams continuously** - no manual clicks needed!
        5. Frames are processed automatically via Redis
        6. After 10 seconds of drowsiness, alarm activates
        """)
        
        st.markdown("---")
        st.subheader("Status Indicators")
        st.markdown("""
        - 🟢 **Green box** = Alert
        - 🟠 **Orange box** = Drowsy
        - 🔴 **Red box** = Alarm active
        """)


if __name__ == "__main__":
    main()
