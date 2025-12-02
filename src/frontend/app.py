"""Streamlit frontend for driver drowsiness detection with continuous video streaming."""
import time
import uuid
import logging
from typing import Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
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
    set_current_status,
    set_alarm_active,
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

# WebRTC configuration with multiple STUN servers for better connectivity
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }
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
            
            # Initialize alarm state if not set (ensure it defaults to False)
            if get_current_status(st.session_state.redis_client) is None:
                set_current_status(st.session_state.redis_client, "alert")
                set_alarm_active(st.session_state.redis_client, False)
                logger.info("Initialized Redis alarm state to default (alert, alarm=False)")
            else:
                # Double-check alarm state is valid - if it's somehow invalid, reset to False
                alarm_state = is_alarm_active(st.session_state.redis_client)
                if alarm_state is None:
                    set_alarm_active(st.session_state.redis_client, False)
                    logger.info("Reset invalid alarm state to False")
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
        
        # Get current status from Redis - force fresh read
        try:
            # Force fresh read (no caching)
            current_status = get_current_status(st.session_state.redis_client)
            alarm_active = is_alarm_active(st.session_state.redis_client)
            consecutive_frames = get_consecutive_drowsy_frames(st.session_state.redis_client)
            
            # Update session state for auto-refresh
            if "last_alarm_state" not in st.session_state:
                st.session_state.last_alarm_state = alarm_active
            
            # Auto-refresh if alarm state changes or if alarm is active (poll every 1 second)
            if st.session_state.last_alarm_state != alarm_active:
                st.session_state.last_alarm_state = alarm_active
                time.sleep(0.1)
                st.rerun()
            elif alarm_active:
                # If alarm is active, refresh every 1 second to keep UI updated
                if "last_refresh" not in st.session_state:
                    st.session_state.last_refresh = time.time()
                elif time.time() - st.session_state.last_refresh > 1.0:
                    st.session_state.last_refresh = time.time()
                    st.rerun()
            
            if alarm_active:
                st.error("🚨 **ALARM ACTIVE**")
            elif current_status == "drowsy":
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
            async_processing=True,
        )
        
        # Display status based on Redis state - force refresh by reading directly
        # Check alarm status regardless of video playing state
        try:
            # Force fresh read from Redis (no caching)
            current_status = get_current_status(st.session_state.redis_client)
            alarm_active = is_alarm_active(st.session_state.redis_client)
            
            # Ensure we have valid defaults - alarm should be False by default
            if alarm_active is None:
                alarm_active = False
            if current_status is None:
                current_status = "alert"
            
            # Debug: Show actual values
            logger.info(f"Status check - alarm_active: {alarm_active}, current_status: {current_status}")
            
            # Play audio alarm if alarm is active - show this regardless of video state
            # ONLY show alarm if it's explicitly True
            if alarm_active is True:
                st.error("## 🚨 ALARM ACTIVE - Please Wake Up!")
                # Use JavaScript to play beep sound continuously
                alarm_js = """
                <script>
                    (function() {
                        var audioContext = null;
                        var alarmInterval = null;
                        
                        // Clear any existing interval
                        if (window.currentAlarmInterval) {
                            clearInterval(window.currentAlarmInterval);
                            window.currentAlarmInterval = null;
                        }
                        
                        function playBeep() {
                            try {
                                if (!audioContext) {
                                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                                }
                                
                                var oscillator = audioContext.createOscillator();
                                var gainNode = audioContext.createGain();
                                
                                oscillator.connect(gainNode);
                                gainNode.connect(audioContext.destination);
                                
                                oscillator.frequency.value = 800;
                                oscillator.type = 'sine';
                                
                                gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
                                
                                oscillator.start(audioContext.currentTime);
                                oscillator.stop(audioContext.currentTime + 0.3);
                            } catch(e) {
                                console.log('Audio error:', e);
                            }
                        }
                        
                        // Play beep immediately and then every 0.5 seconds
                        playBeep();
                        window.currentAlarmInterval = setInterval(playBeep, 500);
                    })();
                </script>
                """
                components.html(alarm_js, height=0)
                
                # Auto-refresh page every 0.5 seconds when alarm is active
                time.sleep(0.5)
                st.rerun()
            elif current_status == "drowsy":
                st.warning("## ⚠️ DROWSY - Stay Alert!")
            else:
                st.success("## ✅ ALERT - You're awake!")
        except Exception as e:
            logger.error(f"Error getting status: {e}", exc_info=True)
            # On error, default to alert state (not alarm!)
            st.warning("⚠️ Could not read status from Redis. Defaulting to ALERT state.")
            st.success("## ✅ ALERT - You're awake!")
        
        if webrtc_ctx.state.playing:
            st.success("✅ Video stream active - Processing frames continuously")
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
