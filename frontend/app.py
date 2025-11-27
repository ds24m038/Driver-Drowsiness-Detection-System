import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import redis
import av
import cv2
import numpy as np
import os
import time
import pandas as pd
from datetime import datetime

# --- Config ---
st.set_page_config(page_title="Drowsiness Monitor", layout="wide")
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')

# Initialize Redis connection
@st.cache_resource
def get_redis_client():
    return redis.Redis(host=REDIS_HOST, port=6379, db=0)

redis_client = get_redis_client()

# --- Page 1: The Live Monitor (Your original code) ---
class FrameProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_prediction = "Initializing..."
        self.last_confidence = 0.0
        # Load Face Cascade
        try:
            # Try loading from local file if downloaded, else use cv2 internal path
            if os.path.exists('haarcascade_frontalface_default.xml'):
                self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            else:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            self.face_cascade = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Fetch latest prediction non-blocking
        try:
            # Read only the very last entry
            last_entry = redis_client.xrevrange("prediction_stream", count=1)
            if last_entry:
                _, data = last_entry[0]
                self.last_prediction = data[b'class'].decode('utf-8')
                self.last_confidence = float(data[b'confidence'])
        except Exception:
            pass 

        # 2. Visual Overlays
        if self.face_cascade:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            color = (0, 0, 255) if self.last_prediction == "Drowsy" else (0, 255, 0)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
                label = f"{self.last_prediction} ({self.last_confidence:.2f})"
                cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 3. Send to Backend (Every 5th frame)
        if self.frame_count % 5 == 0:
            _, buffer = cv2.imencode('.jpg', img)
            redis_client.xadd("camera_stream", {"frame_id": self.frame_count, "data": buffer.tobytes()}, maxlen=100)
            
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def run_monitor():
    st.title("😴 Live Drowsiness Monitor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        webrtc_streamer(
            key="drowsiness-cam",
            video_processor_factory=FrameProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )

    with col2:
        st.subheader("Real-time Status")
        placeholder = st.empty()
        
        # Polling loop for the UI status box
        while True:
            try:
                last_entry = redis_client.xrevrange("prediction_stream", count=1)
                if last_entry:
                    _, data = last_entry[0]
                    state = data[b'class'].decode('utf-8')
                    conf = float(data[b'confidence'])
                    
                    if state == "Drowsy":
                        placeholder.error(f"⚠️ DROWSY ({conf:.1%})")
                    else:
                        placeholder.success(f"✅ AWAKE ({conf:.1%})")
            except:
                placeholder.info("Waiting for stream...")
            
            time.sleep(0.5)

# --- Page 2: The Redis Logger (New) ---
def run_debugger():
    st.title("🔍 System Inspector (Redis Backlog)")
    st.markdown("View the raw data flowing through the message broker.")

    # Refresh Button
    if st.button("🔄 Refresh Logs"):
        st.rerun()

    # Get Stream Statistics
    try:
        cam_len = redis_client.xlen("camera_stream")
        pred_len = redis_client.xlen("prediction_stream")
    except:
        cam_len = 0
        pred_len = 0
        st.error("Could not connect to Redis.")

    col1, col2 = st.columns(2)
    col1.metric("Camera Stream Queue", f"{cam_len} frames")
    col2.metric("Prediction Stream Queue", f"{pred_len} results")

    st.divider()

    # --- Prediction Log ---
    st.subheader("📝 Prediction History (Backend Output)")
    try:
        # Fetch last 20 entries. XRANGE returns [(id, {data}), ...]
        raw_preds = redis_client.xrevrange("prediction_stream", count=20)
        
        pred_data = []
        for msg_id, fields in raw_preds:
            # Timestamp calculation from Redis ID (Unix ms)
            ts = int(msg_id.decode().split('-')[0]) / 1000.0
            time_str = datetime.fromtimestamp(ts).strftime('%H:%M:%S.%f')[:-3]
            
            pred_data.append({
                "Redis ID": msg_id.decode(),
                "Time": time_str,
                "Prediction": fields[b'class'].decode(),
                "Confidence": f"{float(fields[b'confidence']):.4f}"
            })
            
        if pred_data:
            st.dataframe(pd.DataFrame(pred_data), width=800, use_container_width=True)
        else:
            st.info("No predictions found in stream.")

    except Exception as e:
        st.error(f"Error reading prediction stream: {e}")

    st.divider()

    # --- Camera Log ---
    st.subheader("📷 Camera Stream History (Frontend Output)")
    try:
        # Fetch last 10 entries
        raw_imgs = redis_client.xrevrange("camera_stream", count=10)
        
        img_data = []
        for msg_id, fields in raw_imgs:
            ts = int(msg_id.decode().split('-')[0]) / 1000.0
            time_str = datetime.fromtimestamp(ts).strftime('%H:%M:%S.%f')[:-3]
            
            # Calculate image size in KB
            img_bytes = fields.get(b'data', b'')
            size_kb = len(img_bytes) / 1024
            
            img_data.append({
                "Redis ID": msg_id.decode(),
                "Time": time_str,
                "Frame ID": int(fields.get(b'frame_id', 0)),
                "Size (KB)": f"{size_kb:.2f} KB"
            })
            
        if img_data:
            st.dataframe(pd.DataFrame(img_data), use_container_width=True)
        else:
            st.info("No camera frames found in stream.")
            
    except Exception as e:
        st.error(f"Error reading camera stream: {e}")
        
    # Dangerous Zone
    st.divider()
    with st.expander("⚠️ Danger Zone"):
        if st.button("Flush All Redis Data"):
            redis_client.flushall()
            st.toast("Redis flushed!", icon="🧹")
            time.sleep(1)
            st.rerun()

# --- Main Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Live Monitor", "Redis Logger"])

if page == "Live Monitor":
    run_monitor()
else:
    run_debugger()