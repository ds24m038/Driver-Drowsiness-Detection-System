"""Centralized configuration settings for the driver drowsiness detection system."""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Alarm threshold configuration
ALARM_THRESHOLD_SECONDS = int(os.getenv("ALARM_THRESHOLD_SECONDS", "10"))
FPS = int(os.getenv("FPS", "5"))  # Frames per second for threshold calculation
ALARM_THRESHOLD_FRAMES = FPS * ALARM_THRESHOLD_SECONDS  # 50 frames at 5 FPS

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Redis Stream keys
FRAMES_STREAM = "frames_stream"
PREDICTIONS_STREAM = "predictions_stream"
ALARM_STREAM = "alarm_stream"

# Redis state keys
CURRENT_STATUS_KEY = "current_status"
CONSECUTIVE_DROWSY_FRAMES_KEY = "consecutive_drowsy_frames"
ALARM_ACTIVE_KEY = "alarm_active"

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "best_model.pth"))
MODEL_INPUT_SIZE = (227, 227)  # Width, Height
NUM_CLASSES = 2
CLASS_NAMES = ["alert", "drowsy"]  # Index 0: alert, Index 1: drowsy

# Prediction threshold - lower value = more pessimistic (more likely to predict drowsy)
# 0.15 means if drowsy probability > 0.15, classify as drowsy (even if alert is higher)
# This makes the model very pessimistic - it will predict drowsy even with low confidence
DROWSY_THRESHOLD = float(os.getenv("DROWSY_THRESHOLD", "0.15"))  # Default: 15% drowsy probability triggers drowsy

# Weights & Biases configuration
WANDB_PROJECT = "SDC Project Final"
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")

# FastAPI configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Streamlit configuration
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "0.0.0.0")
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

