"""Configuration settings for the driver drowsiness detection system."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Alarm Configuration
ALARM_THRESHOLD_SECONDS = int(os.getenv("ALARM_THRESHOLD_SECONDS", "10"))
FPS = int(os.getenv("FPS", "5"))  # Frames per second
ALARM_THRESHOLD_FRAMES = FPS * ALARM_THRESHOLD_SECONDS  # 50 frames for 10 seconds at 5 FPS

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Redis Stream Keys
STREAM_FRAMES = "frames_stream"
STREAM_PREDICTIONS = "predictions_stream"
STREAM_ALARM = "alarm_stream"

# Redis State Keys
KEY_CURRENT_STATUS = "current_status"
KEY_CONSECUTIVE_DROWSY_FRAMES = "consecutive_drowsy_frames"
KEY_ALARM_ACTIVE = "alarm_active"
KEY_RESET_ALARM = "reset_alarm"  # Signal for alarm reset

# Consumer Group Names
CONSUMER_GROUP_INFERENCE = "inference_workers"
CONSUMER_GROUP_ALARM = "alarm_managers"

# Model Configuration
MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "best_model.pth"))
MODEL_INPUT_SIZE = (227, 227)  # Width, Height
MODEL_NUM_CLASSES = 2
MODEL_CLASSES = ["alert", "drowsy"]

# Weights & Biases Configuration
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "SDC Project Final")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")

# FastAPI Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Streamlit Configuration
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "0.0.0.0")
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# Data Configuration
DATA_DIR = PROJECT_ROOT / "Data"
DROWSY_DIR = DATA_DIR / "Drowsy"
NON_DROWSY_DIR = DATA_DIR / "Non Drowsy"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

