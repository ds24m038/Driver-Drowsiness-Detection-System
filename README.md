# Driver Drowsiness Detection System

A complete end-to-end machine learning solution for detecting driver drowsiness in real-time using webcam footage. The system uses a PyTorch CNN model, FastAPI backend, Redis Streams for inter-process communication, and a Streamlit frontend.

## 🎯 Overview

This system monitors a driver's face through a webcam and classifies their state as either **alert** or **drowsy**. If drowsiness is detected continuously for 10 seconds (50 frames at 5 FPS), the system triggers visual and audio alarms to alert the driver.

## 🏗️ Architecture

```
┌─────────────┐
│  Streamlit  │  Webcam Capture & Face Detection
│  Frontend  │  ────────────────────────────────┐
└─────────────┘                                   │
                                                  ▼
┌─────────────┐                              ┌─────────┐
│   Redis     │◄─────────────────────────────┤ frames  │
│   Streams   │                              │ stream  │
└─────────────┘                              └─────────┘
       │                                              │
       │                                              │
       ▼                                              │
┌─────────────┐                              ┌─────────┐
│  Inference  │◄─────────────────────────────┤         │
│   Worker    │                              │         │
└─────────────┘                              └─────────┘
       │
       │ predictions_stream
       ▼
┌─────────────┐
│   Alarm     │  Tracks consecutive drowsy frames
│  Manager    │  Activates alarm when threshold reached
└─────────────┘
       │
       │ alarm_stream
       ▼
┌─────────────┐
│   Redis     │  State Management
│   State     │  - current_status
│             │  - consecutive_drowsy_frames
│             │  - alarm_active
└─────────────┘
```

### Components

1. **Model Training** (Jupyter Notebooks)
   - PyTorch CNN for binary classification
   - Weights & Biases integration for experiment tracking
   - Hyperparameter tuning support

2. **FastAPI Backend**
   - REST API for model inference
   - `/health` - Health check endpoint
   - `/predict` - Image prediction endpoint

3. **Inference Worker**
   - Reads frames from Redis Streams
   - Processes frames with CNN model
   - Publishes predictions to Redis

4. **Alarm Manager**
   - Monitors predictions stream
   - Tracks consecutive drowsy frames
   - Manages alarm activation (10-second threshold)

5. **Streamlit Frontend**
   - Webcam capture
   - Face detection (OpenCV Haar Cascade)
   - Real-time status display
   - Visual and audio alarms
   - "I am awake" reset button

6. **Redis**
   - Central state management
   - Redis Streams for message passing
   - Real-time coordination between components

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Trained model file (`models/best_model.pth`)
- W&B API key (for training, optional for inference)

### 1. Train the Model

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run training notebook
jupyter notebook notebooks/02_model_training.ipynb
```

### 2. Start the System

```bash
# Start all services
docker compose up --build

# Or in detached mode
docker compose up -d --build
```

### 3. Access the UI

Open your browser and navigate to:
- **Streamlit Frontend**: http://localhost:8501
- **FastAPI Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📋 Usage

1. **Start Monitoring**: Click "Start Monitoring" in the Streamlit UI
2. **Allow Webcam Access**: Grant camera permissions when prompted
3. **Monitor Status**: Watch the real-time status and drowsy frame counter
4. **Alarm Activation**: If drowsy for 10+ seconds, alarm will activate
5. **Reset Alarm**: Click "I am awake" button to reset the alarm state

## 🔧 Configuration

Key configuration parameters (set in `.env` or environment variables):

- `ALARM_THRESHOLD_SECONDS=10` - Seconds of drowsiness before alarm
- `FPS=5` - Frames per second for processing
- `REDIS_HOST=redis` - Redis server hostname
- `REDIS_PORT=6379` - Redis server port
- `MODEL_PATH=models/best_model.pth` - Path to trained model

## 📁 Project Structure

```
.
├── Data/                    # Dataset (Drowsy/Non Drowsy images)
├── notebooks/               # Jupyter notebooks for training
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_inference_demo.ipynb
│   └── 04_wandb_reporting.ipynb
├── src/
│   ├── backend/            # FastAPI application
│   ├── frontend/           # Streamlit application
│   ├── inference_worker/   # Inference worker service
│   ├── alarm_manager/      # Alarm management service
│   ├── models/             # PyTorch model definition
│   └── config/             # Configuration and Redis utilities
├── docker/                 # Dockerfiles
├── models/                 # Trained model storage
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Production dependencies
└── requirements-dev.txt   # Development dependencies
```

## 🧪 Testing

### Individual Component Testing

```bash
# Test FastAPI backend
curl http://localhost:8000/health

# Test Redis connection
docker exec -it drowsiness_redis redis-cli ping
```

### End-to-End Testing

1. Start all services: `docker compose up`
2. Open Streamlit UI: http://localhost:8501
3. Start monitoring and verify:
   - Webcam capture works
   - Face detection shows bounding box
   - Predictions appear in status
   - Alarm triggers after threshold
   - "I am awake" button resets state

## 📊 Weights & Biases

The project uses W&B for experiment tracking:

- **Project**: `SDC Project Final`
- Logs metrics, hyperparameters, and model artifacts
- Supports hyperparameter sweeps
- See `notebooks/04_wandb_reporting.ipynb` for details

## 🐳 Docker Services

- `redis` - Redis server (port 6379)
- `backend` - FastAPI service (port 8000)
- `inference_worker` - Inference processing service
- `alarm_manager` - Alarm management service
- `frontend` - Streamlit UI (port 8501)

## 📝 Documentation

- **README.md** - This file (overview and quickstart)
- **HOWTO.md** - Detailed step-by-step instructions
- **Notebooks** - Training and exploration guides

## 🔍 Troubleshooting

### Model Not Found
- Ensure `models/best_model.pth` exists
- Train the model using `notebooks/02_model_training.ipynb`

### Redis Connection Errors
- Verify Redis container is running: `docker ps`
- Check Redis logs: `docker logs drowsiness_redis`

### Webcam Not Working
- Grant camera permissions in browser
- Check if webcam is available: `ls /dev/video*` (Linux)

### Services Not Starting
- Check Docker logs: `docker compose logs [service_name]`
- Verify all dependencies are installed
- Ensure ports are not already in use

## 📄 License

This project is part of the "Solution Deployment & Communication" course assignment.

## 👥 Authors

Course project implementation.

