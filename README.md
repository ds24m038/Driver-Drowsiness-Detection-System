# Driver Drowsiness Detection System

A complete end-to-end machine learning solution for detecting driver drowsiness using real-time webcam analysis, PyTorch CNN models, FastAPI backend, Streamlit frontend, Redis Streams for inter-process communication, and Weights & Biases for experiment tracking.

## Overview

This system monitors a driver's state in real-time through webcam footage and triggers visual and audio alarms when drowsiness is detected for a sustained period (10 seconds). The architecture is fully containerized with Docker and can be deployed with a single command.

## System Architecture

```
┌─────────────┐
│  Streamlit  │  Webcam Capture → Face Detection → Frame Publishing
│  Frontend   │──────────────────────────────────────────────────┐
└─────────────┘                                                  │
                                                                  ▼
                                                          ┌──────────────┐
                                                          │    Redis     │
                                                          │   Streams    │
                                                          └──────────────┘
                                                                  │
                    ┌────────────────────────────────────────────┼─────────────────────────────┐
                    │                                            │                             │
                    ▼                                            ▼                             ▼
          ┌─────────────────┐                        ┌──────────────────┐          ┌─────────────────┐
          │ Inference Worker│                        │  Alarm Manager  │          │  FastAPI Backend│
          │                 │                        │                  │          │                 │
          │ Consumes frames │                        │ Tracks drowsiness│          │ Model Serving   │
          │ Runs inference  │                        │ Manages alarms   │          │ REST API        │
          │ Publishes preds │                        │ Updates state    │          │                 │
          └─────────────────┘                        └──────────────────┘          └─────────────────┘
```

### Components

1. **FastAPI Backend** (`src/backend/`)
   - REST API for model serving
   - Health check and prediction endpoints
   - Loads PyTorch CNN model at startup

2. **Inference Worker** (`src/inference_worker/`)
   - Consumes frames from Redis Streams
   - Runs inference using the trained model
   - Publishes predictions to Redis Streams

3. **Alarm Manager** (`src/alarm_manager/`)
   - Consumes predictions from Redis Streams
   - Tracks consecutive drowsy frames
   - Manages alarm state (activates after 10 seconds of drowsiness)

4. **Streamlit Frontend** (`src/frontend/`)
   - Webcam capture and face detection (OpenCV Haar Cascades)
   - Real-time visualization with bounding boxes
   - Displays driver status and triggers alarms
   - "I am awake" button to reset alarm state

5. **Redis** (Official Docker image)
   - Central state management
   - Redis Streams for async message passing
   - State keys for current status, counters, and alarm flags

6. **Training Notebooks** (`notebooks/`)
   - Data exploration
   - Model training with W&B integration
   - Inference demos

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Weights & Biases API key (for automatic model download)
- OR a trained model file at `models/best_model.pth` (see HOWTO.md for training instructions)

### Running the System

1. **Set up environment variables:**
   ```bash
   # Create a .env file with your W&B API key
   echo "WANDB_API_KEY=your_wandb_api_key_here" > .env
   
   # Optional: Specify W&B project and artifact version
   echo "WANDB_PROJECT=Driver-Drowsiness-Training" >> .env
   echo "WANDB_ARTIFACT_VERSION=latest" >> .env
   ```
   
   Get your W&B API key from: https://wandb.ai/authorize

2. **Start all services:**
   ```bash
   docker compose up --build
   ```
   
   The system will automatically:
   - Download the model from W&B if not found locally
   - Fall back to local model if available
   - Use the downloaded model for inference

3. **Access the Streamlit UI:**
   - Open your browser to `http://localhost:8501`
   - Click "Start Detection"
   - Position your face in front of the webcam
   - The system will monitor your state and trigger alarms if drowsy

### Service Endpoints

- **Streamlit Frontend**: http://localhost:8501
- **FastAPI Backend**: http://localhost:8000
  - Health: http://localhost:8000/health
  - API Docs: http://localhost:8000/docs
- **Redis**: localhost:6379

## Configuration

Key configuration parameters (in `src/config/settings.py`):

- `ALARM_THRESHOLD_SECONDS = 10` - Seconds of consecutive drowsiness before alarm
- `FPS = 5` - Frames per second for threshold calculation
- `ALARM_THRESHOLD_FRAMES = 50` - Frames threshold (FPS × seconds)

### Model Loading

The system supports automatic model loading from Weights & Biases:

- **Automatic Download**: If the model file is not found locally, the system will automatically download it from W&B
- **Environment Variables**:
  - `WANDB_API_KEY` - Your W&B API key (required for download)
  - `WANDB_PROJECT` - W&B project name (default: "Driver-Drowsiness-Training")
  - `WANDB_ARTIFACT_VERSION` - Artifact version to download (default: "latest")
- **Artifact Name**: The system looks for the artifact named `drowsiness_detection_model`
- **Fallback**: If W&B download fails, the system will look for a local model file

This means you don't need to commit large model files to GitHub - they're stored in W&B and downloaded automatically when needed.

## Technology Stack

- **ML Framework**: PyTorch
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Message Queue**: Redis with Redis Streams
- **Experiment Tracking**: Weights & Biases (W&B)
- **Containerization**: Docker & Docker Compose
- **Package Manager**: UV

## Project Structure

```
.
├── Data/                    # Driver Drowsiness Dataset
│   ├── Drowsy/
│   └── Non Drowsy/
├── notebooks/               # Jupyter notebooks for training
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_inference_demo.ipynb
├── src/
│   ├── backend/            # FastAPI application
│   ├── frontend/           # Streamlit application
│   ├── inference_worker/   # Inference service
│   ├── alarm_manager/      # Alarm logic service
│   └── config/             # Shared configuration
├── docker/                 # Dockerfiles
├── models/                 # Trained model storage
├── docker-compose.yml      # Service orchestration
├── pyproject.toml         # Dependencies (UV)
├── README.md              # This file
└── HOWTO.md               # Detailed setup guide
```

## Documentation

- **README.md** (this file) - Overview and quickstart
- **HOWTO.md** - Step-by-step setup and usage instructions
- **Approach.md** - Complete project specification


