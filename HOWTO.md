# How-To Guide - Driver Drowsiness Detection System

This guide provides detailed step-by-step instructions for setting up and running the complete driver drowsiness detection system.

## Table of Contents

1. [Project Setup](#1-project-setup)
2. [Environment Configuration](#2-environment-configuration)
3. [Data Preparation](#3-data-preparation)
4. [Model Training](#4-model-training)
5. [Docker Setup](#5-docker-setup)
6. [Running the System](#6-running-the-system)
7. [Testing and Validation](#7-testing-and-validation)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Project Setup

### 1.1 Clone/Create Project Structure

The project structure should already be in place. Verify the following directories exist:

```bash
.
├── Data/
│   ├── Drowsy/
│   └── Non Drowsy/
├── notebooks/
├── src/
├── docker/
├── models/
└── static/
```

### 1.2 Install Python Dependencies

For development (including Jupyter notebooks):

```bash
pip install -r requirements-dev.txt
```

For production (Docker containers):

```bash
pip install -r requirements.txt
```

---

## 2. Environment Configuration

### 2.1 Create `.env` File

Create a `.env` file in the project root (copy from `.env.example` if available):

```bash
# Weights & Biases Configuration
WANDB_API_KEY=your_wandb_api_key_here

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Alarm Configuration
ALARM_THRESHOLD_SECONDS=10
FPS=5

# Model Configuration
MODEL_PATH=models/best_model.pth

# Service Configuration
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_HOST=0.0.0.0
STREAMLIT_PORT=8501

# Logging
LOG_LEVEL=INFO
```

### 2.2 Get W&B API Key

1. Sign up/login at https://wandb.ai
2. Go to Settings → API Keys
3. Copy your API key
4. Add it to `.env` file

---

## 3. Data Preparation

### 3.1 Verify Dataset

The dataset should already be in the `Data/` directory:

```
Data/
├── Drowsy/          # Drowsy driver images
└── Non Drowsy/      # Alert driver images
```

### 3.2 Explore Dataset (Optional)

Run the data exploration notebook:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This will show:
- Number of images in each class
- Image size statistics
- Sample visualizations

---

## 4. Model Training

### 4.1 Open Training Notebook

```bash
jupyter notebook notebooks/02_model_training.ipynb
```

### 4.2 Configure W&B

Ensure your `.env` file has the correct `WANDB_API_KEY`. The notebook will:
- Login to W&B
- Create/use project: `SDC Project Final`
- Log all metrics and artifacts

### 4.3 Run Training

Execute all cells in the notebook. The training process will:

1. Load and split the dataset (70% train, 15% val, 15% test)
2. Initialize the CNN model
3. Train for specified epochs (default: 10)
4. Log metrics to W&B:
   - Loss, accuracy, precision, recall, F1
   - Training and validation metrics
   - Learning rate
5. Save the best model to `models/best_model.pth`
6. Evaluate on test set
7. Log confusion matrix

### 4.4 Verify Model

After training, verify the model file exists:

```bash
ls -lh models/best_model.pth
```

### 4.5 (Optional) Hyperparameter Tuning

You can create a W&B sweep for hyperparameter tuning. See W&B documentation for sweep configuration.

---

## 5. Docker Setup

### 5.1 Verify Docker Installation

```bash
docker --version
docker compose version
```

### 5.2 Build Docker Images

Build all images:

```bash
docker compose build
```

Or build individual services:

```bash
# Backend
docker build -f docker/Dockerfile.backend -t drowsiness-backend .

# Frontend
docker build -f docker/Dockerfile.frontend -t drowsiness-frontend .

# Inference Worker
docker build -f docker/Dockerfile.inference_worker -t drowsiness-inference-worker .

# Alarm Manager
docker build -f docker/Dockerfile.alarm_manager -t drowsiness-alarm-manager .
```

### 5.3 Verify Docker Compose Configuration

Check `docker-compose.yml` configuration:

```bash
docker compose config
```

---

## 6. Running the System

### 6.1 Start All Services

Start all services with Docker Compose:

```bash
docker compose up --build
```

Or in detached mode:

```bash
docker compose up -d --build
```

### 6.2 Verify Services Are Running

Check running containers:

```bash
docker compose ps
```

You should see:
- `drowsiness_redis`
- `drowsiness_backend`
- `drowsiness_inference_worker`
- `drowsiness_alarm_manager`
- `drowsiness_frontend`

### 6.3 Check Service Logs

View logs for all services:

```bash
docker compose logs -f
```

View logs for specific service:

```bash
docker compose logs -f frontend
docker compose logs -f backend
docker compose logs -f inference_worker
docker compose logs -f alarm_manager
```

### 6.4 Access Services

- **Streamlit Frontend**: http://localhost:8501
- **FastAPI Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Redis**: localhost:6379 (if exposed)

---

## 7. Testing and Validation

### 7.1 Test FastAPI Backend

Health check:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": 1234567890.123
}
```

### 7.2 Test Redis Connection

```bash
docker exec -it drowsiness_redis redis-cli ping
```

Expected: `PONG`

### 7.3 Test Redis Streams

Check if streams exist:

```bash
docker exec -it drowsiness_redis redis-cli XINFO STREAM frames_stream
docker exec -it drowsiness_redis redis-cli XINFO STREAM predictions_stream
```

### 7.4 End-to-End Testing

1. **Open Streamlit UI**: http://localhost:8501

2. **Start Monitoring**:
   - Click "Start Monitoring" button
   - Allow webcam access when prompted

3. **Verify Face Detection**:
   - Green rectangle should appear around detected face
   - Status should show "Alert" or "Drowsy"

4. **Verify Frame Processing**:
   - Check "Frames Processed" counter increases
   - Check Redis streams have messages:
     ```bash
     docker exec -it drowsiness_redis redis-cli XREAD COUNT 1 STREAMS frames_stream 0
     ```

5. **Test Alarm Trigger** (Simulation):
   - To test alarm, you can manually set drowsy state in Redis:
     ```bash
     docker exec -it drowsiness_redis redis-cli SET consecutive_drowsy_frames 50
     docker exec -it drowsiness_redis redis-cli SET alarm_active true
     ```
   - Refresh Streamlit UI to see alarm

6. **Test Reset Button**:
   - Click "I am awake" button
   - Verify alarm state resets
   - Check Redis:
     ```bash
     docker exec -it drowsiness_redis redis-cli GET alarm_active
     docker exec -it drowsiness_redis redis-cli GET consecutive_drowsy_frames
     ```

### 7.5 Monitor System Behavior

Watch logs in real-time:

```bash
docker compose logs -f
```

Look for:
- Frame processing messages
- Prediction results
- Alarm activation/deactivation
- Any error messages

---

## 8. Troubleshooting

### 8.1 Model Not Found

**Problem**: Services fail with "Model file not found"

**Solution**:
1. Ensure model is trained: `ls models/best_model.pth`
2. Check model path in `.env`: `MODEL_PATH=models/best_model.pth`
3. Verify volume mount in `docker-compose.yml`

### 8.2 Redis Connection Errors

**Problem**: Services can't connect to Redis

**Solution**:
1. Check Redis is running: `docker compose ps redis`
2. Verify Redis logs: `docker compose logs redis`
3. Check network: `docker network ls`
4. Verify environment variables: `REDIS_HOST=redis`

### 8.3 Webcam Not Working

**Problem**: Streamlit can't access webcam

**Solution**:
1. Grant camera permissions in browser
2. Check browser console for errors
3. Try different browser (Chrome recommended)
4. Verify webcam is not used by another application

### 8.4 Services Not Starting

**Problem**: Docker containers exit immediately

**Solution**:
1. Check logs: `docker compose logs [service_name]`
2. Verify dependencies in `requirements.txt`
3. Check Dockerfile syntax
4. Ensure ports are not in use: `lsof -i :8501`

### 8.5 No Predictions Appearing

**Problem**: Status stays "Waiting..." in UI

**Solution**:
1. Check inference worker logs: `docker compose logs inference_worker`
2. Verify frames are being published: Check `frames_stream` in Redis
3. Check model is loaded in worker logs
4. Verify Redis Streams consumer groups exist

### 8.6 Alarm Not Triggering

**Problem**: Drowsy state detected but alarm doesn't activate

**Solution**:
1. Check alarm manager logs: `docker compose logs alarm_manager`
2. Verify threshold: `ALARM_THRESHOLD_FRAMES=50` (10 seconds × 5 FPS)
3. Check consecutive counter: `docker exec -it drowsiness_redis redis-cli GET consecutive_drowsy_frames`
4. Verify alarm manager is reading predictions stream

### 8.7 Performance Issues

**Problem**: System is slow or laggy

**Solution**:
1. Reduce FPS: Set `FPS=3` in `.env`
2. Reduce frame processing rate in frontend
3. Check system resources: `docker stats`
4. Consider using GPU for inference (modify Dockerfiles)

---

## 9. Stopping the System

### 9.1 Stop All Services

```bash
docker compose down
```

### 9.2 Stop and Remove Volumes

```bash
docker compose down -v
```

### 9.3 Stop Individual Service

```bash
docker compose stop frontend
```

---

## 10. Additional Resources

- **W&B Dashboard**: https://wandb.ai/[your-entity]/SDC%20Project%20Final
- **FastAPI Docs**: http://localhost:8000/docs
- **Redis CLI**: `docker exec -it drowsiness_redis redis-cli`

---

## Notes

- The system requires a trained model before running inference services
- Webcam access is required for the frontend to work
- Redis must be running before other services start
- All services communicate via Redis Streams and state keys
- The alarm threshold is configurable via environment variables

