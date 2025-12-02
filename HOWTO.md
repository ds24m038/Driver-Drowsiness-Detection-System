# HOWTO - Driver Drowsiness Detection System

Complete step-by-step guide for setting up and running the driver drowsiness detection system.

## Table of Contents

1. [Project Setup](#1-project-setup)
2. [Data Preparation](#2-data-preparation)
3. [Environment Configuration](#3-environment-configuration)
4. [Model Training](#4-model-training)
5. [Docker Setup](#5-docker-setup)
6. [Running the System](#6-running-the-system)
7. [Testing](#7-testing)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Project Setup

### 1.1 Clone/Download the Project

Ensure you have the project directory with the following structure:

```
Project/
├── Data/
│   ├── Drowsy/
│   └── Non Drowsy/
├── src/
├── notebooks/
├── docker/
└── ...
```

### 1.2 Install Prerequisites

**Required:**
- Python 3.9 or higher
- Docker and Docker Compose
- UV package manager (will be installed automatically in Docker)

**Optional (for local development):**
- Jupyter Notebook
- Virtual environment (venv or conda)

---

## 2. Data Preparation

The dataset should already be in the `Data/` directory:

- `Data/Drowsy/` - Contains drowsy driver face images (PNG, 227×227)
- `Data/Non Drowsy/` - Contains alert driver face images (PNG, 227×227)

**Verify data:**
```bash
# Count images
find Data/Drowsy -name "*.png" | wc -l
find Data/Non\ Drowsy -name "*.png" | wc -l
```

Expected: ~22,348 drowsy images, ~19,445 non-drowsy images.

---

## 3. Environment Configuration

### 3.1 Create `.env` File

The `.env` file should already exist in the project root. It must contain:

```env
WANDB_API_KEY=your_wandb_api_key_here
```

**To get your W&B API key:**
1. Sign up/login at https://wandb.ai
2. Go to Settings → API keys
3. Copy your API key
4. Add it to `.env`

### 3.2 Verify Configuration

Check `src/config/settings.py` for default settings:
- Alarm threshold: 10 seconds (50 frames at 5 FPS)
- Redis host/port: redis:6379 (Docker) or localhost:6379 (local)
- Model path: `models/best_model.pth`

---

## 4. Model Training

### 4.1 Install Dependencies (Local Development)

If running notebooks locally (not in Docker):

```bash
# Install UV
pip install uv

# Install dependencies
uv pip install -e .
```

Or use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt  # If you create one, or use UV
```

### 4.2 Run Data Exploration Notebook

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook:
- Explores dataset structure
- Shows image statistics
- Visualizes sample images

**Expected output:** Dataset statistics and sample visualizations.

### 4.3 Train the Model

```bash
jupyter notebook notebooks/02_model_training.ipynb
```

**Steps in the notebook:**
1. Initialize W&B project ("SDC Project Final")
2. Load and preprocess dataset
3. Split into train/validation/test (70/15/15)
4. Train PyTorch CNN model
5. Save best model to `models/best_model.pth`
6. Log metrics and artifacts to W&B

**Training parameters (configurable in notebook):**
- Learning rate: 0.001
- Batch size: 32
- Epochs: 20
- Optimizer: Adam

**Expected duration:** ~30-60 minutes on CPU, ~5-10 minutes on GPU.

**Verify model was saved:**
```bash
ls -lh models/best_model.pth
```

### 4.4 Test Model Inference (Optional)

```bash
jupyter notebook notebooks/03_inference_demo.ipynb
```

This notebook loads the trained model and runs inference on sample images.

---

## 5. Docker Setup

### 5.1 Build Docker Images

All Dockerfiles use UV as the package manager. Build images:

```bash
# Build all services
docker compose build

# Or build individually
docker build -f docker/Dockerfile.backend -t drowsiness-backend .
docker build -f docker/Dockerfile.frontend -t drowsiness-frontend .
docker build -f docker/Dockerfile.worker -t drowsiness-worker .
docker build -f docker/Dockerfile.alarm -t drowsiness-alarm .
```

### 5.2 Verify Docker Compose Configuration

Check `docker-compose.yml`:
- Services: `redis`, `api`, `inference_worker`, `alarm_manager`, `frontend`
- Ports: 8501 (frontend), 8000 (API), 6379 (Redis)
- Volumes: `models/` directory mounted for model access

---

## 6. Running the System

### 6.1 Start All Services

```bash
docker compose up
```

Or in detached mode:
```bash
docker compose up -d
```

**Expected output:**
- Redis starts and becomes healthy
- API starts and loads model
- Inference worker connects to Redis
- Alarm manager starts monitoring
- Frontend becomes available

### 6.2 Access the Streamlit UI

1. Open browser: http://localhost:8501
2. You should see the "Driver Drowsiness Detection System" interface

### 6.3 Using the System

1. **Start Detection:**
   - Click "▶️ Start Detection" button in sidebar
   - Allow webcam access when prompted

2. **Monitor Status:**
   - Green bounding box = Alert
   - Orange bounding box = Drowsy (not yet alarm)
   - Red bounding box = Alarm active

3. **Alarm Behavior:**
   - After 10 seconds of consecutive drowsiness:
     - Bounding box turns red
     - Visual warning appears
     - Audio alarm plays
     - Status shows "🚨 ALARM ACTIVE"

4. **Reset Alarm:**
   - Click "🙋 I am Awake" button
   - Alarm resets, counter clears
   - Monitoring continues

### 6.4 Stop the System

```bash
docker compose down
```

To also remove volumes:
```bash
docker compose down -v
```

---

## 7. Testing

### 7.1 Test Individual Components

**Test Redis:**
```bash
docker exec -it drowsiness_redis redis-cli ping
# Should return: PONG
```

**Test API:**
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy","message":"API is running and model is loaded"}
```

**Test API Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@Data/Drowsy/A0001.png"
```

### 7.2 Test Redis Streams

```bash
# Check frames stream
docker exec -it drowsiness_redis redis-cli XINFO STREAM frames_stream

# Check predictions stream
docker exec -it drowsiness_redis redis-cli XINFO STREAM predictions_stream

# Check alarm state
docker exec -it drowsiness_redis redis-cli GET alarm_active
docker exec -it drowsiness_redis redis-cli GET current_status
```

### 7.3 End-to-End Test

1. Start system: `docker compose up`
2. Open Streamlit UI
3. Start detection
4. Simulate drowsiness (close eyes, look tired)
5. Verify:
   - Frames are captured
   - Face is detected (green box)
   - Predictions are made
   - After 10 seconds, alarm activates
   - "I am awake" button resets alarm

---

## 8. Troubleshooting

### 8.1 Model Not Found

**Error:** `FileNotFoundError: Model file not found`

**Solution:**
- Train the model first using `notebooks/02_model_training.ipynb`
- Ensure `models/best_model.pth` exists
- Check model path in `src/config/settings.py`

### 8.2 Redis Connection Failed

**Error:** `Failed to connect to Redis`

**Solution:**
- Ensure Redis container is running: `docker compose ps`
- Check Redis host/port in environment variables
- Verify network connectivity: `docker network ls`

### 8.3 Webcam Not Working

**Error:** No webcam feed in Streamlit

**Solution:**
- Grant browser permissions for camera
- Check if webcam is available: `ls /dev/video*` (Linux)
- Try different browser (Chrome/Firefox)
- For Docker: may need to pass device access (see Docker docs)

### 8.4 Face Detection Not Working

**Error:** "No face detected"

**Solution:**
- Ensure good lighting
- Position face clearly in frame
- Check OpenCV Haar Cascade is loaded (should be cached)
- Verify OpenCV is installed correctly

### 8.5 Alarm Not Triggering

**Issue:** Drowsiness detected but alarm doesn't activate

**Solution:**
- Check consecutive drowsy frames counter in Redis
- Verify threshold: should be 50 frames (10 seconds at 5 FPS)
- Check alarm manager logs: `docker logs drowsiness_alarm`
- Verify predictions are being published to Redis Streams

### 8.6 W&B Integration Issues

**Error:** W&B login failed

**Solution:**
- Verify `WANDB_API_KEY` in `.env` file
- Check API key is valid at https://wandb.ai/settings
- Ensure project name is exactly "SDC Project Final"
- Check network connectivity

### 8.7 Docker Build Fails

**Error:** UV installation or dependency errors

**Solution:**
- Ensure Docker has internet access
- Check `pyproject.toml` for correct dependencies
- Try rebuilding: `docker compose build --no-cache`
- Check Docker logs: `docker compose logs`

### 8.8 Port Already in Use

**Error:** Port 8501 or 8000 already in use

**Solution:**
- Change ports in `docker-compose.yml`
- Or stop conflicting services:
  ```bash
  # Find process using port
  lsof -i :8501
  # Kill process
  kill <PID>
  ```

### 8.9 View Logs

```bash
# All services
docker compose logs

# Specific service
docker compose logs frontend
docker compose logs inference_worker
docker compose logs alarm_manager
docker compose logs api

# Follow logs
docker compose logs -f
```

---

## Additional Notes

### Performance Tuning

- **Frame Rate:** Adjust `FPS` in `src/config/settings.py` (default: 5)
- **Alarm Threshold:** Adjust `ALARM_THRESHOLD_SECONDS` (default: 10)
- **Batch Processing:** Inference worker processes one frame at a time (can be optimized)

### Development Mode

For local development without Docker:

1. Start Redis locally: `redis-server`
2. Run services separately:
   ```bash
   # Terminal 1: API
   uvicorn src.backend.main:app --reload
   
   # Terminal 2: Inference Worker
   python -m src.inference_worker.worker
   
   # Terminal 3: Alarm Manager
   python -m src.alarm_manager.manager
   
   # Terminal 4: Frontend
   streamlit run src/frontend/app.py
   ```

### Model Retraining

To retrain with different hyperparameters:

1. Modify config in `notebooks/02_model_training.ipynb`
2. Run training notebook
3. New model will be saved to `models/best_model.pth`
4. Restart Docker services to load new model

---

## Summary

This system provides a complete end-to-end ML solution for driver drowsiness detection. The architecture is modular, containerized, and ready for deployment. Follow the steps above to set up, train, and run the system successfully.

For questions or issues, refer to the logs and troubleshooting section above.

