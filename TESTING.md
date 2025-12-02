# System Testing Guide

## Training Complete! ✅

**Model Training Results:**
- Best Validation Accuracy: **99.98%**
- Model saved: `models/best_model.pth` (300MB)
- W&B Project: [Driver-Drowsiness-Training](https://wandb.ai/ds24m038-fh-technikum-wien/Driver-Drowsiness-Training)
- Model Artifact: Uploaded to W&B

---

## Next Steps: Testing the System

### 1. Start Docker Desktop

Make sure Docker Desktop is running on your Mac:
```bash
# Check if Docker is running
docker ps
```

If you get an error, start Docker Desktop from Applications.

### 2. Build Docker Images

```bash
cd "/Users/sinahaghgoo/Library/CloudStorage/OneDrive-Persönlich/Solution Deployment/Project"
docker compose build
```

This will build all services:
- `api` - FastAPI backend
- `frontend` - Streamlit UI
- `inference_worker` - Inference service
- `alarm_manager` - Alarm logic service
- `redis` - Redis server (uses official image)

### 3. Start All Services

```bash
docker compose up
```

Or in detached mode:
```bash
docker compose up -d
```

### 4. Access the System

- **Streamlit Frontend**: http://localhost:8501
- **FastAPI Backend**: http://localhost:8000
  - Health check: http://localhost:8000/health
  - API docs: http://localhost:8000/docs
- **Redis**: localhost:6379

### 5. Test the System

#### 5.1 Test Backend API

```bash
# Health check
curl http://localhost:8000/health

# Test prediction (using a sample image)
curl -X POST "http://localhost:8000/predict" \
  -F "file=@Data/Drowsy/A0001.png"
```

#### 5.2 Test Redis

```bash
# Connect to Redis
docker exec -it drowsiness_redis redis-cli

# Check streams
XINFO STREAM frames_stream
XINFO STREAM predictions_stream

# Check state
GET current_status
GET alarm_active
GET consecutive_drowsy_frames
```

#### 5.3 Test Frontend

1. Open http://localhost:8501 in your browser
2. Click "▶️ Start" button
3. Allow webcam access
4. Position your face in front of the camera
5. Verify:
   - Face detection (green bounding box)
   - Status updates in sidebar
   - Frame publishing to Redis Streams
   - Predictions being made
   - Alarm triggers after 10 seconds of drowsiness

#### 5.4 Test Alarm System

1. Simulate drowsiness (close eyes, look tired)
2. Wait for 10 seconds (50 frames at 5 FPS)
3. Verify:
   - Bounding box turns red
   - Visual alarm appears
   - Status shows "🚨 ALARM ACTIVE"
4. Click "🙋 I am Awake" button
5. Verify:
   - Alarm resets
   - Counter clears
   - Bounding box turns green

### 6. View Logs

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

### 7. Stop the System

```bash
docker compose down
```

To also remove volumes:
```bash
docker compose down -v
```

---

## Troubleshooting

### Docker Build Fails

If build fails with UV installation issues:
1. Check internet connection
2. Try: `docker compose build --no-cache`
3. Check logs: `docker compose build 2>&1 | tee build.log`

### Model Not Found

Ensure model exists:
```bash
ls -lh models/best_model.pth
```

### Redis Connection Issues

Check Redis is running:
```bash
docker compose ps
docker compose logs redis
```

### Port Already in Use

If ports 8501 or 8000 are in use:
1. Change ports in `docker-compose.yml`
2. Or stop conflicting services

---

## Expected Behavior

1. **Webcam Capture**: Streamlit captures frames from webcam
2. **Face Detection**: OpenCV detects face and draws bounding box
3. **Frame Publishing**: Frames published to `frames_stream` in Redis
4. **Inference**: Worker consumes frames, runs model inference
5. **Predictions**: Predictions published to `predictions_stream`
6. **Alarm Logic**: Alarm manager tracks consecutive drowsy frames
7. **Alarm Trigger**: After 10 seconds, alarm activates
8. **UI Updates**: Frontend displays status and alarms
9. **Reset**: "I am awake" button resets alarm state

---

## Success Criteria

✅ All services start without errors  
✅ Backend API responds to health check  
✅ Redis streams are created and working  
✅ Frontend displays webcam feed  
✅ Face detection works (green bounding box)  
✅ Predictions are being made  
✅ Alarm triggers after 10 seconds of drowsiness  
✅ "I am awake" button resets alarm  
✅ System is stable and responsive  

---

## Next Steps After Testing

Once testing is complete:
1. Document any issues found
2. Update HOWTO.md with any fixes
3. Prepare final submission

