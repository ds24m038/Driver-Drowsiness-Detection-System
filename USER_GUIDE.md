# User Guide - How to Run and Test the System

## Quick Start

The system is already running! Just follow these steps:

---

## Step 1: Access the Frontend

1. **Open your web browser** (Chrome, Firefox, or Safari)
2. **Navigate to**: http://localhost:8501
3. You should see the **"Driver Drowsiness Detection System"** interface

---

## Step 2: Start Detection

1. **Click the "▶️ Start" button** in the sidebar
2. **Allow webcam access** when your browser prompts you
   - Click "Allow" when asked for camera permission
3. **Position your face** in front of the camera
   - Make sure your face is clearly visible
   - Good lighting helps with face detection

---

## Step 3: Observe the System

### What You Should See:

1. **Webcam Feed**
   - Your live video feed appears in the main area
   - A **green bounding box** appears around your face when detected

2. **Status Information** (in the sidebar):
   - Current status: "✅ Status: **ALERT**" or "⚠️ Status: **DROWSY**"
   - Consecutive drowsy frames counter: `X/50`
   - Alarm status: "✅ No alarm" or "🚨 **ALARM ACTIVE**"

3. **Bounding Box Colors**:
   - **Green**: Driver is alert (no alarm)
   - **Orange**: Driver is drowsy (not yet alarm threshold)
   - **Red**: Alarm is active (10+ seconds of drowsiness)

---

## Step 4: Test Drowsiness Detection

### Simulate Drowsiness:

1. **Close your eyes** or look tired
2. **Keep your eyes closed** for at least 10 seconds
3. **Watch for**:
   - Bounding box turns **red**
   - Status shows "🚨 **ALARM ACTIVE**"
   - Visual warning appears
   - Counter shows drowsy frames increasing

### Expected Behavior:

- After **10 seconds** (50 frames at 5 FPS) of consecutive drowsiness:
  - Alarm activates
  - Bounding box turns red
  - Warning message appears
  - Status updates to show alarm is active

---

## Step 5: Test Alarm Reset

1. **Click the "🙋 I am Awake" button** in the sidebar
2. **Observe**:
   - Alarm resets immediately
   - Bounding box turns back to green
   - Counter resets to 0
   - Status shows "✅ No alarm"
   - Monitoring continues

---

## Step 6: Monitor System Activity

### View Logs in Real-Time:

Open a terminal and run:

```bash
cd "/Users/sinahaghgoo/Library/CloudStorage/OneDrive-Persönlich/Solution Deployment/Project"

# Watch all services
docker compose logs -f

# Or watch specific services
docker compose logs -f frontend
docker compose logs -f inference_worker
docker compose logs -f alarm_manager
docker compose logs -f api
```

### What to Look For:

**Frontend logs:**
- Frame publishing messages
- Redis connection status

**Inference Worker logs:**
- "Processing frame..." messages
- "Published prediction..." messages

**Alarm Manager logs:**
- "Processing prediction..." messages
- "ALARM ACTIVATED!" when threshold reached

---

## Step 7: Test Backend API (Optional)

### Test Health Endpoint:

```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
    "status": "healthy",
    "message": "API is running and model is loaded"
}
```

### Test Prediction Endpoint:

```bash
# Using a sample image from your dataset
curl -X POST "http://localhost:8000/predict" \
  -F "file=@Data/Drowsy/A0001.png"
```

**Expected Response:**
```json
{
    "predicted_class": "drowsy",
    "confidence": 0.99,
    "probabilities": {
        "alert": 0.01,
        "drowsy": 0.99
    },
    "timestamp": 1234567890.123
}
```

### View API Documentation:

Open in browser: http://localhost:8000/docs

This shows an interactive API documentation where you can test endpoints directly.

---

## Step 8: Check Redis State (Optional)

### Connect to Redis:

```bash
docker exec drowsiness_redis redis-cli
```

### Check Current State:

```redis
# Check current status
GET current_status

# Check alarm state
GET alarm_active

# Check consecutive drowsy frames
GET consecutive_drowsy_frames

# Check streams
XINFO STREAM frames_stream
XINFO STREAM predictions_stream
```

### Exit Redis CLI:

Type `exit` or press `Ctrl+D`

---

## Troubleshooting

### Issue: Webcam Not Working

**Solutions:**
1. **Check browser permissions**:
   - Chrome: Settings → Privacy → Site Settings → Camera
   - Firefox: Preferences → Privacy → Permissions → Camera
   - Safari: Safari → Settings → Websites → Camera

2. **Try a different browser**:
   - Chrome usually works best
   - Firefox is also good
   - Safari may have limitations

3. **Check if webcam is in use**:
   - Close other applications using the camera
   - Restart the browser

### Issue: Face Not Detected

**Solutions:**
1. **Improve lighting**: Make sure your face is well-lit
2. **Position correctly**: Face the camera directly
3. **Remove obstructions**: Remove glasses, masks, etc. if needed
4. **Check distance**: Not too close or too far

### Issue: No Predictions Being Made

**Check logs:**
```bash
docker compose logs inference_worker
```

**Look for:**
- "Processing frame..." messages
- Any error messages
- Redis connection issues

### Issue: Alarm Not Triggering

**Check:**
1. Are predictions being made? (check inference worker logs)
2. Is alarm manager running? (check alarm manager logs)
3. Are you drowsy for 10+ seconds continuously?

**Verify threshold:**
- Should be 50 frames (10 seconds at 5 FPS)
- Check counter in sidebar

### Issue: Services Not Running

**Restart services:**
```bash
cd "/Users/sinahaghgoo/Library/CloudStorage/OneDrive-Persönlich/Solution Deployment/Project"
docker compose restart
```

**Or restart specific service:**
```bash
docker compose restart frontend
docker compose restart inference_worker
docker compose restart alarm_manager
```

### Issue: Port Already in Use

**Check what's using the port:**
```bash
# Check port 8501 (frontend)
lsof -i :8501

# Check port 8000 (API)
lsof -i :8000
```

**Kill the process or change ports in docker-compose.yml**

---

## System Status Commands

### Check All Services:

```bash
docker compose ps
```

### View Service Logs:

```bash
# All services
docker compose logs

# Specific service
docker compose logs frontend
docker compose logs api
docker compose logs inference_worker
docker compose logs alarm_manager
docker compose logs redis
```

### Stop the System:

```bash
docker compose down
```

### Start the System:

```bash
docker compose up -d
```

### Rebuild and Restart:

```bash
docker compose down
docker compose build
docker compose up -d
```

---

## Expected System Behavior

### Normal Operation:

1. **Webcam captures frames** → Frontend
2. **Face detected** → Green bounding box appears
3. **Frame published** → Redis Streams (`frames_stream`)
4. **Inference runs** → Worker processes frame
5. **Prediction published** → Redis Streams (`predictions_stream`)
6. **Alarm logic** → Manager tracks consecutive drowsy frames
7. **UI updates** → Frontend displays status and alarms

### Alarm Flow:

1. **Drowsiness detected** → Status changes to "DROWSY"
2. **Counter increments** → Consecutive drowsy frames increase
3. **Threshold reached** → 50 frames (10 seconds)
4. **Alarm activates** → Red bounding box, warning message
5. **User clicks "I am awake"** → Alarm resets, counter clears

---

## Performance Tips

1. **Close unnecessary applications** to free up resources
2. **Use Chrome browser** for best performance
3. **Good lighting** improves face detection accuracy
4. **Stable internet** (if using remote Docker)

---

## Success Indicators

✅ **System is working correctly if:**
- Webcam feed appears
- Face is detected (green box)
- Status updates in real-time
- Predictions are being made (check logs)
- Alarm triggers after 10 seconds of drowsiness
- "I am awake" button resets alarm
- No errors in logs

---

## Need Help?

Check the logs:
```bash
docker compose logs -f
```

Check service status:
```bash
docker compose ps
```

View test results:
```bash
cat TEST_RESULTS.md
```

---

## Quick Reference

| Action | Command/URL |
|--------|-------------|
| Open Frontend | http://localhost:8501 |
| API Health | http://localhost:8000/health |
| API Docs | http://localhost:8000/docs |
| View Logs | `docker compose logs -f` |
| Check Status | `docker compose ps` |
| Restart | `docker compose restart` |
| Stop | `docker compose down` |
| Start | `docker compose up -d` |

---

Enjoy testing your driver drowsiness detection system! 🚗👁️

