# 🚗 Driver Drowsiness Detection System

A real-time ML-powered system that monitors driver alertness through webcam analysis and triggers an alarm sound and warning when drowsiness is detected.


---

## ✨ Features

- **Real-time Detection** — Continuous webcam monitoring with face detection
- **Deep Learning Model** — Custom CNN trained on 41K+ driver images
- **Microservices Architecture** — Decoupled services communicating via Redis Streams
- **Automatic Alarm System** — Visual + audio alerts after 10 seconds of drowsiness
- **Auto Model Loading** — Downloads trained model from W&B automatically
- **Fully Containerized** — Single-command deployment with Docker Compose

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Streamlit  │────▶│    Redis     │◀────│ Inference Worker│
│  Frontend   │     │   Streams    │     │   (PyTorch)     │
└─────────────┘     └──────────────┘     └─────────────────┘
       │                   │
       │                   ▼
       │            ┌──────────────┐     ┌─────────────────┐
       └───────────▶│ Alarm Manager│     │  FastAPI API    │
                    └──────────────┘     └─────────────────┘
```

| Service | Description |
|---------|-------------|
| **Frontend** | Webcam capture, face detection, real-time display |
| **Inference Worker** | Consumes frames, runs CNN inference |
| **Alarm Manager** | Tracks drowsy frames, triggers alarms |
| **FastAPI** | REST API for model serving |
| **Redis** | Message broker + state management |

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Weights & Biases API key ([get one here](https://wandb.ai/authorize))

### 1. Clone & Configure

```bash
git clone https://github.com/YOUR_USERNAME/driver-drowsiness-detection.git
cd driver-drowsiness-detection

# Create environment file
cp .env.example .env
# Edit .env and add your WANDB_API_KEY
```

### 2. Start the System

```bash
docker compose up --build
```

The system will automatically download the trained model from W&B.

### 3. Open the App

- **Frontend**: [http://localhost:8501](http://localhost:8501)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📖 Usage

1. Click **"▶️ Start Detection"** in the sidebar
2. Click **"START"** on the video player and allow camera access
3. Position your face in the camera view
4. The system will monitor continuously:
   - 🟢 **Green box** = Alert
   - 🟠 **Orange box** = Drowsy
   - 🔴 **Red box** = Alarm active (after 10s of drowsiness)
5. Click **"🙋 I am Awake"** to reset the alarm

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_API_KEY` | — | Your W&B API key (required) |
| `WANDB_PROJECT` | `Driver-Drowsiness-Training` | W&B project name |
| `WANDB_ARTIFACT_VERSION` | `latest` | Model version to download |
| `ALARM_THRESHOLD_SECONDS` | `10` | Seconds before alarm triggers |
| `FPS` | `5` | Frame processing rate |

---

## 🧪 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Predict from uploaded image |
| `/predict/base64` | POST | Predict from base64 image |
| `/docs` | GET | Interactive API documentation |

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@image.png"
```

---

## 🏋️ Training the Model

The dataset is available on Kaggle: [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)

```bash
# Place data in Data/ folder
# Data/Drowsy/      - ~22K drowsy images
# Data/Non Drowsy/  - ~19K alert images
```

### Standard Training

```bash
python train_model.py --mode train
```

### Hyperparameter Tuning with W&B Sweeps

The training script supports **automatic hyperparameter optimization** using Bayesian search:

```bash
# Run 10 trials of hyperparameter search
python train_model.py --mode sweep --sweep-count 10
```

**Parameters searched:**

| Parameter | Search Space |
|-----------|--------------|
| `learning_rate` | 0.0001 - 0.01 (log uniform) |
| `batch_size` | 16, 32, 64 |
| `epochs` | 5, 10, 15 |
| `dropout_rate` | 0.3, 0.5, 0.7 |
| `optimizer` | Adam, SGD, AdamW |

View sweep results at [wandb.ai](https://wandb.ai) in the Sweeps dashboard.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | PyTorch |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit + WebRTC |
| Message Queue | Redis Streams |
| Experiment Tracking | Weights & Biases |
| Containerization | Docker Compose |
| Package Manager | UV |

---

## 📁 Project Structure

```
├── src/
│   ├── backend/          # FastAPI app + CNN model
│   ├── frontend/         # Streamlit app
│   ├── inference_worker/ # Frame processing worker
│   ├── alarm_manager/    # Alarm logic
│   └── config/           # Settings + Redis utils
├── docker/               # Dockerfiles
├── notebooks/            # Jupyter notebooks
├── models/               # Trained model (auto-downloaded)
├── train_model.py        # Training script
└── docker-compose.yml    # Service orchestration
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Check `WANDB_API_KEY` in `.env` |
| Webcam not working | Grant browser camera permissions |
| Redis connection failed | Ensure `docker compose up` completed |
| Alarm not triggering | Check logs: `docker compose logs alarm_manager` |



## 🙏 Acknowledgments

- Dataset: [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) by Ismail Nasri
- Face Detection: OpenCV Haar Cascades
