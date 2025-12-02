# Driver Drowsiness Detection & Alarm – Final Project Specification

## 1. Project Context

This project is the final assignment for the course **“Solution Deployment & Communication”**.  
The goal is to implement a **cohesive end-to-end ML solution** that applies the key topics from the course:

- **Model Serving & API Development (FastAPI)**  
- **Docker & Docker Compose for ML services**  
- **Model serving with TensorFlow Serving / TorchServe concepts**  
- **Experiment tracking and model versioning with Weights & Biases (W&B)**  
- **Solution deployment, monitoring, and communication**  

The implementation must focus on **practical deployment of a machine learning model** within a software system, not just model training.

---

## 2. High-Level Goal

**Goal:**  
Provide a **driver drowsiness detection system** that:

1. Uses a **PyTorch CNN** model to classify a driver’s state as:
   - `alert`
   - `drowsy`
2. Serves the trained model via a **FastAPI REST API** (dockerized).
3. Provides a **Streamlit frontend** that:
   - Captures webcam frames.
   - Visualizes the driver’s face with a rectangle (green or red).
   - Shows the predicted state (alert / drowsy).
   - Triggers and displays alarms (visual + audio).
4. Uses **Redis (with Redis Streams)** as a central component for:
   - Real-time state management.
   - Inter-process communication between:
     - Webcam capture module
     - CNN inference module
     - User interface (Streamlit)
     - Alarm system
5. Uses **Weights & Biases (W&B)** to:
   - Track experiments, datasets, and model versions.
   - Log metrics, hyperparameters, and artifacts.
   - Perform and log hyperparameter tuning.
6. Is fully **containerized with Docker**, and can be started with a single command (e.g. using Docker Compose).

---

## 3. System Overview

The system consists of the following main components:

1. **Model Training & Experimentation (Offline)**
   - Implemented in Jupyter Notebooks (PyTorch).
   - Uses the Driver Drowsiness Dataset (DDD) from the provided `Data` folder.
   - Integrated with W&B for:
     - Experiment tracking
     - Hyperparameter tuning
     - Model and dataset versioning
     - Performance visualization

2. **Model Serving / Backend API**
   - **FastAPI** application.
   - Loads a trained PyTorch CNN model (exported from training phase).
   - Provides REST endpoints for:
     - Health checks
     - Single-frame inference
     - Optional batching if needed

3. **Frontend / User Interface**
   - **Streamlit** application.
   - Captures webcam frames from the user’s device.
   - Sends frames (or references to frames) for inference.
   - Displays:
     - Webcam feed with a rectangle around the driver’s face.
     - Current driver status (alert / drowsy).
     - Active alarms (visual warnings and audio alarm).
   - Provides a button **“I am awake”** to reset the alarm state.

4. **Redis / Redis Streams**
   - Central system for **real-time state management** and **communication**.
   - Mandatory usage of **Redis Streams** as part of the architecture.
   - Responsibilities include:
     - Store the current driver status: `alert` / `drowsy`.
     - Maintain a **consecutive drowsiness counter** based on timestamps.
     - Manage an **alarm activation flag**.

5. **Containerization**
   - All components run in **Docker containers**.
   - Optionally orchestrated via **Docker Compose**, e.g.:
     - `api` (FastAPI backend)
     - `frontend` (Streamlit app)
     - `redis` (Redis server)
     - `trainer` (optional, for running scheduled training jobs or notebooks in a container)
   - Build and run instructions must be documented.

6. **Documentation & HOW-TO**
   - `README.md` – project overview & quickstart.
   - `HOWTO.md` – step-by-step instructions from project creation to running the full system.
   - Jupyter notebooks – detailed HOW-TOs and documentation for:
     - Data exploration
     - Model training
     - W&B integration
     - Deployment outline (if applicable)

---

## 4. Project Idea – Functional Description

### 4.1 Core Use Case

**Use Case:**  
Detect if a driver is drowsy using webcam footage and trigger alarms if drowsiness persists.

**Desired Workflow (User-Level):**

1. User starts the system (e.g., using `docker compose up`).
2. User opens the Streamlit UI in a browser.
3. User starts the webcam stream within Streamlit:
   - A **green rectangle** appears around the detected face.
   - The system begins continuous inference using the CNN model.
4. At each time step:
   - The model predicts `alert` or `drowsy` for the current frame.
   - The current state and counters are updated in Redis.
   - The rectangle color and UI are updated accordingly:
     - **Green** – driver is alert.
     - **Red** – driver is in an active drowsiness alarm state.
5. If the driver is classified as `drowsy` for **at least 10 consecutive seconds**, then:
   - A **visual alarm** is shown in the UI (e.g., big red warning message).
   - An **audio alarm** is played (e.g., sound file from the frontend).
   - The rectangle around the face turns **red**.
   - An **alarm flag** is set in Redis.
6. To stop the alarm:
   - The driver clicks the **“I am awake”** button in the UI.
   - The system:
     - Resets the drowsiness counter.
     - Clears the alarm flag.
     - Turns the rectangle back to **green**.
   - Monitoring continues from that point onward.

### 4.2 Timing Logic (Drowsiness Threshold)

- Define a **configuration parameter**:
  - `ALARM_THRESHOLD_SECONDS = 10`  
- At each frame, the system has:
  - A predicted label: `alert` or `drowsy`.
  - A timestamp (or frame index).
- The Redis-backed logic must ensure:
  - If the driver is `drowsy` **continuously** for at least `ALARM_THRESHOLD_SECONDS`, alarm is triggered.
  - If the driver becomes `alert` before that threshold, the counter is reset.

Implementation details (for LLM clarity):

- The system can:
  - Either track timestamps and compute elapsed time.
  - Or assume a fixed frame rate `FPS` and require `ALARM_THRESHOLD_FRAMES = FPS * ALARM_THRESHOLD_SECONDS` consecutive `drowsy` predictions.
- This mapping (time ↔ frames) must be implemented explicitly and consistently in the code.

---

## 5. Data & Model

### 5.1 Dataset Description

The system uses the **Driver Drowsiness Dataset (DDD)** provided in the `Data` folder.

Short description (adapted for implementation):

- Source: Extracted and cropped faces from the **Real-Life Drowsiness Dataset**.
- Preprocessing:
  - Frames extracted from videos as images.
  - Viola–Jones algorithm used to extract regions of interest (driver faces).
- Properties:
  - Color format: **RGB images**
  - Number of classes: **2**
    - `Drowsy`
    - `Non Drowsy` (i.e., alert)
  - Image size: **227 × 227**
  - Number of images: **> 41,790**
  - Total file size: **≈ 2.32 GB**

### 5.2 Model Requirements

- Framework: **PyTorch**
- Model type: **Convolutional Neural Network (CNN)** for image classification.
- Input: Single RGB face image (shape e.g. `(3, 227, 227)`).
- Output:
  - Either:
    - Logits for 2 classes, or
    - Probabilities for 2 classes
  - Final decision: `alert` / `drowsy` (mapping from labels to human-readable strings must be consistent).

### 5.3 Training & Evaluation

- Implement training in Jupyter notebooks (e.g. `notebooks/train_model.ipynb`).
- Use standard train/validation/test splits.
- Use data augmentations as needed (optional).
- Integrate with **W&B**:
  - Create a new W&B project: **`SDC Project Final`**.
  - Use the API key from the `.env` file.
  - Log:
    - Metrics (accuracy, loss, precision/recall, etc.).
    - Hyperparameters (learning rate, batch size, model architecture details).
    - Model checkpoints (best model artifacts).
    - Dataset versions (if feasible).
  - Optionally perform **hyperparameter tuning** with W&B sweeps.

---

## 6. Technical Requirements

### 6.1 REST API (FastAPI)

- The backend must be a **FastAPI** application.
- The model must be exposed via REST endpoints, for example:

  - `GET /health`  
    - Returns a simple JSON indicating the service is alive.

  - `POST /predict`  
    - Input: image data (e.g. base64-encoded image, or multipart/form-data).
    - Output: JSON containing:
      - Predicted class: `"alert"` or `"drowsy"`.
      - Optional: probability scores for each class.
      - Timestamp of prediction.

- The API must:
  - Load the trained PyTorch model at startup.
  - Handle errors gracefully (invalid input, missing model, etc.).
  - Be suitable for running inside a Docker container.

### 6.2 Frontend (Streamlit)

- The frontend must be a **Streamlit** app.
- Responsibilities:
  - Access the local webcam of the user.
  - Display the video stream in real time.
  - Periodically send frames to the backend (directly or via Redis, depending on design).
  - Display:
    - A rectangle around the detected face:
      - Green if no active alarm.
      - Red if drowsiness alarm is active.
    - Current status text: `"Driver is alert"` or `"Driver is drowsy"`.
  - Play an **audio alarm** when Redis indicates alarm is active.
  - Show a prominent visual alarm message (e.g., red banner or dialog).
  - Provide the button: **“I am awake”**, which:
    - Sends a signal to reset the alarm and counters (via Redis or API).

### 6.3 Redis & Redis Streams

- A **Redis** instance is mandatory and must be part of the architecture.
- **Redis Streams** must be used, not just simple keys (to fulfil the course requirement).
- Suggested responsibilities (can be refined in implementation, but must be consistent):

  1. **Frame Stream**
     - Stream key example: `frames_stream`
     - Producer: Webcam/Frontend module.
     - Consumers:
       - Inference worker (model inference service).
     - Each message can contain:
       - Frame ID
       - Timestamp
       - Encoded image data or reference path

  2. **Prediction Stream**
     - Stream key example: `predictions_stream`
     - Producer: Inference worker.
     - Consumers:
       - Alarm manager logic
       - Frontend (optionally)
     - Each message can contain:
       - Frame ID
       - Timestamp
       - Predicted label (`alert` / `drowsy`)
       - Confidence scores

  3. **State Keys**
     - Regular Redis keys/hashes for current state:
       - `current_status` → `"alert"` or `"drowsy"`
       - `consecutive_drowsy_seconds` or `consecutive_drowsy_frames`
       - `alarm_active` → `true` / `false`

  4. **Alarm Stream (Optional)**
     - Stream key example: `alarm_stream`
     - Used to broadcast alarm events:
       - Alarm activated
       - Alarm cleared by user

- The exact schema can be defined in the code, but the following requirements must be met:
  - Redis is used as **central coordination** between processes.
  - The drowsiness counting and alarm logic rely on Redis state.
  - Redis Streams are explicitly utilized for messaging between components.

### 6.4 Docker & Infrastructure

- All main components must be dockerized:
  - `FastAPI` backend (model serving).
  - `Streamlit` frontend.
  - `Redis` server (official image can be used).
  - Optional additional services (e.g. dedicated inference worker, training service).
- A `Dockerfile` must exist for each custom service.
- Optionally provide a `docker-compose.yml` that:
  - Starts all required services with one command.
  - Defines network and environment variables.
- Logging & basic troubleshooting (e.g. logging to stdout / files) should be supported.
- As package manager use UV instead of PIP!! 

### 6.5 Weights & Biases (W&B) – Detailed Requirements

- Use the provided API key from `.env` (e.g. `WANDB_API_KEY`).
- **Project name** must be exactly:  
  `SDC Project Final`
- Required:
  - At least one fully logged training run.
  - Metrics tracking (loss, accuracy, etc.).
  - Model artifact saving (best-performing model).
  - Experiment report or W&B dashboard showing:
    - Model performance
    - Possibly a comparison of multiple runs
  - Hyperparameter tuning:
    - At least one W&B sweep or manual multi-run experiment with varying hyperparameters.
- All W&B-related configuration (project name, entity, etc.) should be clearly documented in `HOWTO.md`.

---

## 7. Deliverables & Documentation

### 7.1 Mandatory Files

1. **`README.md`**
   - Overview of the project.
   - Short description of the system and its components.
   - Quickstart instructions:
     - How to build and run with Docker / Docker Compose.
     - How to access the Streamlit UI.
   - High-level architecture diagram (textual or linked image).
   - Brief explanation of Redis, W&B, and model.

2. **`HOWTO.md`**
   - Detailed, step-by-step instructions including:
     1. How the project was created from scratch.
     2. How the data was obtained/placed in the `Data` folder.
     3. How the environment was set up (conda/venv, dependencies).
     4. How the PyTorch model was trained (with references to notebooks).
     5. How W&B was configured and used.
     6. How Docker images were built.
     7. How Docker Compose (if used) is configured and started.
     8. How to run the complete system and test it:
        - Starting all services.
        - Using the Streamlit UI to trigger the detection and alarms.
        - Stopping the system.
   - Any known issues or troubleshooting tips.

3. **Jupyter Notebooks (HOW-TO Documentation)**
   - At least the following recommended notebooks:
     - `notebooks/01_data_exploration.ipynb`  
       - Load dataset, show examples, basic statistics.
     - `notebooks/02_model_training.ipynb`  
       - Full training code, W&B logging, evaluation.
     - `notebooks/03_inference_demo.ipynb` (optional but recommended)  
       - Load trained model and run inference on sample images.
     - `notebooks/04_wandb_reporting.ipynb` (optional)  
       - Show how to interpret W&B runs, metrics, and artifacts.

   - These notebooks must function as **HOW-TOs**, explaining:
     - What each step does.
     - How the user can reproduce or modify the workflow.

---

## 8. Suggested Repository Structure (Example)

> This is a suggestion to help structure the project; the exact structure can be adapted but should remain clear and documented.

```text
.
├── Data/
│   └── ... (Driver Drowsiness Dataset - DDD)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_inference_demo.ipynb
│   └── 04_wandb_reporting.ipynb
├── src/
│   ├── backend/
│   │   ├── main.py                # FastAPI app
│   │   ├── models.py              # PyTorch model definition & loading
│   │   ├── schemas.py             # Pydantic models for API
│   │   └── redis_utils.py         # Redis / Streams helpers
│   ├── frontend/
│   │   └── app.py                 # Streamlit frontend
│   ├── inference_worker/
│   │   └── worker.py              # Optional: separate service for inference via Redis streams
│   └── config/
│       └── settings.py            # Config (env vars, constants like ALARM_THRESHOLD_SECONDS)
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── Dockerfile.worker          # If applicable
├── docker-compose.yml             # If used for orchestration
├── .env                           # Contains W&B API key and other secrets (not committed)
├── README.md
└── HOWTO.md
