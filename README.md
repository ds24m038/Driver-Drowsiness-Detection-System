# 🚗 Driver Drowsiness Detection & Alarm

**Course:** Solution Deployment & Communication  
**Stack:** Python 3.11, FastAPI, Streamlit, PyTorch, Docker, Weights & Biases

## 📖 Project Overview

This repository contains a full-stack machine learning application designed to detect driver drowsiness in real-time. It was built to demonstrate MLOps best practices, including model serving, containerization, and experiment tracking.

The system captures webcam footage, analyzes the driver's face using a lightweight CNN, and triggers an **visual and audio alarm** if the driver appears drowsy for consecutive frames.

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Web UI & Webcam handling)
* **Backend:** FastAPI (REST API for Model Inference)
* **ML Model:** PyTorch (MobileNet/ResNet Transfer Learning)
* **Tracking:** Weights & Biases (Experiment logging)
* **Infrastructure:** Docker & Docker Compose

## 📂 Project Structure (TO BE MODIFED!!!)

```text
.
├── backend/           # FastAPI app, model serving logic, and Dockerfile
├── frontend/          # Streamlit app, UI logic, and Dockerfile
├── notebooks/         # Jupyter notebooks for data prep, training, and evaluation
├── docker-compose.yml # Orchestration for backend and frontend
└── README.md          # Project documentation
