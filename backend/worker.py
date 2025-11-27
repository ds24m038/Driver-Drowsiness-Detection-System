import redis
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import time
from PIL import Image
import wandb

# --- Config ---
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0)

# --- Wandb Setup ---
print("Initializing Weights & Biases...")
# Priority: Check Environment Variable first (from Docker Compose), then hardcoded fallback
wandb_api_key = os.getenv("WANDB_API_KEY", "b0d817ff452695d2ed53a6aa4a3810aa4323ab8c")

if wandb_api_key:
    wandb.login(key=wandb_api_key)
    print("Logged in to wandb")
    
    try:
        wandb.init(
            project="drowsiness-detection",
            config={
                "architecture": "resnet18",
                "dataset": "live-webcam",
                "device": "cpu"
            }
        )
        print("Wandb initialized successfully")
    except Exception as e:
        print(f"Wandb init failed: {e}")
else:
    print("No WANDB API Key found. Running offline.")
    os.environ["WANDB_MODE"] = "offline"

# --- Model Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model_path = "../models/resnet18_drowsiness.pt"

if os.path.exists(model_path):
    print(f"Loading {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # Handle Checkpoint vs State Dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
else:
    print(f"Warning: {model_path} not found. Predictions will be random/simulated.")

model = model.to(device)
model.eval()

# Transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_prediction(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv2 is None: return "Error", 0.0

        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            
        conf, pred_idx = torch.max(probs, 0)
        label = "Drowsy" if pred_idx.item() == 1 else "Awake"
        confidence = conf.item()

        # WandB Logging
        if wandb.run is not None:
            wandb.log({
                "class": label,
                "confidence": confidence,
                "prob_awake": probs[0].item(),
                "prob_drowsy": probs[1].item()
            })

        return label, confidence
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Error", 0.0

# --- Loop ---
last_id = '$'
print("Worker listening...")
while True:
    try:
        streams = redis_client.xread({'camera_stream': last_id}, count=1, block=1000)
        if not streams: continue
        
        for _, messages in streams:
            for message_id, data in messages:
                last_id = message_id
                label, conf = get_prediction(data[b'data'])
                
                redis_client.xadd("prediction_stream", 
                                  {"class": label, "confidence": conf}, 
                                  maxlen=100)
                print(f"Processed {message_id}: {label} ({conf:.2f})")
                
    except Exception as e:
        print(f"Connection Error: {e}")
        time.sleep(1)