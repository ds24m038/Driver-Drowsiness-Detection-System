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
import modal  # Required to fetch data from cloud

# --- Config ---
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0)
VOLUME_NAME = "drowsiness-dataset-vol"
REMOTE_MODEL_PATH = "drowsiness_resnet18.pth"
LOCAL_MODEL_PATH = "./models/resnet18_drowsiness.pth"

# --- Wandb Setup ---
print("Initializing Weights & Biases...")
wandb_api_key = os.getenv("WANDB_API_KEY", "b0d817ff452695d2ed53a6aa4a3810aa4323ab8c")

if wandb_api_key:
    try:
        wandb.login(key=wandb_api_key)
        wandb.init(
            project="drowsiness-detection",
            config={"architecture": "resnet18", "dataset": "live-webcam", "device": "cpu"}
        )
        print("Logged in to wandb")
    except Exception as e:
        print(f"Wandb init failed: {e}")
else:
    print("No WANDB API Key found. Running offline.")
    os.environ["WANDB_MODE"] = "offline"

# --- Modal Model Download ---
def download_model_from_modal():
    """Downloads the trained model from the Modal Volume to local disk."""
    print(f"Checking for model in Modal Volume: {VOLUME_NAME}...")
    try:
        # Connect to the volume
        vol = modal.Volume.from_name(VOLUME_NAME)
        
        # Ensure local directory exists
        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
        
        print(f"Downloading {REMOTE_MODEL_PATH} to {LOCAL_MODEL_PATH}...")
        
        # Read file from volume and write locally
        # Note: Depending on Modal version, we might read chunks or copy. 
        # reading into a buffer is the most compatible standard way.
        data_chunks = []
        try:
            for chunk in vol.read_file(REMOTE_MODEL_PATH):
                data_chunks.append(chunk)
            
            with open(LOCAL_MODEL_PATH, "wb") as f:
                for chunk in data_chunks:
                    f.write(chunk)
            
            print("Download successful.")
            return True
        except FileNotFoundError:
            print("Model file not found in Modal Volume! Have you run the training script?")
            return False
            
    except Exception as e:
        print(f"Failed to download from Modal: {e}")
        print("Ensure you have run 'modal setup' locally.")
        return False

# Attempt download if file doesn't exist locally
if not os.path.exists(LOCAL_MODEL_PATH):
    download_model_from_modal()

# --- Model Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = models.resnet18(weights=None)

# CRITICAL FIX: The training script uses 3 classes (0, 5, 10), so we must use 3 here.
# If we use 2, loading state_dict will fail with a size mismatch.
model.fc = torch.nn.Linear(model.fc.in_features, 3) 

if os.path.exists(LOCAL_MODEL_PATH):
    print(f"Loading {LOCAL_MODEL_PATH}...")
    try:
        checkpoint = torch.load(LOCAL_MODEL_PATH, map_location=device)
        
        # Handle Checkpoint vs State Dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
else:
    print(f"Warning: {LOCAL_MODEL_PATH} not found. Predictions will be random/simulated.")

model = model.to(device)
model.eval()

# Transforms (Must match training: 224x224, Normalized)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)), # Resize to fixed dimensions (not just 256)
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
        
        # Mapping based on training script: 0=Alert, 1=Low Vigilance, 2=Drowsy
        idx = pred_idx.ite9m()
        if idx == 0:
            label = "Awake"
        elif idx == 1:
            label = "Low Vigilance"
        elif idx == 2:
            label = "Drowsy"
        else:
            label = "Unknown"
            
        confidence = conf.item()

        # WandB Logging
        if wandb.run is not None:
            wandb.log({
                "class": label,
                "confidence": confidence,
                "prob_awake": probs[0].item() if len(probs) > 0 else 0,
                "prob_low_vigilance": probs[1].item() if len(probs) > 1 else 0,
                "prob_drowsy": probs[2].item() if len(probs) > 2 else 0
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