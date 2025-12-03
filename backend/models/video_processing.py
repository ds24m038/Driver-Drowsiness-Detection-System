import modal
import os
import cv2
import numpy as np
from pathlib import Path
import random

# 1. Define the Modal App and Image
app = modal.App("drowsiness-detection-pipeline")

# Added torch, torchvision, wandb for training
image = (
    modal.Image.debian_slim()
    .pip_install(
        "opencv-python-headless", 
        "numpy", 
        "pandas", 
        "mediapipe", 
        "torch", 
        "torchvision", 
        "wandb",
        "scikit-learn",
        "tqdm" # Added tqdm for progress bars
    )
    .apt_install("libgl1-mesa-glx", "libglib2.0-0") 
)

# 2. Define a Volume
# CRITICAL: This must be an absolute path (starts with /). 
# If set to "data", it will fail with "InvalidError: Volume data must be a canonical, absolute path"
VOLUME_ROOT = "/data" 

data_volume = modal.Volume.from_name("drowsiness-dataset-vol", create_if_missing=True)
RAW_DATA_PATH = f"{VOLUME_ROOT}/raw"
PROCESSED_DATA_PATH = f"{VOLUME_ROOT}/processed"

# --- PART 1: Data Processing ---

@app.function(image=image, volumes={VOLUME_ROOT: data_volume}, timeout=7200)
def extract_frames_and_crop(video_info):
    """
    Worker function to process a single video file.
    Detects faces, crops them, and saves as .npy files.
    """
    import mediapipe as mp
    
    video_path, label, dest_root = video_info
    
    mp_face_detection = mp.solutions.face_detection
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return 0

    frame_count = 0
    saved_count = 0
    sampling_rate = 10 
    
    # Get video name and parent folder name to create a unique ID
    # This prevents 'Folder1/0.mov' and 'Folder2/0.mov' from overwriting each other
    video_name = Path(video_path).stem
    parent_name = Path(video_path).parent.name
    
    save_dir = os.path.join(dest_root, str(label))
    os.makedirs(save_dir, exist_ok=True)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sampling_rate == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                if results.detections:
                    detection = results.detections[0]
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)

                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, iw - x)
                    h = min(h, ih - y)

                    face_crop = frame[y:y+h, x:x+w]

                    if face_crop.size > 0:
                        # Include parent_name in filename to ensure uniqueness
                        out_name = f"{parent_name}_{video_name}_frame_{frame_count}.npy"
                        out_path = os.path.join(save_dir, out_name)
                        np.save(out_path, face_crop)
                        saved_count += 1

            frame_count += 1

    cap.release()
    return saved_count

@app.function(image=image, volumes={VOLUME_ROOT: data_volume}, timeout=7200)
def process_dataset():
    """
    Main coordinator function for data processing.
    """
    print(f"Scanning raw data in volume at: {RAW_DATA_PATH}")
    
    # DEBUG: Print file structure to help debug "0 videos found"
    print("--- DEBUG: File System Check ---")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"ERROR: The path {RAW_DATA_PATH} does not exist inside the container.")
        print(f"Contents of {VOLUME_ROOT}: {os.listdir(VOLUME_ROOT) if os.path.exists(VOLUME_ROOT) else 'Volume root not found'}")
        return

    debug_count = 0
    for root, dirs, files in os.walk(RAW_DATA_PATH):
        for name in files:
            if debug_count < 5:
                print(f"  Found file: {os.path.join(root, name)}")
            debug_count += 1
    print(f"--- DEBUG: Total files found in scan: {debug_count} ---")

    tasks = []
    
    for cls in ["0", "5", "10"]:
        os.makedirs(os.path.join(PROCESSED_DATA_PATH, cls), exist_ok=True)

    # Walk through the raw data directory
    # Looks for structure: /data/raw/ParticipantFolder/0.mov
    found_files = 0
    skipped_log_count = 0
    
    for root, dirs, files in os.walk(RAW_DATA_PATH):
        for file in files:
            if file.lower().endswith(('.mov', '.mp4', '.avi')):
                full_path = os.path.join(root, file)
                
                # UPDATED LOGIC: Get label from filename (e.g., '0.mov' -> '0')
                file_stem = Path(file).stem
                
                if file_stem in ["0", "5", "10"]:
                    found_files += 1
                    # Pass the file stem as the label
                    tasks.append((full_path, file_stem, PROCESSED_DATA_PATH))
                else:
                    # DEBUG: Print why we are skipping files (only first 5)
                    if skipped_log_count < 5:
                        print(f"  Skipping file: {file} (Stem '{file_stem}' is not in ['0', '5', '10'])")
                        skipped_log_count += 1

    print(f"Found {found_files} videos matching class names (0, 5, 10). Queuing {len(tasks)} tasks.")
    
    if len(tasks) == 0:
        print("No videos found matching criteria! Check the DEBUG logs above to see what files exist.")
        return

    total_frames = 0
    results = extract_frames_and_crop.map(tasks)
    
    for count in results:
        total_frames += count
        
    print(f"Processing complete. Extracted {total_frames} face arrays.")
    data_volume.commit()

# --- PART 2: Training ---

# Define the training function to run on a GPU
# REQUIRES: modal secret create wandb-secret WANDB_API_KEY=your_key
@app.function(
    image=image, 
    volumes={VOLUME_ROOT: data_volume}, 
    gpu=["A10G", "T4"],  # Prioritize A10G, fallback to T4 if unavailable
    timeout=3600,
    # Updated to use the specific API key you provided directly
    secrets=[modal.Secret.from_dict({"WANDB_API_KEY": "b0d817ff452695d2ed53a6aa4a3810aa4323ab8c"})]
)
def train_model(batch_size=32, epochs=10, learning_rate=0.001):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    import wandb
    from PIL import Image
    from tqdm import tqdm # Import tqdm

    print("Setting up training environment...")
    
    # Path for intermediate checkpoints
    CHECKPOINT_PATH = f"{VOLUME_ROOT}/models/training_checkpoint.pth"

    # 1. Custom Dataset
    class DrowsinessDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.samples = []
            
            # Map folder names (0, 5, 10) to classes (0, 1, 2)
            # 0: Alert, 1: Low Vigilance, 2: Drowsy
            self.class_map = {"0": 0, "5": 1, "10": 2}
            
            for label_name, label_idx in self.class_map.items():
                class_dir = os.path.join(root_dir, label_name)
                if os.path.isdir(class_dir):
                    for file in os.listdir(class_dir):
                        if file.endswith('.npy'):
                            self.samples.append((os.path.join(class_dir, file), label_idx))
            
            print(f"Dataset loaded: {len(self.samples)} samples found.")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            
            # Load numpy array
            img_array = np.load(path)
            
            # Convert BGR (OpenCV default) to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image for torchvision transforms
            image = Image.fromarray(img_array)
            
            if self.transform:
                image = self.transform(image)
                
            return image, label

    # 2. Setup Data Transforms
    # ResNet18 expects 224x224 and normalized data
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Create Datasets and Loaders
    full_dataset = DrowsinessDataset(PROCESSED_DATA_PATH, transform=data_transforms)
    
    if len(full_dataset) == 0:
        print("Error: No processed data found. Run process_dataset first.")
        return

    # Split: 80% train, 20% val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 4. Initialize WandB
    wandb.init(
        project="drowsiness-detection-resnet",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "architecture": "resnet18",
            "dataset": "UTA-Real-Life"
        }
    )

    # 5. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Modify final layer for 3 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- RESUME LOGIC START ---
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Found checkpoint at {CHECKPOINT_PATH}. Resuming training...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"✅ Resuming successfully from Epoch {start_epoch}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}. Starting from scratch.")
    else:
        print("🚀 No checkpoint found. Starting training from scratch.")
    # --- RESUME LOGIC END ---

    # 6. Training Loop
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Wrap the iterator with tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Optional: Update progress bar with current loss
            pbar.set_postfix({'loss': loss.item()})

        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Val Acc: {val_acc:.2f}%")
        
        # Log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        })
        
        # --- SAVE CHECKPOINT AFTER EPOCH ---
        print(f"Saving checkpoint for epoch {epoch+1}...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, CHECKPOINT_PATH)
        data_volume.commit() # Important: Commit to volume immediately

    # Save the model to the volume
    model_save_path = f"{VOLUME_ROOT}/drowsiness_resnet18.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    data_volume.commit()
    
    wandb.finish()

@app.local_entrypoint()
def main():
    print("--- Drowsiness Pipeline Started ---")
    
    # Sanity Check for Configuration
    if not VOLUME_ROOT.startswith("/"):
        raise ValueError(f"Configuration Error: VOLUME_ROOT must be an absolute path (e.g., '/data'). Current value: '{VOLUME_ROOT}'")

    # -----------------------------------------------------
    # TOGGLE THESE FLAGS TO CONTROL THE PIPELINE
    # -----------------------------------------------------
    RUN_PROCESSING = False # Set to False since data is already processed
    RUN_TRAINING = True    # Set to True to train the ResNet model
    # -----------------------------------------------------

    if RUN_PROCESSING:
        print("Step 1: Launching data processing job...")
        process_dataset.remote()
    
    if RUN_TRAINING:
        print("Step 2: Launching training job...")
        train_model.remote()
        
    if not RUN_PROCESSING and not RUN_TRAINING:
        print("No jobs selected! Please set RUN_PROCESSING or RUN_TRAINING to True in the main function.")