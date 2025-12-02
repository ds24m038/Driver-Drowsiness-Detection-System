"""Standalone training script for driver drowsiness detection model."""
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import wandb
from dotenv import load_dotenv

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

from src.backend.models import DrowsinessCNN, create_model
from src.config.settings import WANDB_API_KEY, MODEL_INPUT_SIZE

# Use a separate W&B project for this training run
WANDB_PROJECT = "Driver-Drowsiness-Training"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"W&B Project: {WANDB_PROJECT}")


# Dataset class
class DrowsinessDataset(Dataset):
    def __init__(self, drowsy_dir, non_drowsy_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load Drowsy images (label 1)
        drowsy_path = Path(drowsy_dir)
        print(f"Loading Drowsy images from {drowsy_path}...")
        for img_path in drowsy_path.glob("*.png"):
            self.images.append(str(img_path))
            self.labels.append(1)  # drowsy = 1
        
        # Load Non Drowsy images (label 0)
        non_drowsy_path = Path(non_drowsy_dir)
        print(f"Loading Non Drowsy images from {non_drowsy_path}...")
        for img_path in non_drowsy_path.glob("*.png"):
            self.images.append(str(img_path))
            self.labels.append(0)  # alert = 0
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# Data transforms
train_transform = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    
    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, precision, recall, f1, precision_per_class, recall_per_class, f1_per_class, cm


def main():
    # Initialize W&B
    print("\n=== Initializing Weights & Biases ===")
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        name="drowsiness_cnn_training",
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "model_architecture": "DrowsinessCNN",
            "input_size": MODEL_INPUT_SIZE,
            "num_classes": 2,
            "optimizer": "Adam",
            "dataset": "Driver Drowsiness Dataset (DDD)",
        }
    )
    print(f"✓ W&B initialized. View run at: {wandb.run.url}")
    
    # Load dataset
    print("\n=== Loading Dataset ===")
    DATA_DIR = PROJECT_ROOT / "Data"
    drowsy_dir = DATA_DIR / "Drowsy"
    non_drowsy_dir = DATA_DIR / "Non Drowsy"
    
    full_dataset = DrowsinessDataset(drowsy_dir, non_drowsy_dir, transform=None)
    print(f"Total images loaded: {len(full_dataset)}")
    
    # Split dataset: 70% train, 15% val, 15% test
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create data loaders
    batch_size = wandb.config.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Log dataset info to W&B
    wandb.config.update({
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "total_samples": len(full_dataset),
    })
    
    # Initialize model
    print("\n=== Initializing Model ===")
    # Use MPS (Metal Performance Shaders) for Apple Silicon GPUs
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    model = create_model(num_classes=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created on device: {device}")
    print(f"Total parameters: {total_params:,}")
    
    wandb.config.update({"total_parameters": total_params})
    
    # Training loop
    print("\n=== Starting Training ===")
    epochs = wandb.config.epochs
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, precision, recall, f1, prec_per_class, rec_per_class, f1_per_class, cm = validate(
            model, val_loader, criterion, device
        )
        
        # Log to W&B
        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_precision_alert": prec_per_class[0],
            "val_precision_drowsy": prec_per_class[1],
            "val_recall_alert": rec_per_class[0],
            "val_recall_drowsy": rec_per_class[1],
            "val_f1_alert": f1_per_class[0],
            "val_f1_drowsy": f1_per_class[1],
        }
        
        # Log confusion matrix
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=[0] * cm[0, 0] + [0] * cm[0, 1] + [1] * cm[1, 0] + [1] * cm[1, 1],
                preds=[0] * cm[0, 0] + [1] * cm[0, 1] + [0] * cm[1, 0] + [1] * cm[1, 1],
                class_names=["alert", "drowsy"]
            )
        })
        
        wandb.log(log_dict)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, MODEL_PATH)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    print(f"\n=== Training Completed ===")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model and evaluate on test set
    print("\n=== Evaluating on Test Set ===")
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc, test_precision, test_recall, test_f1, test_prec_per_class, test_rec_per_class, test_f1_per_class, test_cm = validate(
        model, test_loader, criterion, device
    )
    
    print(f"Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Per-class Precision: Alert={test_prec_per_class[0]:.4f}, Drowsy={test_prec_per_class[1]:.4f}")
    print(f"  Per-class Recall: Alert={test_rec_per_class[0]:.4f}, Drowsy={test_rec_per_class[1]:.4f}")
    print(f"  Per-class F1: Alert={test_f1_per_class[0]:.4f}, Drowsy={test_f1_per_class[1]:.4f}")
    
    # Log test metrics to W&B
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_precision_alert": test_prec_per_class[0],
        "test_precision_drowsy": test_prec_per_class[1],
        "test_recall_alert": test_rec_per_class[0],
        "test_recall_drowsy": test_rec_per_class[1],
        "test_f1_alert": test_f1_per_class[0],
        "test_f1_drowsy": test_f1_per_class[1],
    })
    
    # Log test confusion matrix
    wandb.log({
        "test_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=[0] * test_cm[0, 0] + [0] * test_cm[0, 1] + [1] * test_cm[1, 0] + [1] * test_cm[1, 1],
            preds=[0] * test_cm[0, 0] + [1] * test_cm[0, 1] + [0] * test_cm[1, 0] + [1] * test_cm[1, 1],
            class_names=["alert", "drowsy"]
        )
    })
    
    # Save model as W&B artifact
    print("\n=== Saving Model Artifact to W&B ===")
    artifact = wandb.Artifact(
        name="drowsiness_detection_model",
        type="model",
        description="PyTorch CNN model for driver drowsiness detection. Trained on Driver Drowsiness Dataset (DDD)."
    )
    artifact.add_file(str(MODEL_PATH))
    artifact.add_file(str(PROJECT_ROOT / "src" / "backend" / "models.py"), name="model_architecture.py")
    
    # Add metadata
    artifact.metadata = {
        "model_architecture": "DrowsinessCNN",
        "input_size": MODEL_INPUT_SIZE,
        "num_classes": 2,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "total_parameters": total_params,
    }
    
    wandb.log_artifact(artifact)
    print(f"✓ Model artifact saved to W&B")
    print(f"  Artifact name: drowsiness_detection_model")
    
    wandb.finish()
    print("\n✓ Training complete! Model saved and logged to W&B.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"View your run at: https://wandb.ai")


if __name__ == "__main__":
    main()

