"""Standalone training script for driver drowsiness detection model with W&B Sweeps support."""
import os
import sys
import argparse
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

# Constants
WANDB_PROJECT = "Driver-Drowsiness-Training"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Sweep configuration for hyperparameter search
SWEEP_CONFIG = {
    "method": "bayes",  # Bayesian optimization
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 0.0001,
            "max": 0.01,
        },
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 10, 15]},
        "dropout_rate": {"values": [0.3, 0.5, 0.7]},
        "optimizer": {"values": ["Adam", "SGD", "AdamW"]},
    },
}


# Dataset class
class DrowsinessDataset(Dataset):
    def __init__(self, drowsy_dir, non_drowsy_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load Drowsy images (label 1)
        drowsy_path = Path(drowsy_dir)
        for img_path in drowsy_path.glob("*.png"):
            self.images.append(str(img_path))
            self.labels.append(1)  # drowsy = 1
        
        # Load Non Drowsy images (label 0)
        non_drowsy_path = Path(non_drowsy_dir)
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


def get_transforms():
    """Get training and validation transforms."""
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
    
    return train_transform, val_transform


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
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


def validate(model, val_loader, criterion, device):
    """Validate the model."""
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
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
    }


def create_data_loaders(batch_size):
    """Create train, validation, and test data loaders."""
    DATA_DIR = PROJECT_ROOT / "Data"
    drowsy_dir = DATA_DIR / "Drowsy"
    non_drowsy_dir = DATA_DIR / "Non Drowsy"
    
    train_transform, val_transform = get_transforms()
    
    # Create separate datasets for train and val/test to apply different transforms
    train_full_dataset = DrowsinessDataset(drowsy_dir, non_drowsy_dir, transform=train_transform)
    val_full_dataset = DrowsinessDataset(drowsy_dir, non_drowsy_dir, transform=val_transform)
    
    # Calculate split sizes
    total_size = len(train_full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # Generate the same indices for both datasets
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_size, generator=generator).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)
    test_dataset = Subset(val_full_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
    }


def get_optimizer(model, optimizer_name, learning_rate):
    """Get optimizer by name."""
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "AdamW":
        return optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        return optim.Adam(model.parameters(), lr=learning_rate)


def train(config=None, save_best=True):
    """Main training function that works with both normal training and sweeps."""
    with wandb.init(config=config):
        config = wandb.config
        
        # Get hyperparameters
        learning_rate = config.get("learning_rate", 0.001)
        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 10)
        dropout_rate = config.get("dropout_rate", 0.5)
        optimizer_name = config.get("optimizer", "Adam")
        
        print(f"\n{'='*60}")
        print(f"Training with: lr={learning_rate:.6f}, batch={batch_size}, epochs={epochs}")
        print(f"              dropout={dropout_rate}, optimizer={optimizer_name}")
        print(f"{'='*60}\n")
        
        # Setup
        device = get_device()
        print(f"Using device: {device}")
        
        # Create data loaders
        train_loader, val_loader, test_loader, dataset_sizes = create_data_loaders(batch_size)
        print(f"Dataset sizes: {dataset_sizes}")
        
        # Create model
        model = create_model(num_classes=2)
        # Update dropout rate if the model supports it
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
        model = model.to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model, optimizer_name, learning_rate)
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_metrics = validate(model, val_loader, criterion, device)
            
            # Log to W&B
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
            })
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
                  f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.2f}%")
            
            # Save best model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                if save_best:
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_metrics["accuracy"],
                        "config": dict(config),
                    }, MODEL_PATH)
                    print(f"  ✓ Saved best model (val_acc: {val_metrics['accuracy']:.2f}%)")
        
        # Final test evaluation
        test_metrics = validate(model, test_loader, criterion, device)
        wandb.log({
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "best_val_accuracy": best_val_acc,
        })
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best val_acc: {best_val_acc:.2f}%")
        print(f"Test accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"{'='*60}\n")
        
        return best_val_acc


def run_sweep(count=10):
    """Run a W&B sweep for hyperparameter optimization."""
    print(f"\n{'='*60}")
    print("Starting W&B Hyperparameter Sweep")
    print(f"Method: {SWEEP_CONFIG['method']}")
    print(f"Optimizing: {SWEEP_CONFIG['metric']['name']} ({SWEEP_CONFIG['metric']['goal']})")
    print(f"Running {count} trials")
    print(f"{'='*60}\n")
    
    # Create sweep
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_PROJECT)
    
    # Run sweep agent
    wandb.agent(sweep_id, function=lambda: train(save_best=False), count=count)
    
    print(f"\n✓ Sweep complete! View results at: https://wandb.ai/{WANDB_PROJECT}")
    print(f"  Sweep ID: {sweep_id}")


def train_with_best_config():
    """Train with default/best configuration and save the model."""
    print(f"\n{'='*60}")
    print("Training with Default Configuration")
    print(f"{'='*60}\n")
    
    default_config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "dropout_rate": 0.5,
        "optimizer": "Adam",
        "model_architecture": "DrowsinessCNN",
        "input_size": MODEL_INPUT_SIZE,
        "num_classes": 2,
        "dataset": "Driver Drowsiness Dataset (DDD)",
    }
    
    wandb.login(key=WANDB_API_KEY)
    
    with wandb.init(project=WANDB_PROJECT, name="default_training", config=default_config):
        best_val_acc = train(config=wandb.config, save_best=True)
        
        # Save model artifact to W&B
        artifact = wandb.Artifact(
            name="drowsiness_detection_model",
            type="model",
            description="PyTorch CNN model for driver drowsiness detection."
        )
        artifact.add_file(str(MODEL_PATH))
        artifact.metadata = {
            "best_val_accuracy": best_val_acc,
            "config": default_config,
        }
        wandb.log_artifact(artifact)
        print(f"✓ Model artifact saved to W&B")


def main():
    parser = argparse.ArgumentParser(description="Train drowsiness detection model")
    parser.add_argument(
        "--mode", 
        choices=["train", "sweep"], 
        default="train",
        help="'train' for normal training, 'sweep' for hyperparameter search"
    )
    parser.add_argument(
        "--sweep-count", 
        type=int, 
        default=10,
        help="Number of sweep trials to run (default: 10)"
    )
    args = parser.parse_args()
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"W&B Project: {WANDB_PROJECT}")
    
    wandb.login(key=WANDB_API_KEY)
    
    if args.mode == "sweep":
        run_sweep(count=args.sweep_count)
    else:
        train_with_best_config()


if __name__ == "__main__":
    main()
