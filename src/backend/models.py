"""PyTorch CNN model for driver drowsiness detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
from src.config.settings import MODEL_PATH, MODEL_INPUT_SIZE, NUM_CLASSES, CLASS_NAMES, DROWSY_THRESHOLD


class DrowsinessCNN(nn.Module):
    """Convolutional Neural Network for drowsiness detection.
    
    Architecture:
    - Input: 3x227x227 RGB image
    - Output: 2 classes (alert, drowsy)
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES):
        super(DrowsinessCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 4 pooling operations: 227 -> 113 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Convolutional block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Convolutional block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Convolutional block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Convolutional block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 256 * 14 * 14)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def create_model(num_classes: int = NUM_CLASSES) -> DrowsinessCNN:
    """Create a new model instance.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Initialized model
    """
    return DrowsinessCNN(num_classes=num_classes)


def load_model(model_path: Optional[str] = None, device: str = "cpu") -> DrowsinessCNN:
    """Load a trained model from disk.
    
    Args:
        model_path: Path to model checkpoint. If None, uses default from settings.
        device: Device to load model on ("cpu" or "cuda")
        
    Returns:
        Loaded model in evaluation mode
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model
    model = create_model()
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def predict_image(model: nn.Module, image_tensor: torch.Tensor, device: str = "cpu", drowsy_threshold: float = DROWSY_THRESHOLD) -> tuple:
    """Run inference on a single image.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor (3, 227, 227)
        device: Device to run inference on
        drowsy_threshold: Threshold for drowsy prediction (lower = more pessimistic)
        
    Returns:
        Tuple of (predicted_class, confidence, probabilities)
        predicted_class: "alert" or "drowsy"
        confidence: Confidence score for the predicted class
        probabilities: Dict with probabilities for each class
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Add batch dimension if needed
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Forward pass
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get probabilities for each class
        alert_prob = probabilities[0][0].item()  # Index 0 = alert
        drowsy_prob = probabilities[0][1].item()  # Index 1 = drowsy
        
        # Pessimistic prediction: if drowsy probability exceeds threshold, classify as drowsy
        # This makes the model more likely to predict drowsy (more pessimistic)
        if drowsy_prob >= drowsy_threshold:
            predicted_class = "drowsy"
            predicted_idx = 1
            confidence = drowsy_prob
        else:
            predicted_class = "alert"
            predicted_idx = 0
            confidence = alert_prob
        
        # Get all probabilities
        prob_dict = {
            "alert": alert_prob,
            "drowsy": drowsy_prob,
        }
    
    return predicted_class, confidence, prob_dict

