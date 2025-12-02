"""PyTorch CNN model for driver drowsiness detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import logging
import io

from src.config.settings import MODEL_INPUT_SIZE, MODEL_NUM_CLASSES, MODEL_CLASSES

logger = logging.getLogger(__name__)


class DrowsinessCNN(nn.Module):
    """CNN model for binary classification of driver drowsiness."""
    
    def __init__(self, num_classes=2):
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
        
        # Calculate flattened size after convolutions
        # Input: 227x227, after 4 pooling layers: 227/16 = 14.1875 -> 14
        # So: 256 * 14 * 14 = 50176
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
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


def get_transforms():
    """Get image preprocessing transforms for inference."""
    return transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def preprocess_image(image):
    """
    Preprocess an image for model inference.
    
    Args:
        image: PIL Image, numpy array, or bytes
        
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input
    """
    transform = get_transforms()
    
    # Handle different input types
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def load_model(model_path, device=None):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the model file (.pth)
        device: torch device (cpu or cuda), auto-detected if None
        
    Returns:
        DrowsinessCNN: Loaded model in evaluation mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Initialize model
    model = DrowsinessCNN(num_classes=MODEL_NUM_CLASSES)
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {model_path} on device {device}")
    return model


def predict(model, image, device=None):
    """
    Run inference on an image.
    
    Args:
        model: DrowsinessCNN model
        image: Preprocessed image tensor or raw image
        device: torch device
        
    Returns:
        dict: Prediction results with class, confidence, and probabilities
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Preprocess if needed
    if not isinstance(image, torch.Tensor):
        image = preprocess_image(image)
    
    image = image.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get class label
    predicted_class_idx = predicted.item()
    predicted_class = MODEL_CLASSES[predicted_class_idx]
    
    # Get probabilities for both classes
    probs = probabilities[0].cpu().numpy()
    class_probs = {
        MODEL_CLASSES[i]: float(probs[i])
        for i in range(len(MODEL_CLASSES))
    }
    
    return {
        "prediction": predicted_class,
        "confidence": float(confidence.item()),
        "probabilities": class_probs
    }

