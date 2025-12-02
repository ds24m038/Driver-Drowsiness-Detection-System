"""Utilities for downloading models from Weights & Biases."""
import os
import logging
from pathlib import Path
from typing import Optional
import wandb

from src.config.settings import WANDB_PROJECT, WANDB_API_KEY, PROJECT_ROOT

logger = logging.getLogger(__name__)


def download_model_from_wandb(
    artifact_name: str = "drowsiness_detection_model",
    artifact_version: Optional[str] = None,
    download_dir: Optional[Path] = None,
    project: Optional[str] = None
) -> Optional[Path]:
    """Download model artifact from Weights & Biases.
    
    Args:
        artifact_name: Name of the W&B artifact (default: "drowsiness_detection_model")
        artifact_version: Version of the artifact (e.g., "latest", "v1", or None for latest)
        download_dir: Directory to download the model to (default: PROJECT_ROOT / "models")
        project: W&B project name (default: from settings)
        
    Returns:
        Path to downloaded model file, or None if download failed
    """
    if download_dir is None:
        download_dir = PROJECT_ROOT / "models"
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    if project is None:
        # Allow override via environment variable, otherwise use settings default
        project = os.getenv("WANDB_PROJECT", WANDB_PROJECT)
    
    # Check if API key is available
    api_key = WANDB_API_KEY or os.getenv("WANDB_API_KEY")
    if not api_key:
        logger.warning("WANDB_API_KEY not found. Cannot download model from W&B.")
        return None
    
    try:
        # Login to W&B (if not already logged in)
        wandb.login(key=api_key)
        
        # Initialize W&B API
        api = wandb.Api()
        
        # Get artifact version from environment or use provided/default
        artifact_version = artifact_version or os.getenv("WANDB_ARTIFACT_VERSION", "latest")
        
        # Construct artifact reference
        artifact_ref = f"{project}/{artifact_name}:{artifact_version}"
        
        logger.info(f"Downloading model artifact: {artifact_ref}")
        
        # Download artifact
        artifact = api.artifact(artifact_ref)
        artifact_dir = artifact.download(root=str(download_dir))
        artifact_dir = Path(artifact_dir)
        
        # Find the model file (best_model.pth)
        model_file = artifact_dir / "best_model.pth"
        if not model_file.exists():
            # Try to find any .pth file
            pth_files = list(artifact_dir.glob("*.pth"))
            if pth_files:
                model_file = pth_files[0]
                logger.info(f"Found model file: {model_file}")
            else:
                logger.error(f"No .pth file found in artifact directory: {artifact_dir}")
                return None
        
        logger.info(f"Model downloaded successfully to: {model_file}")
        return model_file
        
    except Exception as e:
        logger.error(f"Error downloading model from W&B: {e}")
        return None

