# Weights & Biases Model Loading Setup

This document explains how the system automatically loads models from Weights & Biases, eliminating the need to store large model files in the repository.

## How It Works

1. **On Startup**: When the API or inference worker starts, it checks for a local model file at `MODEL_PATH` (default: `/app/models/best_model.pth`)

2. **Automatic Download**: If the local file doesn't exist and `WANDB_API_KEY` is set, the system automatically downloads the model from W&B

3. **Fallback**: If W&B download fails or API key is not set, the system falls back to looking for a local model file

## Setup Instructions

### 1. Get Your W&B API Key

1. Go to https://wandb.ai/authorize
2. Copy your API key

### 2. Create Environment File

Create a `.env` file in the project root:

```bash
# Required: Your W&B API key
WANDB_API_KEY=your_wandb_api_key_here

# Optional: W&B project name (default: Driver-Drowsiness-Training)
WANDB_PROJECT=Driver-Drowsiness-Training

# Optional: Artifact version (default: latest)
# Options: "latest", "v1", "v2", or a specific version hash
WANDB_ARTIFACT_VERSION=latest
```

### 3. Start Docker Containers

```bash
docker compose up --build
```

The containers will automatically:
- Check for local model
- Download from W&B if needed
- Use the downloaded model for inference

## Artifact Information

- **Artifact Name**: `drowsiness_detection_model`
- **Model File**: `best_model.pth` (inside the artifact)
- **Project**: `Driver-Drowsiness-Training` (configurable via `WANDB_PROJECT`)

## Troubleshooting

### Model Not Downloading

1. **Check API Key**: Ensure `WANDB_API_KEY` is set correctly
2. **Check Project Name**: Verify `WANDB_PROJECT` matches your W&B project
3. **Check Artifact**: Ensure the artifact `drowsiness_detection_model` exists in your W&B project
4. **Check Logs**: Look for error messages in container logs:
   ```bash
   docker compose logs api
   docker compose logs inference_worker
   ```

### Using a Specific Artifact Version

To use a specific version instead of "latest":

```bash
# In .env file
WANDB_ARTIFACT_VERSION=v1

# Or via environment variable
export WANDB_ARTIFACT_VERSION=v1
docker compose up
```

### Local Model Override

If you have a local model file and want to use it instead of downloading from W&B:

1. Place your model at `models/best_model.pth`
2. The system will use the local file if it exists
3. To force W&B download, remove the local file first

## Benefits

✅ **No Large Files in Git**: Model files don't need to be committed to GitHub  
✅ **Automatic Updates**: Easy to switch between model versions  
✅ **Version Control**: W&B tracks all model versions  
✅ **Team Collaboration**: Team members can use the same model without sharing files  
✅ **CI/CD Friendly**: Automated deployments can download models automatically

