#!/usr/bin/env python3
"""Validation script to check if all components are properly set up."""
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    path = Path(filepath)
    if path.exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} (MISSING)")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    path = Path(dirpath)
    if path.exists() and path.is_dir():
        print(f"✓ {description}: {dirpath}")
        return True
    else:
        print(f"✗ {description}: {dirpath} (MISSING)")
        return False

def check_import(module_name, description):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {description}: {module_name} (IMPORT ERROR: {e})")
        return False

def main():
    """Run validation checks."""
    print("=" * 60)
    print("Driver Drowsiness Detection System - Setup Validation")
    print("=" * 60)
    print()
    
    all_checks_passed = True
    
    # Check project structure
    print("1. Project Structure")
    print("-" * 60)
    all_checks_passed &= check_directory_exists("Data/Drowsy", "Drowsy data directory")
    all_checks_passed &= check_directory_exists("Data/Non Drowsy", "Non Drowsy data directory")
    all_checks_passed &= check_directory_exists("notebooks", "Notebooks directory")
    all_checks_passed &= check_directory_exists("src", "Source directory")
    all_checks_passed &= check_directory_exists("src/backend", "Backend directory")
    all_checks_passed &= check_directory_exists("src/frontend", "Frontend directory")
    all_checks_passed &= check_directory_exists("src/inference_worker", "Inference worker directory")
    all_checks_passed &= check_directory_exists("src/alarm_manager", "Alarm manager directory")
    all_checks_passed &= check_directory_exists("src/models", "Models directory")
    all_checks_passed &= check_directory_exists("src/config", "Config directory")
    all_checks_passed &= check_directory_exists("docker", "Docker directory")
    all_checks_passed &= check_directory_exists("models", "Models storage directory")
    print()
    
    # Check key files
    print("2. Key Files")
    print("-" * 60)
    all_checks_passed &= check_file_exists("docker-compose.yml", "Docker Compose file")
    all_checks_passed &= check_file_exists("requirements.txt", "Requirements file")
    all_checks_passed &= check_file_exists("requirements-dev.txt", "Dev requirements file")
    all_checks_passed &= check_file_exists(".env.example", "Environment example file")
    all_checks_passed &= check_file_exists("README.md", "README file")
    all_checks_passed &= check_file_exists("HOWTO.md", "HOWTO file")
    print()
    
    # Check source files
    print("3. Source Files")
    print("-" * 60)
    all_checks_passed &= check_file_exists("src/backend/main.py", "FastAPI backend")
    all_checks_passed &= check_file_exists("src/frontend/app.py", "Streamlit frontend")
    all_checks_passed &= check_file_exists("src/inference_worker/worker.py", "Inference worker")
    all_checks_passed &= check_file_exists("src/alarm_manager/manager.py", "Alarm manager")
    all_checks_passed &= check_file_exists("src/models/cnn_model.py", "CNN model")
    all_checks_passed &= check_file_exists("src/config/settings.py", "Settings")
    all_checks_passed &= check_file_exists("src/config/redis_client.py", "Redis client")
    print()
    
    # Check Dockerfiles
    print("4. Dockerfiles")
    print("-" * 60)
    all_checks_passed &= check_file_exists("docker/Dockerfile.backend", "Backend Dockerfile")
    all_checks_passed &= check_file_exists("docker/Dockerfile.frontend", "Frontend Dockerfile")
    all_checks_passed &= check_file_exists("docker/Dockerfile.inference_worker", "Inference worker Dockerfile")
    all_checks_passed &= check_file_exists("docker/Dockerfile.alarm_manager", "Alarm manager Dockerfile")
    print()
    
    # Check notebooks
    print("5. Jupyter Notebooks")
    print("-" * 60)
    all_checks_passed &= check_file_exists("notebooks/01_data_exploration.ipynb", "Data exploration notebook")
    all_checks_passed &= check_file_exists("notebooks/02_model_training.ipynb", "Model training notebook")
    all_checks_passed &= check_file_exists("notebooks/03_inference_demo.ipynb", "Inference demo notebook")
    all_checks_passed &= check_file_exists("notebooks/04_wandb_reporting.ipynb", "W&B reporting notebook")
    print()
    
    # Check model (optional - may not exist if not trained yet)
    print("6. Trained Model (Optional)")
    print("-" * 60)
    model_exists = check_file_exists("models/best_model.pth", "Trained model")
    if not model_exists:
        print("  ⚠ Note: Model not found. Train the model using notebooks/02_model_training.ipynb")
    print()
    
    # Check imports (add project root to path first)
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    print("7. Python Imports (Requires dependencies)")
    print("-" * 60)
    import_warnings = []
    
    try:
        from src.config import settings
        print("✓ Settings module: src.config.settings")
    except ImportError as e:
        print(f"⚠ Settings module: Missing dependencies (install requirements.txt)")
        import_warnings.append("Install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"✗ Settings module: {e}")
        all_checks_passed = False
    
    try:
        from src.models import cnn_model
        print("✓ CNN model module: src.models.cnn_model")
    except ImportError as e:
        print(f"⚠ CNN model module: Missing dependencies (install requirements.txt)")
    except Exception as e:
        print(f"✗ CNN model module: {e}")
        all_checks_passed = False
    
    try:
        from src.config import redis_client
        print("✓ Redis client module: src.config.redis_client")
    except ImportError as e:
        print(f"⚠ Redis client module: Missing dependencies (install requirements.txt)")
    except Exception as e:
        print(f"✗ Redis client module: {e}")
        all_checks_passed = False
    print()
    
    # Summary
    print("=" * 60)
    if all_checks_passed:
        print("✓ All structural checks passed! System is ready for deployment.")
        print()
        if import_warnings:
            print("⚠ Note: Some Python dependencies are missing.")
            print("   Install them with: pip install -r requirements.txt")
            print()
        print("Next steps:")
        print("1. Install dependencies: pip install -r requirements-dev.txt")
        print("2. Train the model: jupyter notebook notebooks/02_model_training.ipynb")
        print("3. Start services: docker compose up --build")
        print("4. Access UI: http://localhost:8501")
    else:
        print("✗ Some structural checks failed. Please review the errors above.")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()

