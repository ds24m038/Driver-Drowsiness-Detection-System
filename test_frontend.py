#!/usr/bin/env python3
"""Test script to check frontend imports and basic functionality."""
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing frontend imports...")

try:
    print("1. Testing OpenCV import...")
    import cv2
    print("   ✓ OpenCV imported successfully")
    
    print("2. Testing face cascade loading...")
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade is not None:
        print("   ✓ Face cascade loaded successfully")
    else:
        print("   ✗ Face cascade failed to load")
    
    print("3. Testing Redis utils import...")
    from src.config.redis_utils import get_redis_client
    print("   ✓ Redis utils imported successfully")
    
    print("4. Testing Redis connection...")
    client = get_redis_client()
    result = client.ping()
    if result:
        print("   ✓ Redis connection successful")
    else:
        print("   ✗ Redis connection failed")
    
    print("5. Testing settings import...")
    from src.config.settings import ALARM_THRESHOLD_FRAMES, FPS
    print(f"   ✓ Settings imported: FPS={FPS}, Threshold={ALARM_THRESHOLD_FRAMES}")
    
    print("6. Testing frontend app import...")
    from src.frontend.app import main, load_face_cascade, detect_face
    print("   ✓ Frontend app imported successfully")
    
    print("\n✅ All imports successful!")
    print("Frontend should work correctly.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

