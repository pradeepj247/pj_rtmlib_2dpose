# validate_installation.py
"""
PJPose2D - Installation Validation Script
Validates that all components are installed correctly and functioning.
"""

import sys
import os
import subprocess

def get_onnxruntime_version():
    """Get ONNX Runtime version from pip show."""
    try:
        result = subprocess.run(['pip', 'show', 'onnxruntime-gpu'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split()[-1]
        return "Unknown (pip show failed)"
    except:
        return "Unknown (error running pip show)"

def validate_installation():
    print("🔍 PJPose2D - Installation Validation")
    print("=" * 50)
    
    # 1. Check Python version
    print(f"✓ Python version: {sys.version}")
    
    # 2. Check critical package versions
    try:
        import onnxruntime as ort
        onnx_version = get_onnxruntime_version()
        print(f"✓ ONNX Runtime: {onnx_version}")
        print(f"  Available providers: {ort.get_available_providers()}")
    except ImportError as e:
        print(f"❌ ONNX Runtime import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ ONNX Runtime error: {e}")
        return False
    
    try:
        import ultralytics
        print(f"✓ Ultralytics: {ultralytics.__version__}")
    except ImportError as e:
        print(f"❌ Ultralytics import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU device: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    # 3. Check YOLO ONNX model
    print("\n📦 Model Verification:")
    if os.path.exists('models/yolov8s.onnx'):
        model_size = os.path.getsize('models/yolov8s.onnx') / (1024*1024)
        print(f"✅ YOLOv8s ONNX model: {model_size:.1f} MB")
        
        # Verify model can be loaded
        try:
            session = ort.InferenceSession('models/yolov8s.onnx')
            inputs = [input.name for input in session.get_inputs()]
            outputs = [output.name for output in session.get_outputs()]
            print(f"✅ Model loads successfully")
            print(f"  Inputs: {inputs}")
            print(f"  Outputs: {outputs}")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    else:
        print("❌ YOLOv8s ONNX model not found in models/")
        return False
    
    # 4. Check package structure
    print("\n📁 Package Structure:")
    essential_files = [
        'setup.py', 'requirements.txt', 'setup_colab.sh',
        'examples/pose_estimation.py', 'rtmlib/'
    ]
    
    for file_path in essential_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")
            return False
    
    print("\n🎉 All validation checks passed! Installation is successful.")
    print("🚀 You can now run the examples from the 'examples/' directory.")
    return True

if __name__ == "__main__":
    success = validate_installation()
    sys.exit(0 if success else 1)