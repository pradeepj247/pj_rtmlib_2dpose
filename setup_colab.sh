#!/bin/bash
echo "🚀 PJPose2D - Complete Setup (All-in-One)"

echo "🔧 Step 0: Resolving ONNX Runtime conflicts..."
echo "Removing any conflicting ONNX Runtime CPU version..."
pip uninstall -y onnxruntime

echo "Installing ONNX Runtime GPU version..."
pip install onnxruntime-gpu==1.23.0

echo "📦 Step 1: Installing package and dependencies..."
pip install -e .

echo "📁 Step 2: Creating models directory..."
mkdir -p models

echo "🎯 Step 3: Downloading YOLOv8s weights (ONNX conversion skipped)..."
python -c "
from ultralytics import YOLO
import os

# Check if model already exists
if os.path.exists('models/yolov8s.pt'):
    print('✅ YOLOv8s.pt already exists in models/')
else:
    print('Downloading YOLOv8s weights...')
    try:
        # Just download the weights, no export to ONNX
        model = YOLO('yolov8s.pt')
        # Move to models folder if it was downloaded to current directory
        if os.path.exists('yolov8s.pt'):
            import shutil
            shutil.move('yolov8s.pt', 'models/yolov8s.pt')
            print('✅ YOLOv8s.pt saved to models/yolov8s.pt')
        else:
            print('✅ YOLOv8s weights are ready for use')
    except Exception as e:
        print(f'❌ Error during YOLOv8 setup: {e}')
"

echo "🔍 Step 4: Running installation validation..."
python validate_installation.py

echo ""
echo "✅ Setup complete! Ready to use PJPose2D."
echo "📁 Models are organized in: models/"
echo "💡 If you encounter any issues, restart the runtime and run the validation script again."
