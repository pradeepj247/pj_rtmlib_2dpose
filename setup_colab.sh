#!/bin/bash
echo "🚀 PJ RTMLib 2D Pose - Complete Setup (All-in-One)"

echo "🔧 Step 0: Resolving ONNX Runtime conflicts..."
echo "Removing any conflicting ONNX Runtime CPU version..."
pip uninstall -y onnxruntime

echo "Installing ONNX Runtime GPU version..."
pip install onnxruntime-gpu==1.19.2

echo "📦 Step 1: Installing package and dependencies..."
pip install -e .

echo "📁 Step 2: Creating models directory..."
mkdir -p models

echo "🎯 Step 3: Setting up YOLOv8..."
python -c "
from ultralytics import YOLO
import os
import shutil

# Check if model already exists
if os.path.exists('models/yolov8s.onnx'):
    print('✅ YOLOv8s ONNX model already exists in models/')
else:
    print('Downloading and converting YOLOv8s to ONNX...')
    try:
        # Download and convert directly
        model = YOLO('yolov8s.pt')
        model.export(format='onnx', opset=12, simplify=True, dynamic=False)
        
        # Move to models folder
        if os.path.exists('yolov8s.onnx'):
            shutil.move('yolov8s.onnx', 'models/yolov8s.onnx')
            print('✅ YOLOv8s ONNX model saved to models/yolov8s.onnx')
            
            # Clean up the .pt file (optional - remove this if you want to keep it)
            if os.path.exists('yolov8s.pt'):
                os.remove('yolov8s.pt')
                print('🧹 Cleaned up temporary files')
        else:
            print('❌ YOLOv8s ONNX file was not created')
    except Exception as e:
        print(f'❌ Error during YOLOv8 setup: {e}')
"

echo "🔍 Step 4: Running installation validation..."
python validate_installation.py

echo ""
echo "✅ Setup complete! Ready to use PJ RTMLib 2D Pose."
echo "📁 Models are organized in: models/"
echo "💡 If you encounter any issues, restart the runtime and run the validation script again."
