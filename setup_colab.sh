#!/bin/bash
echo "🚀 RTMPose Complete Setup (All-in-One)"

echo "📦 Step 1: Installing dependencies..."
pip install -r requirements.txt

echo "🔧 Step 2: Setting up RTMLib (local copy)..."
cd rtmlib
pip install -e .
cd ..

echo "🎯 Step 3: Setting up YOLOv8..."
python -c "
from ultralytics import YOLO
import os
model = YOLO('yolov8s.pt')
model.export(format='onnx', opset=12, simplify=True, dynamic=False)
os.rename('yolov8s.onnx', 'rtmlib/yolov8s.onnx')
"

echo "✅ Setup complete! Ready to use."
