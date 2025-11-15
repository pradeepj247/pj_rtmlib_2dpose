#!/bin/bash
echo "🚀 Setting up RTMPose on Google Colab..."

echo "📦 Step 1: Installing dependencies..."
pip install -r requirements.txt

echo "🔧 Step 2: Setting up RTMLib..."
git clone https://github.com/Tau-J/rtmlib.git
cd rtmlib
pip install -e .
cd ..

echo "🎯 Step 3: Downloading and converting YOLOv8..."
python -c "
from ultralytics import YOLO
import os
model = YOLO('yolov8s.pt')
model.export(format='onnx', opset=12, simplify=True, dynamic=False)
os.rename('yolov8s.onnx', 'rtmlib/yolov8s.onnx')
"

echo "✅ Setup complete!"
echo "💡 Run: python examples/single_image_demo.py to test"
