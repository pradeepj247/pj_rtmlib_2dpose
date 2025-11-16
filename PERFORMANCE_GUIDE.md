# 🚀 Performance Optimization Guide

## Quick Start (83 FPS Pose Estimation)

```python
import cv2
import numpy as np
from rtmlib import Body
from ultralytics import YOLO

# Initialize models
detector = YOLO('yolov8s.pt')
pose_estimator = Body(backend='onnxruntime', device='cuda')

# Load image
img = cv2.imread('path/to/image.jpg')

# Detect persons
results = detector(img, classes=[0])
boxes = results[0].boxes.xyxy.cpu().numpy()

# Get largest person
areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
largest_idx = np.argmax(areas)
x1, y1, x2, y2 = map(int, boxes[largest_idx])
crop = img[y1:y2, x1:x2]

# Pose estimation (83 FPS!)
keypoints, scores = pose_estimator.pose_model(crop)
```

## Key Configuration

### requirements.txt
```
onnxruntime-gpu==1.23.0
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.21.0,<2.3.0
torch>=1.8.0
```

### Body Initialization
```python
# ✅ CORRECT (83 FPS)
Body(backend='onnxruntime', device='cuda')

# ❌ WRONG (0.6 FPS)
Body(backend='onnxruntime')  # Missing device='cuda'
```

## Performance Tips

1. **Always use device='cuda'** in Body initialization
2. **Warm up models** before timing
3. **Process cropped images** not full images
4. **Use YOLOv8s.pt** for fast detection
5. **Test with multiple runs** for accurate FPS

## Expected Performance
- Pose Estimation: 80-90 FPS
- Detection: 40-50 FPS
- Total Pipeline: 25-30 FPS