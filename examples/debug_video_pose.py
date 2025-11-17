"""
Debug version to see the actual error
"""

import cv2
import numpy as np
import os
import sys
import traceback
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmlib import Body

# Test with just one frame to see the exact error
detector = YOLO('models/yolov8s.pt')
pose_estimator = Body(backend='onnxruntime', device='cuda')

# Load a test frame
cap = cv2.VideoCapture("data/videos/sample_video.mp4")
ret, frame = cap.read()
cap.release()

if ret:
    print("Frame loaded successfully")
    
    # Test detection
    results = detector(frame, classes=[0])
    boxes = results[0].boxes.xyxy.cpu().numpy()
    print(f"Detected {len(boxes)} persons")
    
    if len(boxes) > 0:
        # Get largest person
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_idx = np.argmax(areas)
        x1, y1, x2, y2 = map(int, boxes[largest_idx])
        crop = frame[y1:y2, x1:x2]
        print(f"Crop shape: {crop.shape}")
        
        # Test pose estimation - this is where the error happens
        try:
            print("Testing pose estimation...")
            keypoints, scores = pose_estimator.pose_model(crop)
            print("SUCCESS! Pose estimation worked")
            print(f"Keypoints shape: {keypoints.shape}")
        except Exception as e:
            print("ERROR in pose estimation:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("Full traceback:")
            traceback.print_exc()
else:
    print("Failed to load frame")
