"""
Debug to test the exact constructor that worked
"""

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmlib import Body, draw_skeleton

# Use the correct image path
img_path = "data/images/demo.jpg"
img = cv2.imread(img_path)

if img is None:
    print(f"❌ Could not load image: {img_path}")
    sys.exit(1)

print(f"Image shape: {img.shape}")

# Approach that WORKED in your code - with to_openpose=True
body_estimator = Body(to_openpose=True, mode="balanced", backend="onnxruntime")
print("✅ Body estimator initialized with to_openpose=True, mode='balanced'")

# Test pose estimation
keypoints, scores = body_estimator(img)
print(f"Keypoints shape: {keypoints.shape}")
print(f"Scores shape: {scores.shape}")

# Test visualization
try:
    result = draw_skeleton(img.copy(), keypoints, scores, openpose_skeleton=True, kpt_thr=0.3)
    print("✅ Visualization: SUCCESS")
    cv2.imwrite("debug_success.jpg", result)
    print("✅ Result saved: debug_success.jpg")
except Exception as e:
    print(f"❌ Visualization: {e}")
