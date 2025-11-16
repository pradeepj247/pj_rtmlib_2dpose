"""
test_pose_estimation.py
Comprehensive pose estimation validation test
"""

import cv2
import numpy as np
from rtmlib import Body, draw_skeleton
import onnxruntime as ort
import os
import time

def test_pose_estimation():
    """Comprehensive test of pose estimation functionality"""
    print("🧪 POSE ESTIMATION COMPREHENSIVE TEST")
    print("=" * 50)
    
    # 1. System check
    print("1. System and dependency check...")
    try:
        providers = ort.get_available_providers()
        print(f"   ✅ ONNX Runtime providers: {providers}")
    except Exception as e:
        print(f"   ❌ ONNX Runtime check failed: {e}")
        return False
    
    # 2. File check
    print("2. Required files check...")
    required_files = [
        "models/yolov8s.onnx",
        "data/images/demo.jpg"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)
            print(f"   ✅ {file_path}: {size:.1f} MB")
        else:
            print(f"   ❌ {file_path}: Missing!")
            return False
    
    # 3. Core functionality test
    print("3. Core pose estimation test...")
    try:
        # Load image
        img_path = "data/images/demo.jpg"
        img = cv2.imread(img_path)
        if img is None:
            print(f"   ❌ Failed to load image: {img_path}")
            return False
        
        print(f"   ✅ Image loaded: {img.shape}")
        
        # Initialize pose estimator
        body_estimator = Body(to_openpose=True, mode="balanced", backend="onnxruntime")
        print("   ✅ Models initialized successfully")
        
        # Run pose estimation with performance measurement
        start_time = time.time()
        keypoints, scores = body_estimator(img)
        inference_time = (time.time() - start_time) * 1000
        
        print(f"   ✅ Pose estimation successful:")
        print(f"      - Keypoints shape: {keypoints.shape}")
        print(f"      - Scores shape: {scores.shape}")
        print(f"      - Persons detected: {keypoints.shape[0]}")
        print(f"      - Keypoints per person: {keypoints.shape[1]}")
        print(f"      - Inference time: {inference_time:.1f}ms")
        print(f"      - Estimated FPS: {1000/inference_time:.1f}")
        
        # 4. Visualization test
        print("4. Visualization test...")
        img_show = img.copy()
        
        # Use openpose_skeleton=True for 18 keypoints with to_openpose=True
        vis_img = draw_skeleton(
            img_show,
            keypoints,
            scores,
            openpose_skeleton=True,  # Required for 18 keypoints
            kpt_thr=0.5
        )
        
        # Save visualization result
        cv2.imwrite("pose_test_result.jpg", vis_img)
        print("   ✅ Visualization and save successful")
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 50)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("✅ PJPose2D is fully functional!")
    return True

if __name__ == "__main__":
    success = test_pose_estimation()
    exit(0 if success else 1)