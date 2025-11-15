"""
Pose Estimation - GPU Accelerated Version (Fixed)
"""

import cv2
import numpy as np
from rtmlib import Body, draw_skeleton
import os
import onnxruntime as ort

def main():
    print("🚀 Starting GPU-accelerated pose estimation...")
    
    # Check available providers
    providers = ort.get_available_providers()
    print(f"✅ Available ONNX Runtime providers: {providers}")
    
    # Use CUDA if available, otherwise CPU
    if 'CUDAExecutionProvider' in providers:
        device = "cuda"
        print("🎯 Using CUDA acceleration")
    else:
        device = "cpu"
        print("⚠️ CUDA not available, using CPU")
    
    # Load image
    img_path = "data/images/demo.jpg"
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return
    
    img = cv2.imread(img_path)
    print(f"📷 Image loaded: {img.shape}")
    
    # Initialize pose estimator
    print("🔄 Initializing models...")
    body = Body(
        det="models/yolov8s.onnx",
        det_input_size=(640, 640),
        pose=("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
              "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"),
        pose_input_size=(192, 256),
        backend="onnxruntime",
        device=device  # Use the detected device
    )
    print(f"✅ Models initialized with {device.upper()}")
    
    # Run pose estimation
    print("🎯 Running pose estimation...")
    keypoints, scores = body.pose_model(img)
    print(f"✅ Pose estimation complete")
    print(f"   Keypoints: {keypoints.shape}, Scores: {scores.shape}")
    
    # Visualize results
    print("🖼️ Displaying results...")
    img_show = img.copy()
    img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)
    
    # Save result
    cv2.imwrite("pose_result.jpg", img_show)
    print("💾 Result saved as 'pose_result.jpg'")
    
    print("🎉 POSE ESTIMATION SUCCESSFUL!")
    return keypoints, scores

if __name__ == "__main__":
    main()
