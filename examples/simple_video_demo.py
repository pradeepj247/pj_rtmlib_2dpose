
"""
simple_video_demo.py
Simple demo for video pose estimation with pjpose2d

Usage:
    python examples/simple_video_demo.py
"""

import cv2
import os
import sys

# Add the parent directory to Python path to import pjpose2d modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmlib import Body, draw_skeleton

def quick_video_demo():
    """Quick demo showing pose estimation on video"""
    print("🚀 Quick Video Pose Estimation Demo")
    
    # Setup paths (relative to package root)
    input_video = "data/videos/sample_video.mp4"
    output_dir = "demo_results"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_video):
        print("❌ Sample video not found. Please download it first.")
        print("The sample video should be at: data/videos/sample_video.mp4")
        return
    
    # Initialize pose estimator
    print("📦 Loading pose estimation model...")
    body = Body(to_openpose=True, mode="balanced", backend="onnxruntime")
    
    # Process first 50 frames as a demo
    cap = cv2.VideoCapture(input_video)
    frame_count = 0
    max_frames = 50
    
    print(f"🎬 Processing first {max_frames} frames...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run pose estimation
        keypoints, scores = body(frame)
        
        # Draw results
        result_frame = draw_skeleton(frame, keypoints, scores, 
                                   openpose_skeleton=True, kpt_thr=0.3)
        
        # Save sample frame
        if frame_count == 25:  # Save frame 25 as example
            cv2.imwrite(f"{output_dir}/sample_frame.jpg", result_frame)
            print(f"💾 Saved sample frame to: {output_dir}/sample_frame.jpg")
        
        frame_count += 1
    
    cap.release()
    print(f"✅ Processed {frame_count} frames!")
    print(f"📁 Results saved to: {output_dir}/")

if __name__ == "__main__":
    quick_video_demo()
