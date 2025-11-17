"""
OPTIMIZED Video Pose Estimation using the proven pipeline
Uses EXACT same pattern as ultimate_pipeline.py
"""

import cv2
import numpy as np
import os
import sys
import time
from ultralytics import YOLO

# Add the parent directory to Python path to import pjpose2d modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmlib import Body, draw_skeleton

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("ℹ️ tqdm not installed, progress bars disabled")

class OptimizedVideoPoseEstimator:
    def __init__(self):
        """Initialize with the EXACT proven configuration from ultimate pipeline"""
        print("🚀 Initializing OPTIMIZED Video Pose Estimator")
        print("Using EXACT pattern from ultimate pipeline")
        
        # EXACT PROVEN CONFIGURATION from ultimate_pipeline.py
        self.detector = YOLO('models/yolov8s.pt')  # Our fast detection
        self.pose_estimator = Body(
            backend='onnxruntime',
            device='cuda'  # CRITICAL FOR GPU ACCELERATION
        )
        # We ONLY use pose_estimator.pose_model() - NOT the full Body pipeline
        
        print(f"✅ Detector: YOLOv8s (PyTorch GPU)")
        print(f"✅ Pose model: CUDA enabled")
        
        self.fps = 0
        self.frame_count = 0
        self.resolution = (0, 0)
        
        # WARM-UP with a dummy frame (proven approach)
        print("🔥 Warming up models...")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _ = self.detector(dummy_frame, classes=[0])
        dummy_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        _ = self.pose_estimator.pose_model(dummy_crop)
        print("✅ Models warmed up")
    
    def get_largest_person_crop(self, frame):
        """EXACT same approach as ultimate pipeline"""
        # Run detection (proven approach)
        results = self.detector(frame, classes=[0])  # class 0 = person
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) == 0:
            return None, None
        
        # Pick LARGEST bbox only (proven optimization)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_idx = np.argmax(areas)
        x1, y1, x2, y2 = map(int, boxes[largest_idx])
        
        # Extract crop
        crop = frame[y1:y2, x1:x2]
        
        return crop, (x1, y1, x2, y2)
    
    def process_video(self, input_path, output_path, show_progress=True):
        """
        Process video using the EXACT proven pipeline pattern
        """
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {input_path}")
            return False
        
        # Get video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution = (width, height)
        
        print(f"📹 Processing Video (OPTIMIZED):")
        print(f"   - Input: {input_path}")
        print(f"   - Output: {output_path}")
        print(f"   - Resolution: {width}x{height}")
        print(f"   - FPS: {self.fps:.1f}")
        print(f"   - Frames: {self.frame_count}")
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        # Process frames
        frame_times = []
        successful_frames = 0
        
        progress_bar = tqdm(total=self.frame_count, desc="Processing frames") if (show_progress and HAS_TQDM) else None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            try:
                # EXACT PROVEN PIPELINE: Detection + Pose on largest person
                crop, bbox = self.get_largest_person_crop(frame)
                
                if crop is not None and crop.size > 0:
                    # Run pose estimation on crop using EXACT same call as ultimate pipeline
                    keypoints, scores = self.pose_estimator.pose_model(crop)
                    
                    # Adjust keypoints to original coordinates (proven approach)
                    x1, y1, x2, y2 = bbox
                    keypoints[:, :, 0] += x1
                    keypoints[:, :, 1] += y1
                    
                    # Draw skeleton on original frame
                    result_frame = draw_skeleton(
                        frame, 
                        keypoints, 
                        scores, 
                        openpose_skeleton=True, 
                        kpt_thr=0.3
                    )
                    
                    # Draw bounding box
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                else:
                    # No person detected, use original frame
                    result_frame = frame
                
                # Write frame to output
                out.write(result_frame)
                successful_frames += 1
                
            except Exception as e:
                print(f"⚠️ Error processing frame: {e}")
                out.write(frame)  # Write original frame if processing fails
            
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            
            if progress_bar:
                progress_bar.update(1)
        
        # Cleanup
        cap.release()
        out.release()
        if progress_bar:
            progress_bar.close()
        
        # Print statistics
        if frame_times:
            avg_time = np.mean(frame_times) * 1000
            avg_fps = 1000 / avg_time if avg_time > 0 else 0
            
            print(f"\n📊 OPTIMIZED Processing Complete:")
            print(f"   - Successful frames: {successful_frames}/{self.frame_count}")
            print(f"   - Average processing time: {avg_time:.1f}ms per frame")
            print(f"   - Average FPS: {avg_fps:.1f}")
            print(f"   - Real-time factor: {avg_fps/self.fps:.2f}x")
        
        return True

def main():
    """Main function to demonstrate OPTIMIZED video pose estimation"""
    print("🎯 OPTIMIZED Video Pose Estimation Demo")
    print("Using EXACT pattern from ultimate pipeline")
    print("=" * 50)
    
    # Input and output paths
    input_video = "data/videos/sample_video.mp4"
    output_video = "pose_estimation_results/optimized_sample_video_pose.mp4"
    
    # Create output directory
    os.makedirs("pose_estimation_results", exist_ok=True)
    
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return
    
    # Initialize OPTIMIZED pose estimator
    print("🔄 Initializing OPTIMIZED pose estimator...")
    estimator = OptimizedVideoPoseEstimator()
    
    # Process video
    print("🎬 Starting OPTIMIZED video processing...")
    success = estimator.process_video(input_video, output_video)
    
    if success:
        print("\n🎉 OPTIMIZED video processing completed successfully!")
        print(f"📁 Output: {output_video}")
    else:
        print("\n❌ Video processing failed!")

if __name__ == "__main__":
    main()
