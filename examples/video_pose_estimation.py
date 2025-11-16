
"""
video_pose_estimation.py
Process videos with 2D pose estimation and save results

Usage:
    python examples/video_pose_estimation.py
"""

import cv2
import numpy as np
import os
import sys
import time

# Add the parent directory to Python path to import pjpose2d modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmlib import Body, draw_skeleton

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("ℹ️ tqdm not installed, progress bars disabled")
    print("Install with: pip install tqdm")

class VideoPoseEstimator:
    def __init__(self, model_config='balanced', backend='onnxruntime'):
        """Initialize the video pose estimator"""
        self.body_estimator = Body(
            to_openpose=True, 
            mode=model_config, 
            backend=backend
        )
        self.fps = 0
        self.frame_count = 0
        self.resolution = (0, 0)
    
    def process_video(self, input_path, output_path, show_progress=True):
        """
        Process a video file and save pose estimation results
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to output video
            show_progress (bool): Whether to show progress bar
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
        
        print(f"📹 Processing Video:")
        print(f"   - Input: {input_path}")
        print(f"   - Output: {output_path}")
        print(f"   - Resolution: {width}x{height}")
        print(f"   - FPS: {self.fps:.1f}")
        print(f"   - Frames: {self.frame_count}")
        print(f"   - Duration: {self.frame_count/self.fps:.1f}s")
        
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
                # Run pose estimation
                keypoints, scores = self.body_estimator(frame)
                
                # Draw skeleton on frame
                result_frame = draw_skeleton(
                    frame, 
                    keypoints, 
                    scores, 
                    openpose_skeleton=True, 
                    kpt_thr=0.3
                )
                
                # Write frame to output
                out.write(result_frame)
                successful_frames += 1
                
            except Exception as e:
                print(f"⚠️ Error processing frame: {e}")
                # Write original frame if processing fails
                out.write(frame)
            
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            
            if progress_bar:
                progress_bar.update(1)
            elif show_progress and frame_count % 10 == 0:
                print(f"Processed {frame_count}/{self.frame_count} frames...")
        
        # Cleanup
        cap.release()
        out.release()
        if progress_bar:
            progress_bar.close()
        
        # Print statistics
        if frame_times:
            avg_time = np.mean(frame_times) * 1000
            avg_fps = 1000 / avg_time if avg_time > 0 else 0
            
            print(f"\n📊 Processing Complete:")
            print(f"   - Successful frames: {successful_frames}/{self.frame_count}")
            print(f"   - Average processing time: {avg_time:.1f}ms per frame")
            print(f"   - Average FPS: {avg_fps:.1f}")
            print(f"   - Total processing time: {sum(frame_times):.1f}s")
            print(f"   - Output saved to: {output_path}")
        
        return True

def main():
    """Main function to demonstrate video pose estimation"""
    print("🎯 Video Pose Estimation Demo")
    print("=" * 40)
    
    # Input and output paths (relative to package root)
    input_video = "data/videos/sample_video.mp4"
    output_video = "pose_estimation_results/sample_video_pose.mp4"
    
    # Create output directory
    os.makedirs("pose_estimation_results", exist_ok=True)
    
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        print("Please make sure the sample video is in the correct location.")
        print("You can download it using the provided script.")
        return
    
    # Initialize pose estimator
    print("🔄 Initializing pose estimator...")
    estimator = VideoPoseEstimator(model_config='balanced', backend='onnxruntime')
    
    # Process video
    print("🎬 Starting video processing...")
    success = estimator.process_video(input_video, output_video)
    
    if success:
        print("\n🎉 Video processing completed successfully!")
        print(f"📁 Output: {output_video}")
    else:
        print("\n❌ Video processing failed!")

if __name__ == "__main__":
    main()
