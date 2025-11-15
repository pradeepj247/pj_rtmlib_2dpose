
"""
Video processing pipeline for RTMPose pose estimation
"""

import cv2
import numpy as np
import os
from pose_estimation import PoseEstimator

class VideoProcessor:
    def __init__(self, device="cuda"):
        self.device = device
        self.pose_estimator = PoseEstimator(device)
        
    def process_video(self, input_path, output_path, max_frames=None, show_progress=True):
        """
        Process video file for pose estimation
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            max_frames: Maximum frames to process (None for all)
            show_progress: Show progress updates
        """
        # Initialize video capture
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"🎥 Processing video: {input_path}")
        print(f"📊 FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}")
        
        frame_count = 0
        processing_times = []
        
        # Initialize models
        self.pose_estimator.initialize_models()
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= total_frames:
                break
                
            # Process frame
            start_time = time.time()
            
            # Detect persons and estimate pose
            boxes, crop, bbox = self.pose_estimator.detect_persons(frame)
            
            if crop is not None:
                keypoints, scores = self.pose_estimator.estimate_pose(crop)
                if keypoints is not None:
                    frame = self.pose_estimator.draw_results(frame, crop, bbox, keypoints, scores)
            
            processing_time = (time.time() - start_time) * 1000
            processing_times.append(processing_time)
            
            # Write frame to output
            out.write(frame)
            
            frame_count += 1
            if show_progress and frame_count % 30 == 0:
                avg_time = np.mean(processing_times[-30:])
                print(f"📹 Processed {frame_count}/{total_frames} frames - {avg_time:.1f}ms per frame")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Performance summary
        avg_processing_time = np.mean(processing_times)
        fps_achieved = 1000 / avg_processing_time if avg_processing_time > 0 else 0
        
        print(f"\n✅ Video processing complete!")
        print(f"📊 Performance:")
        print(f"   - Average processing time: {avg_processing_time:.1f}ms")
        print(f"   - Achieved FPS: {fps_achieved:.1f}")
        print(f"   - Output saved to: {output_path}")
        
        return output_path

# Example usage
if __name__ == "__main__":
    # This would be used after we have a video file
    processor = VideoProcessor()
    
    # Example usage (commented out as we need actual video files)
    # processor.process_video("input.mp4", "output_with_pose.mp4")
    
    print("💡 To use this class:")
    print("1. Upload a video file to Colab")
    print("2. Run: processor.process_video('your_video.mp4', 'output.mp4')")
