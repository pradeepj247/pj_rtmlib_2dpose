"""
Video Pipeline with GPU Verification
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from rtmlib import Body, draw_skeleton
import torch

def process_video_gpu_verified():
    print("🚀 Video Pipeline - GPU Verified")
    print("=" * 40)
    
    # Check GPU status
    print("🔍 Hardware Check:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU device: {torch.cuda.get_device_name()}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize models with explicit device settings
    print("\n🔄 Initializing models...")
    detector = YOLO('models/yolov8s.pt')
    pose_estimator = Body(
        to_openpose=True,
        mode='balanced',
        backend='onnxruntime', 
        device='cuda'
    )
    
    # Verify pose model is using GPU
    pose_providers = pose_estimator.pose_model.session.get_providers()
    print(f"   Pose model providers: {pose_providers}")
    
    # Video paths
    input_video = "data/videos/sample_video.mp4"
    output_video = "fresh_examples/video_gpu_verified.mp4"
    
    # Open input
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_video}")
        return
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    
    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Process frames
    frame_times = []
    frame_count = 0
    last_print = 0
    print_interval = 30
    
    print("🎬 Processing video frames...")
    start_total = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        try:
            # Detection (silent)
            results = detector(frame, classes=[0], verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            if len(boxes) > 0:
                # Largest person
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                largest_idx = np.argmax(areas)
                x1, y1, x2, y2 = map(int, boxes[largest_idx])
                
                if y2 > y1 and x2 > x1:
                    crop = frame[y1:y2, x1:x2]
                    
                    # Pose estimation
                    keypoints, scores = pose_estimator.pose_model(crop)
                    
                    # Adjust coordinates
                    keypoints[:, :, 0] += x1
                    keypoints[:, :, 1] += y1
                    
                    # Draw skeleton
                    result_frame = draw_skeleton(
                        frame.copy(), 
                        keypoints, 
                        scores, 
                        openpose_skeleton=True, 
                        kpt_thr=0.3
                    )
                    
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                else:
                    result_frame = frame
            else:
                result_frame = frame
                
            out.write(result_frame)
            frame_count += 1
            
        except Exception:
            out.write(frame)
            frame_count += 1
            
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        
        if frame_count - last_print >= print_interval:
            elapsed = time.time() - start_total
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"   Frame {frame_count}/{total_frames} - FPS: {avg_fps:.1f}")
            last_print = frame_count
    
    # Cleanup
    cap.release()
    out.release()
    total_time = time.time() - start_total
    
    # Final stats
    if frame_times:
        avg_time = np.mean(frame_times) * 1000
        avg_fps = 1000 / avg_time
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Frames: {frame_count}/{total_frames}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Avg FPS: {avg_fps:.1f}")
        print(f"   Real-time: {avg_fps/fps:.2f}x video speed")
        print(f"   Output: {output_video}")

if __name__ == "__main__":
    process_video_gpu_verified()
