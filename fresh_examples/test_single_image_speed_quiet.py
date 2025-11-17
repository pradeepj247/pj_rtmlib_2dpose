"""
Quiet Speed Test - Performance without verbose output
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from rtmlib import Body

def test_speed_quiet():
    print("🚀 Quiet Speed Performance Test")
    print("=" * 40)
    
    # Initialize models
    detector = YOLO('models/yolov8s.pt')
    pose_estimator = Body(
        to_openpose=True,
        mode='balanced', 
        backend='onnxruntime',
        device='cuda'
    )
    
    # Load image
    img = cv2.imread("data/images/demo.jpg")
    if img is None:
        print("❌ Could not load image")
        return
    
    print(f"📷 Image: {img.shape}")
    
    # WARM-UP (silent)
    print("🔥 Warming up...")
    _ = detector(img, classes=[0], verbose=False)
    dummy_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    _ = pose_estimator.pose_model(dummy_crop)
    
    # SPEED TEST
    num_runs = 20
    detection_times = []
    pose_times = []
    
    print(f"🔄 Running {num_runs} iterations...")
    
    for i in range(num_runs):
        # Detection (silent)
        t0 = time.time()
        results = detector(img, classes=[0], verbose=False)
        t1 = time.time()
        detection_times.append((t1 - t0) * 1000)
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_idx = np.argmax(areas)
            x1, y1, x2, y2 = map(int, boxes[largest_idx])
            crop = img[y1:y2, x1:x2]
            
            # Pose timing
            t0 = time.time()
            _ = pose_estimator.pose_model(crop)
            t1 = time.time()
            pose_times.append((t1 - t0) * 1000)
    
    # Results
    avg_detection = np.mean(detection_times)
    avg_pose = np.mean(pose_times)
    total_time = avg_detection + avg_pose
    
    print(f"\n📊 PERFORMANCE ({num_runs} runs):")
    print(f"   Detection: {avg_detection:.1f} ms ({1000/avg_detection:.1f} FPS)")
    print(f"   Pose:      {avg_pose:.1f} ms ({1000/avg_pose:.1f} FPS)")
    print(f"   Total:     {total_time:.1f} ms ({1000/total_time:.1f} FPS)")

if __name__ == "__main__":
    test_speed_quiet()
