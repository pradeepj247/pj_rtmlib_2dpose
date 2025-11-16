
"""
optimized_pose_estimation.py
Fast pose estimation using YOLO detection + single person pose
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys

# Add the parent directory to Python path to import pjpose2d modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmlib import Body

def optimized_pose_estimation():
    print("🚀 OPTIMIZED Pose Estimation (YOLO + Single Person)")
    print("=" * 50)
    
    # Load image
    img_path = "data/images/demo.jpg"
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return
    
    img = cv2.imread(img_path)
    print(f"📷 Image loaded: {img.shape}")
    
    # --------------- Step 1: YOLO Detection ---------------
    print("\n1. Running YOLOv8 person detection...")
    
    # Use PyTorch YOLO (much faster than ONNX for detection)
    detector = YOLO('yolov8s.pt')  # This will auto-download if needed
    
    # Warm-up
    _ = detector(img, classes=[0])  # classes=[0] for person only
    
    # Time detection
    t0 = time.time()
    results = detector(img, classes=[0])
    t1 = time.time()
    
    det_time = (t1 - t0) * 1000
    print(f"✅ Detection time: {det_time:.2f} ms")
    
    # Extract bounding boxes
    boxes = results[0].boxes.xyxy.cpu().numpy()
    print(f"✅ Detected {len(boxes)} persons")
    
    if len(boxes) == 0:
        print("❌ No persons detected!")
        return
    
    # --------------- Step 2: Pick largest person ---------------
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_idx = np.argmax(areas)
    x1, y1, x2, y2 = map(int, boxes[largest_idx])
    
    print(f"✅ Largest person box: ({x1}, {y1}, {x2}, {y2})")
    print(f"✅ Bbox area: {areas[largest_idx]:.0f} pixels")
    
    # Crop to largest person
    crop = img[y1:y2, x1:x2]
    print(f"✅ Cropped region: {crop.shape}")
    
    # --------------- Step 3: Pose Estimation on Single Person ---------------
    print("\n2. Running pose estimation on single person...")
    
    # Initialize pose estimator
    body = Body(to_openpose=True, mode="balanced", backend="onnxruntime")
    
    # Warm-up
    _ = body.pose_model(crop)
    
    # Time pose estimation (multiple runs for accurate timing)
    num_runs = 10
    t0 = time.time()
    for _ in range(num_runs):
        keypoints, scores = body.pose_model(crop)
    t1 = time.time()
    
    avg_pose_time = (t1 - t0) / num_runs * 1000
    fps_pose = 1000 / avg_pose_time
    
    print(f"✅ Pose time (single person): {avg_pose_time:.2f} ms")
    print(f"✅ Pose FPS: {fps_pose:.2f}")
    
    # --------------- Step 4: Total Processing ---------------
    total_time = det_time + avg_pose_time
    total_fps = 1000 / total_time
    
    print(f"\n📊 TOTAL PERFORMANCE:")
    print(f"   - Detection: {det_time:.2f} ms")
    print(f"   - Pose: {avg_pose_time:.2f} ms") 
    print(f"   - TOTAL: {total_time:.2f} ms per frame")
    print(f"   - TOTAL FPS: {total_fps:.2f}")
    print(f"   - Speedup vs full-image: {1969/total_time:.1f}x faster")  # vs our previous 1969ms
    
    # --------------- Step 5: Visualization ---------------
    print("\n3. Generating visualization...")
    
    from rtmlib import draw_skeleton
    
    # Draw on cropped image
    result_crop = draw_skeleton(crop, keypoints, scores, openpose_skeleton=True, kpt_thr=0.3)
    
    # Paste back to original image
    result_img = img.copy()
    result_img[y1:y2, x1:x2] = result_crop
    
    # Draw bounding box
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(result_img, f"FPS: {total_fps:.1f}", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save result
    os.makedirs("optimized_results", exist_ok=True)
    output_path = "optimized_results/optimized_pose_result.jpg"
    cv2.imwrite(output_path, result_img)
    print(f"✅ Result saved: {output_path}")
    
    return total_fps

if __name__ == "__main__":
    optimized_pose_estimation()
