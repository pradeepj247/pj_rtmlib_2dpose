
import cv2
import time
import numpy as np
from rtmlib import Body
from ultralytics import YOLO

def proper_ultimate_pose_estimation(image_path, output_path="proper_ultimate_result.jpg"):
    """
    PROPER Ultimate Pipeline using ALL your proven optimizations:
    1. Explicit warm-up before timing
    2. Multiple inference runs for accurate FPS
    3. Process only largest person crop
    4. Same image used for all runs
    """
    print("🚀 PROPER ULTIMATE POSE ESTIMATION (Your Proven Approach)")
    print("=" * 60)
    
    # Load image ONCE
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"📷 Input image: {img.shape}")
    
    # 1. Initialize models
    print("\\n1. Initializing models...")
    
    # Fast detection model
    detector = YOLO('yolov8s.pt')
    
    # Ultimate pose model configuration
    pose_estimator = Body(
        backend='onnxruntime',
        device='cuda'
    )
    
    print(f"✅ Detector: YOLOv8s (PyTorch GPU)")
    print(f"✅ Pose model: {pose_estimator.pose_model.session.get_providers()[0]}")
    
    # 2. DETECTION: Your proven approach
    print("\\n2. Person Detection (Your Method)...")
    
    # WARM-UP detection
    _ = detector(img, classes=[0])
    
    # TIMING detection (single run as in your approach)
    t0 = time.time()
    results = detector(img, classes=[0])
    t1 = time.time()
    
    det_time = (t1 - t0) * 1000
    print(f"✅ Detection time: {det_time:.2f} ms")
    
    # Extract bounding boxes
    boxes = results[0].boxes.xyxy.cpu().numpy()
    print(f"✅ Persons detected: {len(boxes)}")
    
    if len(boxes) == 0:
        raise ValueError("No person detected in the image!")
    
    # 3. Pick LARGEST bbox only (as in your approach)
    print("\\n3. Selecting largest person...")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_idx = np.argmax(areas)
    x1, y1, x2, y2 = map(int, boxes[largest_idx])
    print(f"✅ Largest person box: ({x1}, {y1}, {x2}, {y2})")
    
    crop = img[y1:y2, x1:x2]
    print(f"✅ Crop size: {crop.shape}")
    
    # 4. POSE ESTIMATION: Your proven approach
    print("\\n4. Pose Estimation (Your Method)...")
    
    # WARM-UP pose (as in your approach)
    _ = pose_estimator.pose_model(crop)
    
    # TIMING with MULTIPLE RUNS for accurate FPS (as in your approach)
    num_runs = 20
    print(f"Running pose estimation {num_runs} times for accurate timing...")
    
    t0 = time.time()
    for _ in range(num_runs):
        _ = pose_estimator.pose_model(crop)  # Same crop, multiple runs
    t1 = time.time()
    
    avg_pose_time = (t1 - t0) / num_runs * 1000
    fps_pose = 1000 / avg_pose_time
    
    print(f"✅ Pose estimation time: {avg_pose_time:.2f} ms/image")
    print(f"✅ Pose estimation FPS: {fps_pose:.2f}")
    
    # 5. Run once for visualization
    print("\\n5. Running for visualization...")
    keypoints, scores = pose_estimator.pose_model(crop)
    
    # Adjust keypoints to original image coordinates
    keypoints[:, :, 0] += x1
    keypoints[:, :, 1] += y1
    
    # 6. Simple visualization
    vis_img = img.copy()
    
    # Draw bounding box
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Draw keypoints
    for kpt, score in zip(keypoints[0], scores[0]):
        if score > 0.3:
            x, y = int(kpt[0]), int(kpt[1])
            cv2.circle(vis_img, (x, y), 4, (0, 255, 0), -1)
    
    cv2.imwrite(output_path, vis_img)
    print(f"✅ Visualization saved: {output_path}")
    
    # 7. Performance summary
    total_time = det_time + avg_pose_time
    total_fps = 1000 / total_time
    
    print(f"\\n🎯 PERFORMANCE SUMMARY (Your Method):")
    print(f"   Detection: {1000/det_time:.1f} FPS")
    print(f"   Pose: {fps_pose:.1f} FPS")
    print(f"   Combined: {total_fps:.1f} FPS")
    print(f"   Persons processed: 1 (largest only)")
    
    return total_fps, fps_pose

# Run the PROPER pipeline
if __name__ == "__main__":
    print("🔥 RUNNING PROPER ULTIMATE PIPELINE (Your Method)")
    print("This follows EXACTLY your proven approach:")
    print("1. Warm-up before timing")
    print("2. Multiple runs for accurate FPS") 
    print("3. Largest person only")
    print("4. Same image for all runs")
    print("=" * 60)
    
    total_fps, pose_fps = proper_ultimate_pose_estimation(
        image_path="/content/pjpose2d/data/images/demo.jpg",
        output_path="proper_ultimate_result.jpg"
    )
    
    print(f"\\n🏆 FINAL RESULT USING YOUR METHOD:")
    print(f"   Total Pipeline FPS: {total_fps:.1f}")
    print(f"   Pose-Only FPS: {pose_fps:.1f}")
    
    if pose_fps >= 50:
        print("🎉 SUCCESS! We achieved 50+ FPS for pose estimation!")
    else:
        print("⚠️  Close! Let's see if we can optimize further...")
