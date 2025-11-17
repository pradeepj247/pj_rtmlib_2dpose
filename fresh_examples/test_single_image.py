"""
Test Single Image Pose Estimation
Uses YOLOv8s for detection + RTMPose for 2D pose + Visualization
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from rtmlib import Body, draw_skeleton

def test_single_image():
    print("🚀 Testing Single Image Pose Estimation")
    print("=" * 50)
    
    # Initialize models
    detector = YOLO('models/yolov8s.pt')
    pose_estimator = Body(
        to_openpose=True, 
        mode='balanced', 
        backend='onnxruntime',
        device='cuda'
    )
    
    # Load image
    img_path = "data/images/demo.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not load image: {img_path}")
        return
    
    print(f"📷 Input image: {img.shape}")
    
    # Detection
    results = detector(img, classes=[0])
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    if len(boxes) == 0:
        print("❌ No persons detected")
        return
    
    # Get largest person
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_idx = np.argmax(areas)
    x1, y1, x2, y2 = map(int, boxes[largest_idx])
    crop = img[y1:y2, x1:x2]
    
    print(f"✅ Largest person: ({x1}, {y1}, {x2}, {y2})")
    
    # Pose estimation
    keypoints, scores = pose_estimator.pose_model(crop)
    
    # Adjust coordinates
    keypoints[:, :, 0] += x1
    keypoints[:, :, 1] += y1
    
    # Visualization
    result_img = draw_skeleton(
        img.copy(), 
        keypoints, 
        scores, 
        openpose_skeleton=True, 
        kpt_thr=0.3
    )
    
    # Draw bounding box
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Save result
    output_path = "fresh_examples/single_image_result.jpg"
    cv2.imwrite(output_path, result_img)
    print(f"✅ Result saved: {output_path}")
    
    print(f"🎯 Persons detected: {len(boxes)}")
    print(f"🎯 Keypoints per person: {keypoints.shape[1]}")

if __name__ == "__main__":
    test_single_image()
