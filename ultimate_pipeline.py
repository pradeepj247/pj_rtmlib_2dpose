
import cv2
import time
import numpy as np
from rtmlib import Body, draw_skeleton
from ultralytics import YOLO

def ultimate_pose_estimation(image_path, output_path="ultimate_result.jpg"):
    """
    ULTIMATE Optimized Pose Estimation Pipeline
    Uses the winning configuration: Body(backend='onnxruntime', device='cuda')
    """
    print("🚀 ULTIMATE OPTIMIZED POSE ESTIMATION")
    print("=" * 50)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"📷 Input image: {img.shape}")
    
    # 1. Initialize models (WINNING CONFIGURATION)
    print("\\n1. Initializing models...")
    
    # Fast detection model
    detector = YOLO('yolov8s.pt')
    
    # ULTIMATE pose model configuration
    pose_estimator = Body(
        backend='onnxruntime',
        device='cuda'  # ← THIS IS THE MAGIC!
    )
    
    print(f"✅ Detector: YOLOv8s (PyTorch GPU)")
    print(f"✅ Pose model: {pose_estimator.pose_model.session.get_providers()[0]}")
    print(f"✅ Det model: {pose_estimator.det_model.session.get_providers()[0]} (not used)")
    
    # 2. Person detection
    print("\\n2. Person detection...")
    t0 = time.time()
    results = detector(img, classes=[0])  # Only detect persons
    det_time = (time.time() - t0) * 1000
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    print(f"✅ Persons detected: {len(boxes)}")
    print(f"⏱️  Detection time: {det_time:.1f}ms ({1000/det_time:.1f} FPS)")
    
    if len(boxes) == 0:
        print("⚠️  No persons found, using full image")
        crops = [img]
        bboxes = [(0, 0, img.shape[1], img.shape[0])]
    else:
        # Get all persons
        crops = []
        bboxes = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]
            # Only process reasonably sized crops
            if crop.shape[0] > 10 and crop.shape[1] > 10:
                crops.append(crop)
                bboxes.append((x1, y1, x2, y2))
                if i < 3:  # Show first 3
                    print(f"   Person {i+1}: {crop.shape}")
        
        print(f"   Total persons to process: {len(crops)}")
    
    # 3. Pose estimation on crops
    print("\\n3. Pose estimation...")
    all_keypoints = []
    all_scores = []
    
    pose_times = []
    for i, crop in enumerate(crops):
        t0 = time.time()
        keypoints, scores = pose_estimator.pose_model(crop)
        pose_time = (time.time() - t0) * 1000
        pose_times.append(pose_time)
        
        # Adjust keypoints to original image coordinates
        x1, y1, x2, y2 = bboxes[i]
        keypoints[:, :, 0] += x1  # x coordinates
        keypoints[:, :, 1] += y1  # y coordinates
        
        all_keypoints.append(keypoints)
        all_scores.append(scores)
        
        if i < 3:  # Show first 3
            print(f"   Person {i+1}: {pose_time:.1f}ms")
    
    if pose_times:
        avg_pose_time = np.mean(pose_times)
        max_pose_time = np.max(pose_times)
        min_pose_time = np.min(pose_times)
        print(f"⏱️  Pose stats: {min_pose_time:.1f}-{max_pose_time:.1f}ms (avg: {avg_pose_time:.1f}ms)")
        print(f"⚡ Pose FPS: {1000/avg_pose_time:.1f}")
    
    # 4. Combine results and visualize
    print("\\n4. Visualization...")
    if all_keypoints:
        combined_keypoints = np.concatenate(all_keypoints, axis=0)
        combined_scores = np.concatenate(all_scores, axis=0)
        
        vis_img = draw_skeleton(
            img.copy(),
            combined_keypoints,
            combined_scores,
            openpose_skeleton=True,
            kpt_thr=0.3
        )
        
        cv2.imwrite(output_path, vis_img)
        print(f"✅ Result saved: {output_path}")
        
        # Total performance
        total_time = det_time + np.sum(pose_times)
        total_fps = 1000 / total_time if total_time > 0 else 0
        
        print(f"\\n🎯 ULTIMATE PERFORMANCE:")
        print(f"   Detection: {1000/det_time:.1f} FPS")
        print(f"   Pose (per person): {1000/avg_pose_time:.1f} FPS")
        print(f"   Total pipeline: {total_fps:.1f} FPS")
        print(f"   Persons processed: {len(crops)}")
        print(f"   Total time: {total_time:.1f}ms")
        
        return total_fps, len(crops)
    else:
        print("❌ No pose estimation results")
        return 0, 0

# Run the ULTIMATE pipeline
if __name__ == "__main__":
    print("🔥 RUNNING ULTIMATE OPTIMIZED PIPELINE")
    fps, persons = ultimate_pose_estimation(
        image_path="/content/pjpose2d/data/images/demo.jpg",
        output_path="ultimate_optimized_result.jpg"
    )
    
    print(f"\\n🏆 FINAL RESULT: {fps:.1f} FPS for {persons} persons")
