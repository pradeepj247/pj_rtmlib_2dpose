
"""
Core pose estimation class for RTMPose + YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
from rtmlib.rtmlib import Body, draw_skeleton
import time

class PoseEstimator:
    def __init__(self, device="cuda"):
        """
        Initialize pose estimation pipeline
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.yolo_model = None
        self.pose_model = None
        self.initialized = False
        
    def initialize_models(self):
        """Initialize YOLO and RTMPose models"""
        if self.initialized:
            return
            
        print("🔄 Initializing models...")
        
        # Initialize YOLOv8 for person detection
        self.yolo_model = YOLO("yolov8s.pt").to(self.device)
        
        # Initialize RTMPose-M for pose estimation
        self.pose_model = Body(
            det="rtmlib/yolov8s.onnx",
            det_input_size=(640, 640),
            pose="rtmpose-m",  # Auto-downloads RTMPose-M
            pose_input_size=(192, 256),
            backend="onnxruntime",
            device=self.device
        )
        
        self.initialized = True
        print("✅ Models initialized successfully!")
    
    def detect_persons(self, image):
        """
        Detect persons in image using YOLOv8
        
        Returns:
            boxes: List of bounding boxes [x1, y1, x2, y2]
            crop: Largest person crop
            bbox: Largest person bbox coordinates
        """
        results = self.yolo_model(image, classes=[0])[0]
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        
        if len(boxes) == 0:
            return [], None, None
            
        # Get largest person by area
        areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
        largest_idx = areas.argmax()
        x1, y1, x2, y2 = boxes[largest_idx]
        crop = image[y1:y2, x1:x2]
        
        return boxes, crop, (x1, y1, x2, y2)
    
    def estimate_pose(self, crop):
        """Estimate pose for a person crop"""
        if crop is None or crop.size == 0:
            return None, None
            
        keypoints, scores = self.pose_model.pose_model(crop)
        return keypoints, scores
    
    def draw_results(self, image, crop, bbox, keypoints, scores, kpt_thr=0.5):
        """Draw pose skeleton on image"""
        if crop is None or keypoints is None:
            return image
            
        # Draw on crop
        crop_with_pose = draw_skeleton(crop.copy(), keypoints, scores, kpt_thr=kpt_thr)
        
        # Overlay back to original image
        x1, y1, x2, y2 = bbox
        image_with_pose = image.copy()
        image_with_pose[y1:y2, x1:x2] = cv2.resize(crop_with_pose, (x2 - x1, y2 - y1))
        
        return image_with_pose
    
    def process_image(self, image_path, output_path=None, display=True):
        """
        Process single image for pose estimation
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            display: Whether to display result
        """
        if not self.initialized:
            self.initialize_models()
            
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"📷 Processing image: {image_path}")
        
        # Detect persons
        boxes, crop, bbox = self.detect_persons(image)
        
        if crop is None:
            print("❌ No persons detected")
            return image
            
        # Estimate pose
        keypoints, scores = self.estimate_pose(crop)
        
        if keypoints is None:
            print("❌ Pose estimation failed")
            return image
            
        # Draw results
        result_image = self.draw_results(image, crop, bbox, keypoints, scores)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"💾 Result saved to: {output_path}")
        
        # Display if requested
        if display:
            from google.colab.patches import cv2_imshow
            cv2_imshow(result_image)
            
        return result_image

# Example usage
if __name__ == "__main__":
    estimator = PoseEstimator()
    result = estimator.process_image("rtmlib/demo.jpg")
