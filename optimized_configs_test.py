
import cv2
import time
import numpy as np
from rtmlib import Body
from ultralytics import YOLO

backend = 'onnxruntime'
device = 'cuda'

# Test different optimized configurations
configs = [
    {
        'name': 'Pose-only with URL',
        'params': {
            'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
            'pose_input_size': (192, 256),
            'backend': backend,
            'device': device
        }
    },
    {
        'name': 'Pose-only with model name', 
        'params': {
            'pose': 'rtmpose-m',
            'pose_input_size': (192, 256),
            'backend': backend,
            'device': device
        }
    },
    {
        'name': 'Minimal pose-only',
        'params': {
            'pose': 'rtmpose-m',
            'backend': backend,
            'device': device
            # Using default pose_input_size=(288, 384)
        }
    },
    {
        'name': 'Default with GPU',
        'params': {
            'backend': backend,
            'device': device
            # Using default models and sizes
        }
    }
]

# Load test image and get crop once
img = cv2.imread("/content/pjpose2d/data/images/demo.jpg")
detector = YOLO('yolov8s.pt')
results = detector(img, classes=[0])
boxes = results[0].boxes.xyxy.cpu().numpy()

if len(boxes) > 0:
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_idx = np.argmax(areas)
    x1, y1, x2, y2 = map(int, boxes[largest_idx])
    crop = img[y1:y2, x1:x2]
    print(f"Test crop size: {crop.shape}")
else:
    print("❌ No persons detected")
    crop = img  # Fallback to full image

print("\\n" + "="*50)

for config in configs:
    print(f"\\n🧪 Testing: {config['name']}")
    print(f"   Parameters: {config['params']}")
    
    try:
        body = Body(**config['params'])
        
        # Check providers
        if hasattr(body, 'pose_model') and hasattr(body.pose_model, 'session'):
            print(f"   ✅ Pose providers: {body.pose_model.session.get_providers()[0]}")
        if hasattr(body, 'det_model') and hasattr(body.det_model, 'session'):
            print(f"   ⚠️  Det providers: {body.det_model.session.get_providers()[0]} (not used)")
        
        # Performance test
        num_runs = 15
        t0 = time.time()
        for _ in range(num_runs):
            _ = body.pose_model(crop)
        t1 = time.time()
        
        avg_pose_time = (t1 - t0) / num_runs * 1000
        fps_pose = 1000 / avg_pose_time
        
        print(f"   ⚡ Pose FPS: {fps_pose:.2f} ({avg_pose_time:.2f}ms)")
        
        del body
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
