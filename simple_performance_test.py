
import cv2
import numpy as np
from rtmlib import Body, draw_skeleton
import time

print("🚀 SIMPLE & ELEGANT POSE ESTIMATION TEST")
print("=" * 50)

# Load test image
img = cv2.imread("data/images/demo.jpg")
if img is None:
    img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    print("⚠️  Using dummy image")
else:
    print(f"✅ Loaded image: {img.shape}")

# Initialize Body the simple way (as in test_pose_estimation.py)
print("\\n1. Initializing Body estimator...")
body_estimator = Body(to_openpose=True, mode="balanced", backend="onnxruntime")
print("✅ Body initialized successfully")

# Test performance with multiple runs
print("\\n2. Performance testing...")

# Warm up
_ = body_estimator(img)

# Benchmark with multiple runs
num_runs = 10
times = []

for i in range(num_runs):
    start_time = time.time()
    keypoints, scores = body_estimator(img)
    end_time = time.time()
    times.append((end_time - start_time) * 1000)
    
    if i == 0:  # Print details for first run
        print(f"   Run {i+1}:")
        print(f"     - Persons detected: {keypoints.shape[0]}")
        print(f"     - Keypoints per person: {keypoints.shape[1]}")
        print(f"     - Inference time: {times[-1]:.1f}ms")

# Calculate statistics
avg_time = np.mean(times)
max_time = np.max(times) 
min_time = np.min(times)
fps = 1000 / avg_time

print(f"\\n3. PERFORMANCE RESULTS ({num_runs} runs):")
print(f"   Average time: {avg_time:.1f}ms")
print(f"   Best time: {min_time:.1f}ms")
print(f"   Worst time: {max_time:.1f}ms")
print(f"   FPS: {fps:.1f}")
print(f"   Performance: {'✅ EXCELLENT' if fps > 30 else '⚠️ NEEDS OPTIMIZATION' if fps > 10 else '❌ POOR'}")

# Test visualization
print("\\n4. Testing visualization...")
try:
    img_show = img.copy()
    vis_img = draw_skeleton(
        img_show,
        keypoints,
        scores,
        openpose_skeleton=True,
        kpt_thr=0.5
    )
    cv2.imwrite("performance_test_result.jpg", vis_img)
    print("✅ Visualization saved: performance_test_result.jpg")
except Exception as e:
    print(f"⚠️  Visualization failed: {e}")
