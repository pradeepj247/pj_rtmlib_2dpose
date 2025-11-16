
# PJPOSE2D OPTIMIZATION TODO LIST
# ================================
# This file tracks all optimization points and improvements identified during development
# Execute these items systematically to improve performance and features

## TODO ITEMS:

### POINT #1: Replace ONNX YOLO with PyTorch YOLO for detection
- STATUS: IDENTIFIED ✅
- PROBLEM: ONNX YOLO refuses to use CUDA, runs on CPU (400ms)
- SOLUTION: Use PyTorch YOLO which automatically uses GPU (22ms)  
- ACTION: Update all detection pipelines to use PyTorch YOLO
- IMPACT: 18x speedup in detection
- FILES AFFECTED: 
  - examples/video_pose_estimation.py
  - examples/optimized_pose_estimation.py
  - Any future detection scripts

### POINT #2: Optimize video processing pipeline
- STATUS: PENDING
- PROBLEM: Current video processing still uses full-frame pose estimation
- SOLUTION: Apply YOLO detection + single person pose to video processing
- ACTION: Create optimized_video_pose.py using the same approach as optimized_pose_estimation.py
- EXPECTED IMPACT: 10-20x faster video processing
- PRIORITY: HIGH

### POINT #3: Fix ONNX Runtime CUDA setup for pose models
- STATUS: INVESTIGATING  
- PROBLEM: RTMPose ONNX models not using CUDAExecutionProvider
- SOLUTION: Investigate model export options or alternative backends
- ACTION: Test PyTorch RTMPose if available, or fix ONNX CUDA setup
- PRIORITY: MEDIUM

### POINT #4: Add batch processing for multiple persons
- STATUS: FUTURE
- PROBLEM: Current single-person approach misses other people in frame
- SOLUTION: Implement batch processing for top N persons
- ACTION: Modify pipeline to process 2-3 largest persons per frame
- PRIORITY: LOW

### POINT #5: Create performance benchmarking suite
- STATUS: PENDING
- PROBLEM: No standardized way to measure performance improvements
- SOLUTION: Create benchmark scripts for FPS, accuracy, memory usage
- ACTION: Create benchmarks/ directory with standardized tests
- PRIORITY: MEDIUM

## COMPLETED ITEMS:
- ✅ Fixed NumPy compatibility issue (downgraded to 1.26.4)
- ✅ Installed correct ONNX Runtime GPU version
- ✅ Created optimized single-person pose estimation (23x speedup)

## EXECUTION NOTES:
- Run items in priority order
- Test each optimization thoroughly before moving to next
- Update this file with progress and new findings
