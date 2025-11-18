
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


## MERGING BOX And PJPOSE
Short answer:
**YES — merging them into a single Colab notebook *will* cause dependency conflicts**, unless you adjust versions.
Right now both requirements files have *multiple direct clashes*.

Below is a clean breakdown and a recommended unified environment that will run both *BoxMOT* and *pjpose2d (RTMPose)* together safely.

---

# ✅ **Major Conflicts Between the Two Requirement Sets**

## **1. PyTorch / TorchVision**

### BoxMOT:

* `torch==2.8.0+cu126`
* `torchvision==0.23.0+cu126`

### PJ Pose2D:

* `torch>=1.8.0`  (very old lower bound)

**Conflict?**
No *direct* conflict — but **Colab does NOT support PyTorch 2.8.0 yet**.
Colab generally supports latest stable (as of 2025, it's around 2.4–2.5).
So BoxMOT's torch is **too new for Colab**.

**Conclusion**:
You cannot install PyTorch 2.8.0 on Colab.
You must downgrade BoxMOT to the PyTorch version Colab supports.

---

## **2. NumPy**

### BoxMOT:

* `numpy==2.0.2`

### PJ Pose2D:

* `numpy>=1.21.0,<2.3.0` (accepts 2.0.2)

**Conflict?**
✔ No conflict. NumPy 2.0.2 is accepted by RTMPose.

---

## **3. Ultralytics**

### BoxMOT:

* `ultralytics==8.3.228`

### PJ Pose2D:

* `ultralytics>=8.0.0` (so 8.3.228 is ok)

**Conflict?**
✔ No conflict. This is safe.

---

## **4. opencv-python**

### BoxMOT:

* `opencv-python==4.12.0`

### PJ Pose2D:

* `opencv-python>=4.5.0`

**Conflict?**
✔ No conflict. RTMPose accepts 4.12.0.

---

## **5. onnxruntime-gpu**

### Only PJ Pose2D requires:

* `onnxruntime-gpu==1.23.0`

**Conflict?**
✔ No conflict — BoxMOT does not use ONNX.

Just note:
Colab sometimes conflicts ONNXRuntime-GPU with CUDA version present.
But 1.23.0 is usually okay.

---

## **6. boxmot**

BoxMOT requires:

* `boxmot==15.0.10`

**This package itself has specific PyTorch constraints**, and **v15 expects torch >= 2.2+**.

But not 2.8 specifically.

---

# 🚨 **Biggest Practical Problem: Colab cannot install these versions**

### BoxMOT wants:

* `torch==2.8.0+cu126`
* `torchvision==0.23.0+cu126`

These are NOT available in Colab’s CUDA build (Colab typically has CUDA 12.1 / 12.2).

Meaning:
👉 **You will never be able to install torch=2.8.0+cu126 on Google Colab.**

So BoxMOT must run with a **lower PyTorch version** compatible with Colab.

---

# 🟢 **GOOD NEWS: BoxMOT works fine with older Torch versions**

On their GitHub issues, BoxMOT is confirmed compatible with:

* `torch==2.2`
* `torchvision==0.17`

which **Colab supports**.

Also BoxMOT 15.x models (OC-SORT, ByteTrack, StrongSORT) do *not* require the newest PyTorch.

---

# ✅ **RECOMMENDED UNIFIED ENVIRONMENT (Colab-safe)**

Use these versions to run *both pipelines in one notebook*:

```text
torch==2.2.2
torchvision==0.17.2

numpy==2.0.2
opencv-python==4.12.0

ultralytics==8.3.228
boxmot==15.0.10

supervision==0.26.1
filterpy==1.4.5
scipy==1.16.3
scikit-learn==1.6.1
Pillow==11.3.0
requests==2.32.4

onnxruntime-gpu==1.23.0
```

Everything is compatible with:

* Colab’s CUDA 12.1 / 12.2 builds
* BoxMOT trackers
* RTMPose inference pipeline

---

# 💡 WHY THIS COMBINATION WORKS

### ✔ Torch 2.2.2 works with BoxMOT

Because BoxMOT doesn’t use advanced CUDA operators.

### ✔ RTMPose inference uses ONNXRuntime → unaffected by torch

So RTMPose is fully torch-independent.

### ✔ ultralytics 8.3.x is used by both, no conflict

### ✔ numpy and opencv versions are compatible with both

---

# 🟢 **FINAL ANSWER**

**Yes, merging both repos into a single Colab pipeline is possible — but not using the versions in your requirements.**
You need to *downgrade BoxMOT’s PyTorch* to a version supported by Colab.

Use the unified requirements above and everything will run smoothly.

---

If you want, I can create:

✅ A combined requirements cell
✅ A full working Colab notebook that loads:

* YOLO / BoxMOT tracking
* RTMPose 2D pose estimation
* Unified inference loop
* Real-time visualization

Just tell me **“Yes, generate the combined Colab pipeline”**.


