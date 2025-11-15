# RTMPose 2D Pose Estimation on Google Colab

**Complete all-in-one package** - Includes RTMLib source + ready-to-use pose estimation pipeline.

## 🚀 One-Command Setup
```bash
git clone https://github.com/pradeepj247/pj_rtmlib_2dpose.git
cd pj_rtmlib_2dpose
bash setup_colab.sh
```

## 📦 What's Included
- ✅ **RTMLib source code** (full library - no external download needed)
- ✅ **Pose estimation wrapper classes** (`pose_estimation.py`, `video_processor.py`)
- ✅ **Video processing pipeline**
- ✅ **Single image demo** (`examples/single_image_demo.py`)
- ✅ **Complete setup scripts**

No external dependencies needed during setup - everything is self-contained!

## 🎯 Quick Start
```python
from pose_estimation import PoseEstimator

# Initialize and use
estimator = PoseEstimator(device=\"cuda\")
result = estimator.process_image(\"rtmlib/demo.jpg\")
```

## 📊 Performance
- **Detection + Pose**: ~39ms per frame
- **Real-time**: ~25 FPS
- **GPU**: Tesla T4 (Colab)

## 🛠 Project Structure
```
pj_rtmlib_2dpose/
├── rtmlib/                 # Complete RTMLib source
├── pose_estimation.py      # Core pose estimation class
├── video_processor.py      # Video processing pipeline
├── setup_colab.sh          # One-click setup script
├── requirements.txt        # Python dependencies
└── examples/
    └── single_image_demo.py
```

## 📝 License
Includes RTMLib under its original license.
